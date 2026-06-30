"""MemNav backbone pieces.

Two groups live here:

1. Reused diffusion building blocks (`SinusoidalPosEmb`, `LearnablePositionalEncoding`,
   `TokenCompressor`) — copied verbatim from NavDP/LoGoPlanner so the diffusion decoder in
   `policy_network.py` is a drop-in match. Keep these identical to
   `NavDP/baselines/navdp/policy_backbone.py`.

2. `LingBotStream` — a *frozen* live wrapper around LingBot-Map's `GCTStream`. It maintains the
   KV cache across `step()` calls (streaming-causal, exactly like the offline precompute) and
   exposes per-frame features for the policy. This is the inference-time analogue of
   `InternNav/scripts/dataset_converters/precompute_lingbot_features.py`.
"""

import math
import sys

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Reused diffusion blocks (verbatim from navdp/logoplanner policy_backbone.py)
# --------------------------------------------------------------------------- #
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_encoding = self.position_embedding(position_ids)
        return position_encoding


class TokenCompressor(nn.Module):
    def __init__(self, embed_dim, num_heads, target_length):
        super(TokenCompressor, self).__init__()
        self.target_length = target_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.target_embedding = nn.Embedding(target_length, embed_dim)
        self.positional_encoding = SinusoidalPosEmb(embed_dim)
        self.token_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.query_positional_encoding = LearnablePositionalEncoding(embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x, padding_mask=None):
        bs, token_len, _ = x.shape
        token_pe = self.token_positional_encoding(x)
        x = x + token_pe
        query = self.target_embedding.weight.unsqueeze(0).expand(bs, -1, -1)
        query_pe = self.query_positional_encoding(query)
        query = query + query_pe
        out, _ = self.cross_attention(query=query, key=x, value=x, key_padding_mask=padding_mask)
        return out


# --------------------------------------------------------------------------- #
# Frozen LingBot-Map streaming feature extractor
# --------------------------------------------------------------------------- #
class LingBotStream(nn.Module):
    """Frozen streaming wrapper over LingBot-Map `GCTStream`.

    Mirrors `precompute_lingbot_features.extract_trajectory`, but incrementally: the first
    `num_scale_frames` frames are buffered and processed as one bidirectional *scale block*, then
    every subsequent frame streams one-at-a-time (causal, `num_frame_per_block=1`). The KV cache
    persists across calls and reproduces LingBot's three tiers (scale full-patch, sliding-window
    full-patch, older anchor-only) natively.

    Per-frame outputs (write-once, identical to the training-time precompute cache):
        anchor     [6, 2C]     — special-token slot of agg_list[-1] (camera+register+scale)
        dino_desc  [D']        — mean-pooled context-free DINOv2 patch descriptor (match key)
        dino_patch [P, D']     — full context-free DINOv2 patch grid (dense cross-view)

    NOTE: this module is FROZEN (eval, requires_grad=False) for v1. Through-time fine-tuning is a
    later phase and would require truncated-BPTT streaming (see project memory: memnav-project).
    """

    def __init__(
        self,
        lingbot_repo="/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map",  # vendored copy
        weights="/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt",
        img_size=518,
        patch_size=14,
        num_scale_frames=8,
        kv_cache_sliding_window=64,
        camera_num_iterations=4,
        use_sdpa=True,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_scale_frames = num_scale_frames
        self.kv_cache_sliding_window = kv_cache_sliding_window

        if lingbot_repo not in sys.path:
            sys.path.insert(0, lingbot_repo)
        from lingbot_map.models.gct_stream import GCTStream
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        self._preprocess_fn = load_and_preprocess_images

        self.model = GCTStream(
            img_size=img_size,
            patch_size=patch_size,
            kv_cache_sliding_window=kv_cache_sliding_window,
            kv_cache_scale_frames=num_scale_frames,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=use_sdpa,
            camera_num_iterations=camera_num_iterations,
        )
        if weights:
            ckpt = torch.load(weights, map_location=device, weights_only=False)
            state_dict = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"[LingBotStream] loaded weights: {len(missing)} missing, {len(unexpected)} unexpected")
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Forward hook on the DINOv2 patch embedder → context-free patch descriptor.
        self._dino_capture = [None]
        self.model.aggregator.patch_embed.register_forward_hook(
            lambda _m, _i, out: self._dino_capture.__setitem__(0, out)
        )

        self.reset()

    # ----- streaming state -------------------------------------------------- #
    def reset(self):
        """Start a new episode: clear KV cache and scale-frame buffer."""
        self.model.clean_kv_cache()
        self._scale_buffer = []      # holds raw frames [1,3,H,W] until scale block fires
        self._scale_done = False
        self._n_frames = 0

    def _pop_dino(self, n_frames):
        pt = self._dino_capture[0]
        self._dino_capture[0] = None
        if isinstance(pt, dict):
            pt = pt["x_norm_patchtokens"]
        pt = pt.reshape(n_frames, -1, pt.shape[-1]).float()   # [n, P, D']
        return pt  # caller pools for dino_desc

    def preprocess(self, rgb):
        """rgb: HxWx3 uint8 ndarray (or path) → [1,3,H,W] float on device.

        TODO: confirm `load_and_preprocess_images` accepts in-memory arrays; the precompute
        passes file paths. May need a small adapter that square-pads to img_size like the
        offline path (`mode='pad'`).
        """
        raise NotImplementedError("wire LingBot image preprocessing for in-memory RGB")

    # ----- goal (context-free, does NOT touch the streaming cache) ---------- #
    @torch.no_grad()
    def encode_goal(self, rgb):
        """Single RGB goal → {dino_desc [D'], dino_patch [P, D']}, context-free.

        Must NOT mutate the episode KV cache. Implement by running only the DINOv2 trunk
        (the `patch_embed` path) on the goal frame and reading the hook — no aggregator
        streaming, no `append_frame`.

        TODO: pick the minimal context-free call (e.g. `model.aggregator.patch_embed(img)` or a
        dedicated `forward_image`) so the goal never enters the cache.
        """
        raise NotImplementedError("context-free goal encode (DINOv2 trunk only)")

    # ----- history (streaming-causal, updates the cache) -------------------- #
    @torch.no_grad()
    def step(self, rgb):
        """Stream one RGB frame. Returns the newest frame's features, or None during the
        scale-frame warmup (before the first `num_scale_frames` arrive).

        Returns dict(anchor [6,2C], dino_desc [D'], dino_patch [P,D']) or list-of-dicts when the
        scale block fires (emits `num_scale_frames` frames at once).

        TODO (cold start): decide warmup behaviour — buffer until scale block, or run a degraded
        scale block on however many frames exist so the robot can act from frame 0.
        """
        raise NotImplementedError(
            "streaming step: scale-block warmup then per-frame causal aggregate "
            "(mirror precompute.extract_trajectory phases 1 & 2)"
        )
