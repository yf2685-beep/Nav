"""LingBot-Map drop-in replacement for LoGoPlanner's Pi3-based GeometryModel.

Replaces `GeometryModel(Pi3)` defined in `geometry_model.py`.

Pipeline:
    context frames (B, N=12, H, W, 3)
       → resize to 518x518
       → frozen GCTStream backbone (DINOv2 ViT-L/14 + frame/global blocks)
       → tap aggregated_tokens_list[-1] (last-layer hidden states)
       → trainable Adapter (2048 → 384, attention-pool patches → 1 token / frame)
       → state_token, scene_token  (B, N, 384)   ← what diffusion head consumes

Two operating modes (selected by `stage1_heads` ctor arg, or env var LOGO_STAGE):
  - stage1_heads=True  → Build 3 small heads on Adapter output (camera, world
    point, local point); forward returns real geometric predictions. Used for
    Stage-1 metric-scale supervision: LingBot encoder frozen, Adapter + heads
    trained with geometric GT (camera pose / point cloud MSE). Stage-1 forces
    Adapter to encode metric scale + 3D structure into state/scene tokens.
  - stage1_heads=False → Returns dummy zeros for geometry (matches Pi3 GT
    shapes). Used for Stage-2 / inference where geometry loss weights are 0
    and only state_token/scene_token matter (fed to diffusion).
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Make lingbot-map repo importable ------------------------------------
# On HPC the repo lives at /scratch/ay2710/LingBot/lingbot-map.
# Override with env var LINGBOT_MAP_REPO if installed elsewhere.
_DEFAULT_LINGBOT_MAP_REPO = '/scratch/ay2710/LingBot/lingbot-map'
_LINGBOT_MAP_REPO = os.environ.get('LINGBOT_MAP_REPO', _DEFAULT_LINGBOT_MAP_REPO)
if _LINGBOT_MAP_REPO not in sys.path:
    sys.path.insert(0, _LINGBOT_MAP_REPO)

from lingbot_map.models.gct_stream import GCTStream  # noqa: E402


_DEFAULT_CKPT = '/scratch/ay2710/LingBot/lingbot-map-ckpt/lingbot-map.pt'


class AttnPool(nn.Module):
    """Pool a variable number of patch tokens into a single token via learned query."""

    def __init__(self, dim: int, heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * (dim ** -0.5))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (BF, P, D) → returns (BF, D)
        bf = tokens.shape[0]
        q = self.query.expand(bf, -1, -1)
        out, _ = self.attn(q, tokens, tokens)
        return out.squeeze(1)


class LingBotMapAdapter(nn.Module):
    """Project LingBot-Map per-frame patch tokens (2*embed_dim=2048) → state/scene tokens (384).

    aggregator output stacks frame-attention and global-attention features, so each
    token in ``aggregated_tokens_list[i]`` has dim ``2 * embed_dim = 2048`` (heads in
    the released ckpt are also built with ``dim_in=2*embed_dim``).
    """

    def __init__(self, in_dim: int = 2048, out_dim: int = 384, heads: int = 8):
        super().__init__()
        self.state_pool = AttnPool(in_dim, heads)
        self.scene_pool = AttnPool(in_dim, heads)
        self.state_proj = nn.Linear(in_dim, out_dim)
        self.scene_proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_tokens: torch.Tensor):
        # patch_tokens: (B, S, P, D)
        B, S, P, D = patch_tokens.shape
        flat = patch_tokens.reshape(B * S, P, D)
        state = self.state_pool(flat)                       # (B*S, D)
        scene = self.scene_pool(flat)                       # (B*S, D)
        state_token = self.state_proj(state).reshape(B, S, -1)
        scene_token = self.scene_proj(scene).reshape(B, S, -1)
        return state_token, scene_token


# ---------------------------------------------------------------------------
# Stage-1 heads on Adapter output
# ---------------------------------------------------------------------------
class CameraHeadNavi(nn.Module):
    """Predict (B, N, 5) per-frame camera pose from concat(state, scene) tokens."""

    def __init__(self, in_dim: int = 768, hidden: int = 128, out_dim: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim) → (B, N, 5)
        return self.net(x)


class PointDecoder(nn.Module):
    """Upsample (B, N, in_dim) per-frame token → (B, N, H, W, 3) dense point cloud.

    Bottleneck spatial size (bh, bw) is set to 12x22 because target_hw=(168,308),
    and 168/14=12, 308/14=22 — i.e. one bottleneck pixel per encoder patch block.
    Then 2x ConvTranspose upsamples to (48, 88), finally bilinear interpolate
    to the exact (168, 308).
    """

    def __init__(
        self,
        in_dim: int = 768,
        target_hw: tuple = (168, 308),
        bh: int = 12,
        bw: int = 22,
        ch_init: int = 64,
    ):
        super().__init__()
        self.target_hw = target_hw
        self.bh, self.bw, self.ch_init = bh, bw, ch_init
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, bh * bw * ch_init),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(ch_init, ch_init // 2, 4, stride=2, padding=1),       # 2x
            nn.GELU(),
            nn.ConvTranspose2d(ch_init // 2, ch_init // 4, 4, stride=2, padding=1),  # 4x
            nn.GELU(),
            nn.Conv2d(ch_init // 4, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim) → (B, N, H, W, 3)
        B, N, D = x.shape
        h = self.proj(x).reshape(B * N, self.ch_init, self.bh, self.bw)
        h = self.up(h)                                            # (B*N, 3, 4*bh, 4*bw)
        h = F.interpolate(h, size=self.target_hw, mode='bilinear', align_corners=False)
        h = h.reshape(B, N, 3, *self.target_hw).permute(0, 1, 3, 4, 2).contiguous()
        return h                                                   # (B, N, H, W, 3)


class LingBotMapGeometryModel(nn.Module):
    """Drop-in replacement for ``GeometryModel(Pi3)`` using LingBot-Map encoder.

    Constructor signature mirrors GeometryModel(context_size, device) so the
    one-line swap in ``policy_network.py`` is sufficient.
    """

    def __init__(
        self,
        context_size: int = 12,
        device: str = 'cuda:0',
        token_dim: int = 384,
        ckpt_path: str = _DEFAULT_CKPT,
        encoder_image_size: int = 518,
        freeze_backbone: bool = True,
        adapter_heads: int = 8,
        stage1_heads: bool = False,
        target_hw: tuple = (168, 308),
    ):
        super().__init__()
        self.context_size = context_size
        self.device = device
        self.encoder_image_size = encoder_image_size
        self.stage1_heads = stage1_heads
        self.target_hw = target_hw

        # ---- 1. Build GCTStream backbone -----------------------------------
        self.backbone = GCTStream(
            img_size=encoder_image_size,
            patch_size=14,
            enable_3d_rope=False,
            max_frame_num=64,
            use_sdpa=True,
            camera_num_iterations=4,
        )

        # ---- 2. Load pretrained weights ------------------------------------
        if ckpt_path and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location='cpu', weights_only=True, mmap=True)
            sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            print(f'[LingBotMap] loaded {ckpt_path}: '
                  f'missing={len(missing)} unexpected={len(unexpected)}')
            if len(missing) > 0:
                print(f'[LingBotMap]   first missing: {missing[:5]}')
            if len(unexpected) > 0:
                print(f'[LingBotMap]   first unexpected: {unexpected[:5]}')
        else:
            print(f'[LingBotMap] WARNING: ckpt not found at {ckpt_path}, using random init')

        # ---- 3. Freeze backbone --------------------------------------------
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # ---- 4. Trainable Adapter ------------------------------------------
        self.adapter = LingBotMapAdapter(in_dim=2048, out_dim=token_dim, heads=adapter_heads)

        # ---- 5. Optional Stage-1 geometric heads ---------------------------
        # Built only when stage1_heads=True. Consume concat(state_token, scene_token)
        # → 2*token_dim, predict camera_poses / world_points / local_points for
        # MSE supervision. Stage-1 forces Adapter to encode metric scale + 3D
        # structure into state/scene tokens so these heads can predict accurately.
        if stage1_heads:
            self.camera_head = CameraHeadNavi(in_dim=2 * token_dim, out_dim=5)
            self.world_point_head = PointDecoder(in_dim=2 * token_dim, target_hw=target_hw)
            self.local_point_head = PointDecoder(in_dim=2 * token_dim, target_hw=target_hw)
            print(f'[LingBotMap] stage1_heads ENABLED — camera/world/local heads built '
                  f'(target_hw={target_hw}).')
        else:
            self.camera_head = None
            self.world_point_head = None
            self.local_point_head = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def _preprocess(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, N, H, W, 3) in any range → (B, N, 3, S, S) in [0, 1]."""
        imgs = imgs.permute(0, 1, 4, 2, 3).contiguous()
        B, N, C, H, W = imgs.shape

        if imgs.max() > 2.0:
            imgs = imgs / 255.0
        imgs = imgs.clamp(0.0, 1.0)

        S = self.encoder_image_size
        if (H, W) != (S, S):
            imgs = imgs.reshape(B * N, C, H, W)
            imgs = F.interpolate(imgs, size=(S, S), mode='bilinear', align_corners=False)
            imgs = imgs.reshape(B, N, C, S, S)
        return imgs

    def forward(self, imgs, depths):
        """Match GeometryModel.forward(imgs, depths) exactly.

        Returns:
            [hidden, state_token, scene_token],
            [camera_poses, local_points, world_points]

        `depths` is accepted but ignored (LingBot-Map is RGB-only).
        Geometry trio is real when stage1_heads=True, dummy zeros otherwise.
        """
        imgs_5d = self._preprocess(imgs)                        # (B, N, 3, S, S)
        B, N = imgs_5d.shape[:2]
        assert N == self.context_size, (
            f'expected context_size={self.context_size} frames, got {N}'
        )

        # KV-cache policy (intentional, see paper):
        #   - training:  cross-batch samples are independent random episodes, so
        #                stale KV cache from a previous batch would corrupt the
        #                current one. Clear before each forward.
        #   - inference: KEEP the cache. Consecutive forward() calls in deployment
        #                represent the same robot's video stream, and reusing
        #                cached K/V is precisely the causal-streaming speedup we
        #                claim against the bidirectional Pi3 baseline. Do NOT
        #                clear in eval mode.
        if self.training and hasattr(self.backbone, 'clean_kv_cache'):
            self.backbone.clean_kv_cache()
        agg_list, patch_start_idx = self.backbone._aggregate_features(imgs_5d)

        last = agg_list[-1]                                     # (B, N, T, 2048)
        patch_tokens = last[:, :, patch_start_idx:]             # (B, N, P, 2048)

        state_token, scene_token = self.adapter(patch_tokens)    # (B, N, 384)

        _, _, H_in, W_in, _ = imgs.shape
        device = state_token.device
        dtype = state_token.dtype

        if self.stage1_heads:
            # concat(state, scene) → 768 fed to all three heads, so geometric
            # supervision pushes BOTH adapter projections to encode metric 3D.
            fused = torch.cat([state_token, scene_token], dim=-1)  # (B, N, 768)
            camera_poses = self.camera_head(fused)                  # (B, N, 5)
            world_points = self.world_point_head(fused)             # (B, N, H, W, 3)
            local_points = self.local_point_head(fused)             # (B, N, H, W, 3)
            # GT spatial size may not match decoder's native output; resize.
            if (H_in, W_in) != self.target_hw:
                wp = world_points.permute(0, 1, 4, 2, 3).reshape(B * N, 3, *self.target_hw)
                lp = local_points.permute(0, 1, 4, 2, 3).reshape(B * N, 3, *self.target_hw)
                wp = F.interpolate(wp, size=(H_in, W_in), mode='bilinear', align_corners=False)
                lp = F.interpolate(lp, size=(H_in, W_in), mode='bilinear', align_corners=False)
                world_points = wp.reshape(B, N, 3, H_in, W_in).permute(0, 1, 3, 4, 2).contiguous()
                local_points = lp.reshape(B, N, 3, H_in, W_in).permute(0, 1, 3, 4, 2).contiguous()
        else:
            camera_poses = torch.zeros(B, N, 5, device=device, dtype=dtype)
            local_points = torch.zeros(B, N, H_in, W_in, 3, device=device, dtype=dtype)
            world_points = torch.zeros(B, N, H_in, W_in, 3, device=device, dtype=dtype)

        # `hidden` is read out of the first tuple slot as `_` in policy_network.py
        hidden = state_token  # placeholder; never consumed downstream
        return [hidden, state_token, scene_token], [camera_poses, local_points, world_points]


if __name__ == '__main__':
    # Smoke test: instantiate, forward random data, print shapes.
    # Run on HPC where /scratch/ay2710/LingBot/lingbot-map{,-ckpt} exist.
    torch.manual_seed(0)
    stage = int(os.environ.get('LOGO_STAGE', '2'))
    print(f'[smoke] stage = {stage}')
    model = LingBotMapGeometryModel(context_size=12, stage1_heads=(stage == 1)).cuda()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'[LingBotMap] trainable {trainable/1e6:.2f}M / total {total/1e6:.2f}M '
          f'({100 * trainable / total:.2f}%)')

    imgs = torch.rand(2, 12, 168, 308, 3, device='cuda')
    depths = torch.rand(2, 12, 168, 308, 1, device='cuda')
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        (hidden, st, sc), (cp, lp, wp) = model(imgs, depths)
    print('state_token ', st.shape, st.dtype)
    print('scene_token ', sc.shape)
    print('camera_poses', cp.shape)
    print('local_points', lp.shape)
    print('world_points', wp.shape)
