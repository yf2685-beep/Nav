"""GeometryModel_LingBot — LingBot-Map-based geometry backbone for LoGoPlanner.

Replaces the Pi3-based `GeometryModel` (see `geometry_model.py`). Preserves
LoGoPlanner's interface: returns a per-frame `state_token` and `scene_token`
(both [B, T, 384]) that the diffusion policy cross-attends to, plus Stage-1
supervision outputs (camera_poses, local_points, world_points) for the existing
LoGoPlanner Stage-1 losses L_pose + L_local + L_world.

Key design:
1. The aggregator from `lingbot_map.aggregator.stream.AggregatorStream` is used
   as the cross-frame transformer (Geometric Context Attention, paged KV cache).
2. The DepthAnythingV2-S metric-scale depth prior is preserved from LoGoPlanner
   and fused with the ViT image tokens BEFORE entering the GCA blocks — so the
   absolute-scale prior survives the backbone swap.
3. Stage-1 heads are simple Linear projections that mirror Pi3's `LinearPts3d`
   shape contract, so the existing trainer's losses work unchanged.

The forward signature `forward(imgs, depths)` matches the original GeometryModel.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Make `lingbot_map` importable. The clone lives at:
#     <root>/1 robot navigation/lingbot-map/
# while this file lives at:
#     <root>/1 robot navigation/Nav/NavDP/baselines/logoplanner/
# So we walk up four levels and append `lingbot-map`.
_THIS = os.path.dirname(os.path.abspath(__file__))
_LINGBOT_ROOT = os.path.normpath(os.path.join(_THIS, '..', '..', '..', '..', 'lingbot-map'))
if _LINGBOT_ROOT not in sys.path:
    sys.path.insert(0, _LINGBOT_ROOT)

from lingbot_map.aggregator.stream import AggregatorStream  # noqa: E402

from depth_anything.depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402
from Pi3.pi3.models.layers.transformer_head import LinearPts3d  # noqa: E402
from policy_backbone import TokenCompressor  # noqa: E402


class GeometryModel_LingBot(nn.Module):
    """LingBot-Map streaming geometric backbone with LoGoPlanner-compatible IO.

    Forward returns the same structure as the Pi3-based `GeometryModel`:
        (
          [hidden, state_token, scene_token],
          [camera_poses, local_points, world_points],
        )
    where state_token and scene_token are (B, T, 384) and feed the policy
    cross-attention; the second tuple is Stage-1 supervision only.
    """

    # Default aggregator config — overridable via kwargs at construction.
    DEFAULTS = dict(
        img_size=224,                # not strictly enforced; we pass actual H,W
        patch_size=14,
        embed_dim=1024,
        depth=24,                    # 12 alternating (frame, global) groups
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        patch_embed='dinov2_vitl14_reg',
        qkv_bias=True,
        qk_norm=True,
        sliding_window_size=-1,
        num_frame_for_scale=1,
        enable_3d_rope=False,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        use_sdpa=True,               # SDPA backend = no FlashInfer dependency
    )

    def __init__(self, context_size=12, device='cuda:0', **kwargs):
        super().__init__()
        self.context_size = context_size
        self.device = device

        # 1. LingBot AggregatorStream (DINOv2 patch_embed + alternating frame/GCA blocks)
        cfg = {**self.DEFAULTS, **kwargs}
        self.aggregator = AggregatorStream(**cfg)
        self.embed_dim = cfg['embed_dim']                          # 1024
        self.patch_start_idx = self.aggregator.patch_start_idx     # 6 for default
        self.patch_size = cfg['patch_size']                        # 14

        # 2. DepthAnythingV2-S for metric scale prior (kept from LoGoPlanner)
        model_configs = {'vits': {'encoder': 'vits', 'features': 64,
                                  'out_channels': [48, 96, 192, 384]}}
        self.depth_model = DepthAnythingV2(**model_configs['vits'])
        self.depth_model = self.depth_model.pretrained.float()
        self.depth_model.train()

        # 3. Fusion head: cat([image_tokens 1024, depth_tokens 384]) -> 1024
        self.fusion_head = nn.Linear(self.embed_dim + 384, self.embed_dim)

        # 4. Stage-1 supervision heads (Pi3-style for loss compatibility)
        # camera pose: 5-dim (x, y, z, sin θ, cos θ) — matches LoGoPlanner's
        # ExtrinctHead output. Takes the per-frame camera_token (1024).
        self.camera_pose_head = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 256),
            nn.GELU(),
            nn.Linear(256, 5),
        )
        # local + world points: per-pixel 3D points via patch-shuffle, same shape
        # contract as Pi3's LinearPts3d so trainer losses work unchanged.
        self.local_point_head = LinearPts3d(patch_size=self.patch_size,
                                            dec_embed_dim=self.embed_dim,
                                            output_dim=3)
        self.world_point_head = LinearPts3d(patch_size=self.patch_size,
                                            dec_embed_dim=self.embed_dim,
                                            output_dim=3)

        # 5. Token compressors → policy interface (state_token, scene_token; 384-d)
        # state_token: from the per-frame camera_token (single token).
        self.state_layer = nn.Linear(self.embed_dim, 384)
        # scene_token: from patch tokens (many) → compressor → 1 token per frame.
        self.scene_layer = nn.Linear(self.embed_dim, 384)
        self.scene_compressor = TokenCompressor(embed_dim=384, num_heads=8,
                                                target_length=1)

    # ----- public forward ---------------------------------------------------
    def forward(self, imgs, depths):
        """Match the Pi3 GeometryModel signature.

        Args:
            imgs:   (B, T, H, W, 3) RGB uint8 in [0,1] or float; will be cast.
            depths: (B, T, H, W, 1) depth (metric).

        Returns:
            ([hidden, state_token, scene_token],
             [camera_poses, local_points, world_points])
        """
        imgs   = torch.as_tensor(imgs,   dtype=torch.float32, device=self.device)
        depths = torch.as_tensor(depths, dtype=torch.float32, device=self.device)
        B, T, H, W, _ = imgs.shape
        assert T == self.context_size, \
            f"context_size mismatch: got {T}, expected {self.context_size}"

        # (B,T,H,W,3) -> (B,T,3,H,W) -> (B*T,3,H,W)
        imgs_t = imgs.permute(0, 1, 4, 2, 3).contiguous()
        # Normalize with ResNet stats (matches AggregatorBase._embed_images)
        mean = self.aggregator._resnet_mean.view(1, 3, 1, 1).to(imgs_t.device)
        std  = self.aggregator._resnet_std.view(1, 3, 1, 1).to(imgs_t.device)
        imgs_flat = imgs_t.view(B * T, 3, H, W)
        imgs_norm = (imgs_flat - mean) / std

        # (1) ViT image patch tokens
        patch_tokens = self.aggregator.patch_embed(imgs_norm)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens['x_norm_patchtokens']
        # patch_tokens: (B*T, P_patch, 1024)
        P_patch = patch_tokens.shape[1]
        C = self.embed_dim

        # (2) DA-S depth-prior features (triplicate depth to 3-channel for DA-S)
        depths_t = depths.permute(0, 1, 4, 2, 3).reshape(B * T, 1, H, W)
        depths_3ch = depths_t.expand(-1, 3, -1, -1)
        depth_tokens = self.depth_model.get_intermediate_layers(depths_3ch)[0]
        # depth_tokens: (B*T, P_patch, 384)

        # (3) Fuse image + depth tokens (preserve metric scale prior)
        fused = self.fusion_head(torch.cat([patch_tokens, depth_tokens], dim=-1))
        # fused: (B*T, P_patch, 1024)

        # (4) Prepend LingBot special tokens (camera + register + scale)
        special_tokens = self.aggregator._prepare_special_tokens(
            B, T, T, C, num_frame_for_scale=1,
        )  # (B*T, num_special, 1024)
        tokens = torch.cat([special_tokens, fused], dim=1)
        # tokens: (B*T, P_total, 1024)
        P_total = tokens.shape[1]

        # (5) RoPE positions
        pos = self.aggregator._get_positions(B, T, H, W, device=tokens.device)

        # (6) Alternating Frame Attention + GCA over aa_block_num groups
        frame_idx, global_idx = 0, 0
        frame_inter, global_inter = None, None
        for _ in range(self.aggregator.aa_block_num):
            for attn_type in self.aggregator.aa_order:
                if attn_type == 'frame':
                    tokens, frame_idx, frame_inter = self.aggregator._process_frame_attention(
                        tokens, B, T, P_total, C, frame_idx, pos=pos,
                    )
                elif attn_type == 'global':
                    tokens, global_idx, global_inter = self.aggregator._process_global_attention(
                        tokens, B, T, T, P_total, C, global_idx,
                        pos=pos, num_frame_for_scale=1,
                        sliding_window_size=None, num_frame_per_block=1,
                        image_height=H, image_width=W,
                    )

        # Defensively normalize tokens layout after the attention loop. The
        # last global-attention block leaves tokens in (B, S*P, C) layout for
        # cross-frame attention; we want (B*S, P, C) per-frame for the heads.
        # .reshape is safe — element count is unchanged.
        tokens = tokens.reshape(B * T, P_total, C)

        # (7) Pull out camera_token (idx 0) and patch tokens (after specials)
        cam_tok   = tokens[:, 0, :]                                # (B*T, 1024)
        patch_tok = tokens[:, self.patch_start_idx:, :]            # (B*T, P_patch, 1024)

        # (8) Stage-1 supervision outputs (Pi3-style shapes for trainer compat)
        # camera_poses: (B, T, 5)
        camera_poses = self.camera_pose_head(cam_tok).reshape(B, T, 5)
        # local/world points: (B*T, H, W, 3) -> (B, T, H, W, 3)
        local_points = self.local_point_head([patch_tok], (H, W)).reshape(B, T, H, W, 3)
        world_points = self.world_point_head([patch_tok], (H, W)).reshape(B, T, H, W, 3)

        # (9) Policy interface tokens (state_token, scene_token; both (B, T, 384))
        state_token = self.state_layer(cam_tok).reshape(B, T, 384)
        scene_token = self.scene_layer(patch_tok)                  # (B*T, P_patch, 384)
        scene_token = self.scene_compressor(scene_token).reshape(B, T, 384)

        return ([tokens, state_token, scene_token],
                [camera_poses, local_points, world_points])
