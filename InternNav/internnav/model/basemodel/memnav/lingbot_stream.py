"""LingBotStream — frozen GCT front-end for the memnav policy.

A thin wrapper around LingBot-Map's ``GCTStream`` (frozen, ``enable_3d_rope=True``)
that the policy calls inside ``forward``. It turns the precomputed KV cache
(``scale_k/v``, ``anchor_k/v``) + raw RGB into the tokens the policy conditions on,
via three operations — all reusing GCT's own attention (no extra trainable
geometry):

  * ``window_forward``  — inject the cache up to ``k`` and recompute the local
        window ``[k-W+1 .. k]`` → the **current post-GCA state**.
  * ``goal_append``     — append the goal as a frame in the stream so GCT
        relocalizes it. Two regimes (chosen by retrieval):
          - revisit (match ``m``): inject cache up to ``m``, promote the matched
            frame to full tokens, stream the goal at **time ``m+1``** → goal pose
            relative to the matched place (direction-through-map).
          - in-FoV (no match): inject up to ``k`` (with the live window), stream
            the goal at **time ``k+1``** → bearing if in view, else weak → explore.
  * ``dino``            — context-free DINOv2 trunk (CLS + dense patches) for the
        on-the-fly retrieval / matching space.

3D-RoPE note: under ``enable_3d_rope=True`` the per-frame temporal index is set by
``aggregator.total_frames_processed`` when a frame is streamed; the cached K/V
carry their original times (baked in at precompute). So a frame streamed with the
counter set to ``t`` is placed at temporal slot ``t`` relative to the cache — this
is exactly how we give the goal a time index it doesn't otherwise have.

FROZEN for v1: ``eval()`` + ``requires_grad_(False)`` + ``no_grad`` calls. Through-
time fine-tuning (flip the no_grad) is a later phase.
"""

import sys

import numpy as np
import torch
import torch.nn as nn


class LingBotStream(nn.Module):
    def __init__(
        self,
        lingbot_repo="/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map",
        weights="/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt",
        img_size=518,
        patch_size=14,
        num_scale=8,
        window=8,
        enable_3d_rope=True,
        max_frame_num=1024,
        camera_num_iterations=4,
        use_sdpa=True,
        device="cuda",
    ):
        super().__init__()
        self.device = device
        self.num_scale = num_scale
        self.window = window
        self.num_special = 1 + 4 + 1  # camera + 4 register + scale  (patch_start_idx)

        if lingbot_repo not in sys.path:
            sys.path.insert(0, lingbot_repo)
        from lingbot_map.models.gct_stream import GCTStream
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        self._preprocess = load_and_preprocess_images
        self.img_size, self.patch_size = img_size, patch_size

        self.model = GCTStream(
            img_size=img_size, patch_size=patch_size,
            enable_3d_rope=enable_3d_rope, max_frame_num=max_frame_num,
            kv_cache_sliding_window=window, kv_cache_scale_frames=num_scale,
            kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
            use_sdpa=use_sdpa, camera_num_iterations=camera_num_iterations,
        )
        if weights:
            ckpt = torch.load(weights, map_location=device, weights_only=False)
            sd = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            print(f"[LingBotStream] weights: {len(missing)} missing, {len(unexpected)} unexpected")
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.agg = self.model.aggregator
        self.depth = self.agg.depth

        # feature-only depth head — geometry feature for the current state
        # (analog of LoGoPlanner's scene_token). Shares the frozen depth-head weights;
        # feature_only returns the fused DPT feature before the depth convs.
        from lingbot_map.heads.dpt_head import DPTHead
        embed_dim = getattr(self.agg, "embed_dim", 1024)
        self.depth_feat_head = DPTHead(
            dim_in=2 * embed_dim, patch_size=patch_size, output_dim=2,
            activation="exp", conf_activation="expp1",
            feature_only=True, down_ratio=patch_size,    # -> patch-res (37x37) feature
        ).to(device).eval()
        # feature_only returns BEFORE the output convs, so drop them (their shapes
        # differ from the full head anyway); load only the shared DPT layers.
        src = {k: v for k, v in self.model.depth_head.state_dict().items()
               if not k.startswith("scratch.output_conv")}
        miss, unexp = self.depth_feat_head.load_state_dict(src, strict=False)
        miss = [m for m in miss if not m.startswith("scratch.output_conv")]
        print(f"[LingBotStream] depth_feat_head: {len(miss)} missing (non-output), {len(unexp)} unexpected")
        for p in self.depth_feat_head.parameters():
            p.requires_grad_(False)
        self.depth_feat_dim = 256   # DPT `features`

    # ------------------------------------------------------------------ #
    # image preprocessing
    # ------------------------------------------------------------------ #
    def load_images(self, rgb_paths):
        """paths -> [N, 3, H, W] preprocessed (matches the cache's preprocessing)."""
        return self._preprocess(rgb_paths, mode="pad", image_size=self.img_size, patch_size=self.patch_size)

    # ------------------------------------------------------------------ #
    # KV-cache injection
    # ------------------------------------------------------------------ #
    def _inject(self, scale_k, scale_v, anchor_k, anchor_v, n_hist, total_frames):
        """Populate the SDPA dict cache: scale (full) in k_i, history specials in
        k_i_special. Tensors expected on device, bfloat16.

          scale_k/v  : [L, H, num_scale, P, d]
          anchor_k/v : [L, H, n_hist, 6, d]   (history frames [num_scale .. ])
        Sets total_frames_processed so subsequently-streamed frames get the right
        temporal index under 3D RoPE.
        """
        self.model.clean_kv_cache()
        kv = self.agg.kv_cache
        for i in range(self.depth):
            kv[f"k_{i}"] = scale_k[i][None]                          # [1,H,num_scale,P,d]
            kv[f"v_{i}"] = scale_v[i][None]
            if n_hist > 0:
                kv[f"k_{i}_special"] = anchor_k[i, :, :n_hist][None]  # [1,H,n_hist,6,d]
                kv[f"v_{i}_special"] = anchor_v[i, :, :n_hist][None]
        self.agg.total_frames_processed = int(total_frames)

    def _inject_camera(self, cam_k, cam_v, n):
        """Inject the camera-head KV cache for frames [0..n-1] so the camera head
        relocalizes a freshly-streamed frame against that history.
          cam_k/v : [N, NI, TD, H, d] on device (bf16)
        Sets frame_idx = n (the next streamed frame lands at temporal slot n)."""
        ch = self.model.camera_head
        ch.clean_kv_cache()
        NI, TD = ch.num_iterations, ch.trunk_depth
        K, V = cam_k[:n], cam_v[:n]                              # [n, NI, TD, H, d]
        cc = []
        for it in range(NI):
            dd = {"_skip_append": False}
            for bl in range(TD):
                dd[f"k_{bl}"] = K[:, it, bl].permute(1, 0, 2)[None, :, :, None, :]  # [1,H,n,1,d]
                dd[f"v_{bl}"] = V[:, it, bl].permute(1, 0, 2)[None, :, :, None, :]
            cc.append(dd)
        ch.kv_cache = cc
        ch.frame_idx = int(n)

    @torch.no_grad()
    def camera_pose(self, cam_k, cam_v, n, agg_tokens):
        """Absolute camera pose in LingBot's map frame: inject the camera-head cache
        [0..n-1], run the frozen camera head on `agg_tokens` (a frame's aggregated_tokens_list),
        return its accumulated output pose `pred_pose_enc_list[-1]` = [S, 9]
        (absT[3], quaR[4], FoV[2]; the sum of 4 delta-refinement iterations). Current + goal
        poses share the scale-frame anchor → their relative is the revisit/aux-pose signal."""
        self._inject_camera(cam_k, cam_v, n)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pose_list = self.model.camera_head(agg_tokens, causal_inference=True,
                                               num_frame_per_block=1, num_frame_for_scale=self.num_scale)
        return pose_list[-1][0].float()                         # [S, 9]

    @staticmethod
    def _cam_to_device(cam_k, cam_v, device):
        """npz cam_k/v [N, NI, TD, H, d] -> device bf16."""
        return (torch.as_tensor(cam_k, device=device, dtype=torch.bfloat16),
                torch.as_tensor(cam_v, device=device, dtype=torch.bfloat16))

    @staticmethod
    def _cache_to_layered(scale_k, scale_v, anchor_k, anchor_v, device):
        """npz arrays -> device bf16, anchor permuted to [L,H,n,6,d].

          stored scale_k  : [L,H,num_scale,P,d]
          stored anchor_k : [N, L, H, 6, d]  -> [L, H, N, 6, d]
        """
        sk = torch.as_tensor(scale_k, device=device, dtype=torch.bfloat16)
        sv = torch.as_tensor(scale_v, device=device, dtype=torch.bfloat16)
        ak = torch.as_tensor(anchor_k, device=device, dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
        av = torch.as_tensor(anchor_v, device=device, dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
        return sk, sv, ak, av

    # ------------------------------------------------------------------ #
    # window-forward: current post-GCA state
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def window_forward(self, cache, window_imgs, k, return_multilayer=False):
        """Recompute the local window [k-W+1 .. k] on the injected cache.

          cache       : dict with scale_k/v [L,H,S,P,d] (layered) + anchor_k/v [L,H,N,6,d]
          window_imgs : [W, 3, H, W] for frames [k-W+1 .. k] (ordered)
        Returns window tokens [W, P, 2C] (current state = last row). If
        return_multilayer, also returns (cur_agg, patch_start_idx) for the depth head.
        """
        W = self.window
        n_hist = (k - W + 1) - self.num_scale     # frames [num_scale .. k-W] kept as specials
        self._inject(cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
                     n_hist=max(0, n_hist), total_frames=k - W + 1)
        outs = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for j in range(W):
                a, psi = self.model._aggregate_features(
                    window_imgs[j:j + 1][None].to(self.device),
                    num_frame_for_scale=self.num_scale, num_frame_per_block=1,
                )
                outs.append(a[-1][:, -1])     # [1, P, 2C]
        window_tokens = torch.cat(outs, 0)
        if return_multilayer:
            # current (last) frame's tokens at all selected layers + patch_start_idx,
            # for the feature-only depth head
            cur_agg = [layer for layer in a]   # each [1, 1, P, 2C]
            return window_tokens, cur_agg, psi
        return window_tokens

    @torch.no_grad()
    def depth_feature(self, cur_agg, cur_img, patch_start_idx):
        """Run the feature-only depth head on the current frame's multi-layer tokens →
        dense geometry feature, flattened to tokens.
          cur_agg : list of [1, 1, P, 2C] (selected layers)
          cur_img : [1, 1, 3, H, W]
        Returns [Hf*Wf, C] geometry tokens (C = depth_feat_dim)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feat = self.depth_feat_head(cur_agg, cur_img.to(self.device), patch_start_idx)  # [1,1,C,Hf,Wf]
        feat = feat[0, 0]                                  # [C, Hf, Wf]
        return feat.flatten(1).transpose(0, 1).float()     # [Hf*Wf, C]

    # ------------------------------------------------------------------ #
    # goal-append: relocalize the goal in the stream
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _stream_one(self, img, return_agg=False):
        """Stream a single frame on the current cache; return its output tokens [1, P, 2C].
        The frame's temporal index is whatever ``total_frames_processed`` currently is
        (and gets incremented), so positions are controlled by the caller via _inject.
        If return_agg, also returns the frame's aggregated_tokens_list (for the camera head)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            a, _ = self.model._aggregate_features(
                img[None, None].to(self.device),
                num_frame_for_scale=self.num_scale, num_frame_per_block=1,
            )
        if return_agg:
            return a[-1][:, -1], [layer for layer in a]   # tokens, agg list
        return a[-1][:, -1]   # [1, P, 2C]

    @torch.no_grad()
    def goal_append(self, goal_img, cache=None, match_idx=None, match_window_imgs=None, return_agg=False):
        """Append the goal as a frame so GCT relocalizes it; return its output tokens
        [1, P, 2C] (camera token at index 0 carries the pose; patches carry dense detail).
        If return_agg, also returns the goal frame's agg list (for ``camera_pose``).

          revisit (match_idx=m): recompute the local window ending at the matched
            frame ``[m-W+1 .. m]`` autoregressively — *the same warmup as the current
            frame's ``window_forward``* — so the matched frame has full recent context;
            then stream the goal at time m+1 → goal tokens relative to the matched place.
            ``match_window_imgs`` = [W, 3, H, W] for frames ``[m-W+1 .. m]``.
          in-FoV (match_idx=None): the cache must ALREADY be at state [0..k] (call
            ``window_forward`` first, which leaves it there + the counter at k+1); the
            goal is streamed at time k+1 → bearing if in view, else weak.
        """
        if match_idx is not None:
            # recompute [m-W+1..m] (leaves the cache at [0..m], counter = m+1)
            self.window_forward(cache, match_window_imgs, int(match_idx))
        return self._stream_one(goal_img, return_agg=return_agg)   # goal at time (m+1) or (k+1)

    # ------------------------------------------------------------------ #
    # context-free DINOv2 (retrieval / matching space)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def dino(self, imgs):
        """Context-free DINOv2 trunk (the GCT input encoder) — for retrieval (CLS)
        and the trainable novel branch (dense patches). Returns
        dict(cls [N, D'], patch [N, P_patch, D']). Does NOT touch the GCT cache.

        Replicates the aggregator's ResNet-normalize + patch_embed forward
        (base.py: ``(images - _resnet_mean)/_resnet_std`` then ``patch_embed``),
        so ``cls`` matches the stored ``dino_cls``.
        """
        imgs = imgs.to(self.device)
        mean = self.agg._resnet_mean.reshape(1, 3, 1, 1).to(imgs.dtype)
        std = self.agg._resnet_std.reshape(1, 3, 1, 1).to(imgs.dtype)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.agg.patch_embed.forward_features((imgs - mean) / std)
        return {"cls": out["x_norm_clstoken"].float(), "patch": out["x_norm_patchtokens"].float()}
