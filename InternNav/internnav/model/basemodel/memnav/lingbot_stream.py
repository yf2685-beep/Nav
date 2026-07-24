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

import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

_DEFAULT_LINGBOT_REPO = "/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map"
_DEFAULT_LINGBOT_WEIGHTS = (
    "/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt"
)

# --------------------------------------------------------------------------- #
# Ground-anchored metric scale — shared helpers.
# Single source of truth for the floor-histogram scale recovery, used by
#   * LingBotStream.compute_metric_scale (on-the-fly fallback at train time), and
#   * scripts/dataset_converters/precompute_lingbot_features.py, which pools the
#     SAME estimate over the whole episode during its continuous stream and stores
#     ``ground_h_est`` in lingbot_cam_cache.npz (zero train-time cost).
# See compute_metric_scale's docstring for the method + validation numbers.
# --------------------------------------------------------------------------- #
GROUND_BIAS_CORRECTION = 1.15    # 1/median(s_raw/s_gt), per-frame-median estimator,
                                 # 16 validate_gated 2-leg eps, outlier-excl (2026-07-21)
GROUND_SCALE_RANGE = (0.8, 6.0)  # CLAMP bounds for the corrected estimate (ground_scale_from_
                                 # h_est now clamps, not rejects). RECALIBRATED 2026-07-22 over
                                 # the full 1919-episode pt1 sweep (diag_ground_scale_sweep.py):
                                 # old ceiling 4.0 (fit on 16 two-leg eps, [1.5,3.9]) is far below
                                 # the full set's true-scale median 3.56 / 37% > 4.0. 6.0 leaves
                                 # ~95.6% untouched; the ~4% above it clamp to 6.0, which beats
                                 # the old pooled-constant fallback in every band (est in (6,8]:
                                 # 12% err vs 59%). Only the ceiling ever binds (0 eps hit 0.8).


@torch.no_grad()
def ground_frame_heights(depth, conf, pose9, conf_quantile=0.5, pixel_stride=4):
    """Unproject depth into the map frame and return PER-FRAME-RELATIVE floor
    candidate heights: each point's map-frame +y-down height minus ITS OWN frame's
    camera height. Relative heights are what make the estimate correct on
    multi-level scenes — the capture camera is a fixed height above the LOCAL
    floor, while absolute map heights differ per level (validated failure:
    1LXtFkjw3qL's split-level episodes, up to 0.6 m GT camera-z spread).

      depth, conf : [F, H, W] on device (full depth head output, lingbot units)
      pose9       : [F, 9] absT+quaR(cam-to-world)+FoV for exactly these frames
    Returns (rel_heights [N], frame_of [N] long) for conf-filtered valid points."""
    from lingbot_map.utils.rotation import quat_to_mat
    pose = pose9.to(depth.device, torch.float32)
    R = quat_to_mat(torch.nn.functional.normalize(pose[:, 3:7], dim=-1))  # [F,3,3] c2w
    t = pose[:, :3]
    F_, H, W = depth.shape
    fy = (H / 2.0) / torch.tan(pose[:, 7] / 2.0)
    fx = (W / 2.0) / torch.tan(pose[:, 8] / 2.0)
    vs = torch.arange(0, H, pixel_stride, device=depth.device, dtype=torch.float32)
    us = torch.arange(0, W, pixel_stride, device=depth.device, dtype=torch.float32)
    v, u = torch.meshgrid(vs, us, indexing="ij")
    d = depth[:, ::pixel_stride, ::pixel_stride]                     # [F, h, w]
    c = conf[:, ::pixel_stride, ::pixel_stride]
    x = (u[None] - W / 2.0) * d / fx[:, None, None]
    y = (v[None] - H / 2.0) * d / fy[:, None, None]
    p_cam = torch.stack([x, y, d], -1)                               # [F, h, w, 3]
    p_world = torch.einsum("fij,fhwj->fhwi", R, p_cam) + t[:, None, None, :]
    rel = p_world[..., 1] - t[:, 1, None, None]                      # minus own cam_y
    # per-frame conf quantile: conf is a relative uncertainty, not calibrated across frames
    keep = (d > 1e-6) & (c >= torch.quantile(
        c.reshape(F_, -1), conf_quantile, dim=1)[:, None, None])
    frame_of = torch.arange(F_, device=depth.device)[:, None, None].expand_as(rel)
    return rel[keep], frame_of[keep]


@torch.no_grad()
def _frame_floor_peak(below, nbins, peak_thresh):
    """One frame's below-camera relative heights -> its floor distance, or None.
    Floor = the DEEPEST peak whose (3-bin smoothed) count clears ``peak_thresh`` of
    the max: the floor is the lowest real surface in view, but at a low mount
    furniture can out-vote it (global argmax was 77x off once); q99 clips
    through-window/reflection points that would fool the deepest rule."""
    hi = torch.quantile(below, 0.99)
    if not torch.isfinite(hi) or hi <= 1e-6:
        return None
    edges = torch.linspace(0.0, float(hi), nbins + 1)
    counts = torch.histogram(below.float().cpu(), bins=edges).hist
    smooth = torch.nn.functional.avg_pool1d(counts[None, None], 3, stride=1,
                                            padding=1, count_include_pad=False)[0, 0]
    cand = (smooth >= peak_thresh * smooth.max()).nonzero().flatten()
    peak = int(cand.max())
    lo_e, hi_e = edges[max(0, peak - 1)].item(), edges[min(nbins, peak + 2)].item()
    band = below[(below >= lo_e) & (below <= hi_e)]
    if band.numel() < 50:
        return None
    return float(band.median())


@torch.no_grad()
def ground_h_est_from_heights(rel_heights, frame_of, nbins=60, peak_thresh=0.3,
                              min_frame_points=500):
    """PER-FRAME floor estimates -> episode camera-to-floor distance (lingbot units).

      rel_heights : [N] per-frame-relative +y-down heights (ground_frame_heights)
      frame_of    : [N] the frame index of each point
    Each frame with enough below-camera points contributes its own deepest-peak
    floor estimate; the episode h_est is the MEDIAN over frames. The median (of
    frames, not pooled points) is what makes this robust to multi-level scenes
    (balcony frames overlooking a lower floor), furniture-dominated frames, and
    slow map-scale drift — each is a minority of frames.
    Returns (h_est float | None, dbg dict)."""
    rel_heights = rel_heights.cpu()
    frame_of = frame_of.cpu()
    n_frames = int(frame_of.max()) + 1 if frame_of.numel() else 0
    dbg = dict(n_points=int(rel_heights.numel()), n_frames=n_frames,
               n_valid=0, h_est=None, h_iqr=None)
    below_mask = rel_heights > 1e-6
    h_list = []
    for f in range(n_frames):
        below = rel_heights[below_mask & (frame_of == f)]
        if below.numel() < min_frame_points:
            continue
        h_f = _frame_floor_peak(below, nbins, peak_thresh)
        if h_f is not None:
            h_list.append(h_f)
    dbg["n_valid"] = len(h_list)
    if len(h_list) < max(3, n_frames // 8):
        return None, dbg
    h = torch.tensor(h_list)
    h_est = float(h.median())
    dbg["h_est"] = h_est
    dbg["h_iqr"] = float(torch.quantile(h, 0.75) - torch.quantile(h, 0.25))
    if h_est <= 1e-6:
        return None, dbg
    return h_est, dbg


def ground_scale_from_h_est(h_est, camera_height_m=0.5,
                            bias_correction=GROUND_BIAS_CORRECTION,
                            scale_range=GROUND_SCALE_RANGE):
    """h_est (lingbot units) -> metric scale multiplier, or None if implausible.
    bias_correction: the raw per-frame-median estimate carries a consistent
    underestimate (deepest-peak rule + depth-head far bias make h_est ~13% too
    deep) — fit as 1/median(s_raw/s_gt_umeyama)=1/0.868 over the 16 validate_gated
    2-leg episodes (outlier-excluded; scripts/diag_ground_scale.py, 2026-07-21).
    Residual after correction: ~8% std, range [0.78, 1.09]; the remaining spread is
    dominated by a per-scene depth bias (17DRP ~0.90 vs 1LX ~0.79 raw medians).
    scale_range: CLAMP the corrected estimate into [lo, hi] (only None if h_est is
    itself invalid). Changed 2026-07-22 from reject-to-None: over the full 1919-episode
    pt1 sweep every out-of-range episode hit the UPPER bound, and clamping to the
    ceiling beat the pooled-constant fallback in every band — est in (6,8] (true s_gt
    median 6.2): clamp-to-6 err 12% vs constant 59%; est > 8 (degenerate/low-motion,
    true s_gt median 15): clamp 61% vs constant 83%. The reject-to-constant design
    assumed out-of-range == floor-miss (est huge, truth ~normal, constant closer), but
    no such episode exists here — high-est episodes have genuinely high true scale, and
    the depth estimate tracks it (scripts/diag_ground_scale_sweep.py). Ceiling recalibr-
    ated to 6.0 (was 4.0). Overridable per run via MEMNAV_GROUND_SCALE_MAX (-> MemNavNet).
    In practice only the upper bound ever binds (0 episodes hit the 0.8 floor)."""
    if h_est is None or h_est <= 1e-6:
        return None
    s = float(bias_correction * camera_height_m / h_est)
    return float(min(max(s, scale_range[0]), scale_range[1]))     # clamp, not reject


class LingBotStream(nn.Module):
    def __init__(
        self,
        lingbot_repo=None,
        weights=None,
        img_size=518,
        patch_size=14,
        num_scale=8,
        window=8,
        enable_3d_rope=True,
        max_frame_num=4096,
        camera_num_iterations=4,
        use_sdpa=True,
        device="cuda",
        scale_lru_size=4,
    ):
        super().__init__()
        self.device = device
        self.num_scale = num_scale
        self.window = window
        self.num_special = 1 + 4 + 1  # camera + 4 register + scale  (patch_start_idx)
        # LRU of on-the-fly-computed scale KV, keyed by rgb_dir. Each entry is
        # (scale_k, scale_v) bf16 on device, ~1.08 GB — keep the cap small.
        self._scale_lru_size = int(scale_lru_size)
        self._scale_lru: "OrderedDict[str, tuple]" = OrderedDict()

        if lingbot_repo is None:
            lingbot_repo = os.environ.get("LINGBOT_REPO", _DEFAULT_LINGBOT_REPO)
        if weights is None:
            weights = os.environ.get("LINGBOT_WEIGHTS", _DEFAULT_LINGBOT_WEIGHTS)

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
            # Load to CPU first, then move to device — on some CUDA/driver combos
            # (observed on H100 + torch 2.8+cu128), map_location=cuda inflates
            # transient host RSS wildly during pickle deserialize.
            ckpt = torch.load(weights, map_location="cpu", weights_only=False)
            sd = ckpt.get("model", ckpt)
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            del ckpt, sd
            import gc
            gc.collect()
            print(f"[LingBotStream] weights: {len(missing)} missing, {len(unexpected)} unexpected")
        self.model = self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 3D-RoPE table extension (mirrors precompute build_model): the aggregator
        # sizes its WanRotaryPosEmbed table to max_frame_num, but the CAMERA HEAD
        # hardcodes max_seq_len=1024. camera_pose() runs the camera head forward at
        # the current frame index k (up to ~1917 for long 3leg episodes), so its
        # table must be extended too or forward() slices past the end and crashes
        # ("size of tensor a (32) must match b (22)"). Rebuild every rope table whose
        # max_seq_len < max_frame_num up to max_frame_num; the overlap region is
        # bit-identical (theta=10000, analytic), so <=1024 behavior is unchanged.
        from lingbot_map.layers.rope import WanRotaryPosEmbed, get_1d_rotary_pos_embed
        n_ext = 0
        for mod in self.model.modules():
            if isinstance(mod, WanRotaryPosEmbed) and mod.max_seq_len < max_frame_num:
                t_dim, h_dim, w_dim = mod.fhw_dim
                old = mod.freqs
                new = torch.cat([get_1d_rotary_pos_embed(
                                    d, max_frame_num, 10000.0, use_real=False,
                                    repeat_interleave_real=False, freqs_dtype=torch.float64)
                                 for d in (t_dim, h_dim, w_dim)], dim=1)
                assert torch.allclose(new[:old.shape[0]].to(old.dtype), old.to(new.dtype).to(old.dtype),
                                      atol=1e-6), "RoPE table rebuild changed the overlap region"
                mod.freqs = new.to(old.device)
                mod.max_seq_len = max_frame_num
                n_ext += 1
        print(f"[LingBotStream] extended {n_ext} RoPE table(s) to max_frame_num={max_frame_num}")

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
    @staticmethod
    def _prefix_count(frame_indices, raw_frame_exclusive):
        """Number of sorted sparse KVs whose raw index is before a boundary."""
        if frame_indices is None:
            return None
        if torch.is_tensor(frame_indices):
            boundary = torch.as_tensor(
                raw_frame_exclusive,
                device=frame_indices.device,
                dtype=frame_indices.dtype,
            )
            return int(torch.searchsorted(frame_indices, boundary).item())
        return int(np.searchsorted(np.asarray(frame_indices), raw_frame_exclusive))

    def _inject(
        self,
        scale_k,
        scale_v,
        anchor_k,
        anchor_v,
        n_hist=None,
        total_frames=None,
        *,
        anchor_frame_indices=None,
        raw_start=None,
    ):
        """Populate the SDPA dict cache: scale (full) in k_i, history specials in
        k_i_special. Tensors expected on device, bfloat16.

          scale_k/v  : [L, H, num_scale, P, d]
          anchor_k/v : [L, H, n_hist, 6, d]   (history frames [num_scale .. ])
        For a versioned sparse cache, ``anchor_frame_indices`` are raw video
        indices and ``raw_start`` is the first live-recomputed frame.  LingBot's
        aggregator temporal counter advances only when a KV is appended, so its
        next index is ``num_scale + number_of_injected_keyframes``.  Legacy dense
        callers may continue to provide ``n_hist`` and ``total_frames`` directly.
        """
        if anchor_frame_indices is not None:
            if raw_start is None:
                raise ValueError("raw_start is required with anchor_frame_indices")
            n_hist = self._prefix_count(anchor_frame_indices, int(raw_start))
            total_frames = self.num_scale + n_hist
        if n_hist is None or total_frames is None:
            raise ValueError("cache injection requires history length and temporal index")
        if not 0 <= int(n_hist) <= int(anchor_k.shape[2]):
            raise ValueError(
                f"invalid anchor prefix {n_hist} for {anchor_k.shape[2]} cached rows"
            )
        self.model.clean_kv_cache()
        kv = self.agg.kv_cache
        for i in range(self.depth):
            kv[f"k_{i}"] = scale_k[i][None]                          # [1,H,num_scale,P,d]
            kv[f"v_{i}"] = scale_v[i][None]
            if n_hist > 0:
                kv[f"k_{i}_special"] = anchor_k[i, :, :n_hist][None]  # [1,H,n_hist,6,d]
                kv[f"v_{i}_special"] = anchor_v[i, :, :n_hist][None]
        self.agg.total_frames_processed = int(total_frames)

    def _inject_camera(self, cam_k, cam_v, n, cam_frame_indices=None):
        """Inject camera KVs before raw frame ``n`` and set its raw RoPE time.

        Sparse camera KVs retain their original raw temporal encoding.  Their row
        count is therefore selected through ``cam_frame_indices < n`` while
        ``frame_idx`` must still be the raw index ``n``; using the sparse row count
        as time would silently shift every goal pose.
        """
        ch = self.model.camera_head
        ch.clean_kv_cache()
        NI, TD = ch.num_iterations, ch.trunk_depth
        n_cached = (
            int(n)
            if cam_frame_indices is None
            else self._prefix_count(cam_frame_indices, int(n))
        )
        if not 0 <= n_cached <= len(cam_k):
            raise ValueError(
                f"invalid camera prefix {n_cached} for {len(cam_k)} cached rows"
            )
        K, V = cam_k[:n_cached], cam_v[:n_cached]
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
    def camera_pose(self, cam_k, cam_v, n, agg_tokens, cam_frame_indices=None):
        """Absolute camera pose in LingBot's map frame: inject the camera-head cache
        [0..n-1], run the frozen camera head on `agg_tokens` (a frame's aggregated_tokens_list),
        return its accumulated output pose `pred_pose_enc_list[-1]` = [S, 9]
        (absT[3], quaR[4], FoV[2]; the sum of 4 delta-refinement iterations). Current + goal
        poses share the scale-frame anchor → their relative is the revisit/aux-pose signal."""
        self._inject_camera(cam_k, cam_v, n, cam_frame_indices)
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
    # on-the-fly scale KV (skip storing scale_k/v on disk)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_scale_kv(self, rgb_paths):
        """Compute scale_k/v [L, H, num_scale, P, d] on the fly from the first
        ``num_scale`` RGB frames of a trajectory.  Mirrors Phase 1 of
        ``precompute_lingbot_features.extract_trajectory`` (single bidirectional
        block of scale frames), so the result matches the on-disk ``scale_k/v``
        up to fp16↔bf16 storage rounding.

          rgb_paths : list of at least ``num_scale`` paths — only the first
                      ``num_scale`` are used.
        Returns (scale_k, scale_v) bf16 on device — same shape/dtype as
        :meth:`_cache_to_layered` produces from the on-disk arrays."""
        scale = self.num_scale
        imgs = self._preprocess(
            rgb_paths[:scale], mode="pad",
            image_size=self.img_size, patch_size=self.patch_size,
        ).unsqueeze(0).to(self.device)                        # [1, scale, 3, H, W]

        self.model.clean_kv_cache()
        kv = self.agg.kv_cache
        with torch.autocast("cuda", dtype=torch.bfloat16):
            self.model._aggregate_features(
                imgs, num_frame_for_scale=scale, num_frame_per_block=scale,
            )
        sk = torch.stack([kv[f"k_{i}"][0, :, :scale].to(torch.bfloat16)
                          for i in range(self.depth)]).contiguous()
        sv = torch.stack([kv[f"v_{i}"][0, :, :scale].to(torch.bfloat16)
                          for i in range(self.depth)]).contiguous()
        self.model.clean_kv_cache()
        return sk, sv

    # ------------------------------------------------------------------ #
    # ground-anchored per-trajectory METRIC scale (VGP-Nav sec III-E)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def compute_metric_scale(self, rgb_paths, cam_pose_enc, camera_height_m=0.5,
                             conf_quantile=0.5, pixel_stride=4, nbins=60,
                             n_frames=64, peak_thresh=0.3,
                             bias_correction=GROUND_BIAS_CORRECTION,
                             scale_range=GROUND_SCALE_RANGE, return_debug=False):
        """Recover the per-trajectory metric scale from LingBot's own geometry by
        anchoring the floor it reconstructs to the known camera mount height
        (VGP-Nav, arXiv:2606.09268 sec III-E "Ground-Anchored Scale Recovery").

        Streams the first ``num_scale`` frames as one bidirectional block (the exact
        stream that defines the trajectory's map frame — same as compute_scale_kv),
        then keeps streaming one frame at a time up to ``n_frames`` (the same
        continuous causal stream precompute runs, so the poses in ``cam_pose_enc``
        apply verbatim) — a single viewpoint often barely sees the floor, and pooling
        ~64 frames is what makes the floor peak dominant (VGP-Nav likewise anchors a
        multi-view reconstruction, not one snapshot). The FULL frozen depth head runs
        on each frame's tokens; points are unprojected into the map frame and the
        floor is the deepest significant peak of the pooled height histogram:

            scale = camera_height_m / (floor_y_peak - median(camera_y))

        Map-frame conventions this relies on (both validated for this checkpoint by
        scripts/diag_ground_scale.py): pose9 absT/quaR decode as CAM-TO-WORLD, and
        the map frame is the frame-0 OpenCV camera frame (y points DOWN), so the
        floor sits at y > camera_y and gravity is +y whenever the capture camera is
        level (true for the MP3D generator: pitch 0, fixed mount).

          rgb_paths      : >= num_scale frame paths of the trajectory (first ones used)
          cam_pose_enc   : [>=num_scale, 9] the trajectory's continuous-stream poses
                           (lingbot_cam_cache.npz), frames aligned with rgb_paths
          camera_height_m: true mount height (gen_meta.json camera_height_m; 0.5 default)
        Returns float scale (multiply LingBot translations by it to get meters), or
        None if too few confident floor points were found (caller falls back to the
        pooled constant)."""
        S = self.num_scale
        n = max(S, min(n_frames, len(rgb_paths)))
        imgs = self._preprocess(
            rgb_paths[:n], mode="pad",
            image_size=self.img_size, patch_size=self.patch_size,
        )                                                    # [n, 3, H, W] (cpu)

        self.model.clean_kv_cache()
        depths, confs = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # scale block (defines the map frame), then continuous per-frame stream —
            # mirrors precompute's extract_trajectory, so cam_pose_enc[i] is frame i's
            # pose for exactly this stream (sliding-window eviction handled internally)
            blk = imgs[:S][None].to(self.device)
            agg, psi = self.model._aggregate_features(
                blk, num_frame_for_scale=S, num_frame_per_block=S)
            pred = self.model._predict_depth(agg, blk, psi)  # full head, fp32 inside
            depths.append(pred["depth"][0, ..., 0].float()); confs.append(pred["depth_conf"][0].float())
            for j in range(S, n):
                fj = imgs[j:j + 1][None].to(self.device)
                a, _ = self.model._aggregate_features(
                    fj, num_frame_for_scale=S, num_frame_per_block=1)
                pj = self.model._predict_depth(a, fj, psi)
                depths.append(pj["depth"][0, ..., 0].float()); confs.append(pj["depth_conf"][0].float())
        self.model.clean_kv_cache()
        depth = torch.cat(depths)                            # [n, H, W]  (lingbot units)
        conf = torch.cat(confs)                              # [n, H, W]

        pose = (cam_pose_enc[:n] if torch.is_tensor(cam_pose_enc)
                else torch.as_tensor(np.asarray(cam_pose_enc)[:n]))
        rel_heights, frame_of = ground_frame_heights(depth, conf, pose,
                                                     conf_quantile=conf_quantile,
                                                     pixel_stride=pixel_stride)
        h_est, dbg = ground_h_est_from_heights(rel_heights, frame_of, nbins=nbins,
                                               peak_thresh=peak_thresh)
        s = ground_scale_from_h_est(h_est, camera_height_m,
                                    bias_correction=bias_correction,
                                    scale_range=scale_range)
        return (s, dbg) if return_debug else s

    def get_metric_scale(self, rgb_dir, cam_pose_enc, camera_height_m=0.5):
        """Per-trajectory cached :meth:`compute_metric_scale`, keyed by ``rgb_dir``.
        Scalars only, so the cache is unbounded (a float per trajectory seen).
        Failures (None) are cached too — a floorless first block stays floorless."""
        if not hasattr(self, "_metric_scale_cache"):
            self._metric_scale_cache = {}
        if rgb_dir in self._metric_scale_cache:
            return self._metric_scale_cache[rgb_dir]
        paths = []
        for i in range(min(64, len(cam_pose_enc))):   # pooled-histogram frame budget
            p = os.path.join(rgb_dir, f"{i}.jpg")
            if not os.path.isfile(p):
                break
            paths.append(p)
        s = self.compute_metric_scale(paths, cam_pose_enc, camera_height_m)
        self._metric_scale_cache[rgb_dir] = s
        return s

    def get_scale_kv(self, rgb_dir):
        """LRU-cached :meth:`compute_scale_kv` keyed by ``rgb_dir`` (trajectory
        identity).  Trains at ~O(#samples-per-traj) recomputes per traj instead
        of one per sample — an 8-frame GCT forward is ~200–400 ms on H100 so
        even LRU=1 is fine; larger caps just help when many samples of the same
        traj land in the same worker in a short window."""
        entry = self._scale_lru.get(rgb_dir)
        if entry is not None:
            self._scale_lru.move_to_end(rgb_dir)
            return entry
        paths = [os.path.join(rgb_dir, f"{i}.jpg") for i in range(self.num_scale)]
        sk, sv = self.compute_scale_kv(paths)
        self._scale_lru[rgb_dir] = (sk, sv)
        while len(self._scale_lru) > self._scale_lru_size:
            self._scale_lru.popitem(last=False)
        return sk, sv

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
        raw_start = k - W + 1
        anchor_indices = cache.get("anchor_frame_indices")
        if anchor_indices is None:
            self._inject(
                cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
                n_hist=max(0, raw_start - self.num_scale), total_frames=raw_start,
            )
        else:
            self._inject(
                cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
                anchor_frame_indices=anchor_indices, raw_start=raw_start,
            )
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
    # BATCHED window-forward: run G independent streams on the batch dim
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _batched_hist_and_start(self, caches, raw_starts):
        """Per-sample (n_hist, temporal f_start) for a batched injection.

        Dense legacy caches: cache row i is frame i+num_scale, so n_hist is the
        raw prefix length and the RoPE offset is the raw start frame.  Versioned
        sparse caches: only keyframe rows exist, so n_hist counts keyframes
        before the raw start (searchsorted) and — because the aggregator's
        temporal counter advances only on appends — the RoPE offset is the
        COMPRESSED time num_scale + n_hist, mirroring :meth:`_inject`.
        """
        S = self.num_scale
        n_hist, f_start = [], []
        for cache, raw_start in zip(caches, raw_starts):
            indices = cache.get("anchor_frame_indices")
            if indices is None:
                n = max(0, int(raw_start) - S)
                f = int(raw_start)
            else:
                n = self._prefix_count(indices, int(raw_start))
                f = S + n
            n_hist.append(n)
            f_start.append(f)
        return n_hist, f_start

    def _inject_batched(self, caches, n_hist_list, f_start_list):
        """Inject G independent per-sample caches stacked on the batch dim, so a
        single streaming pass advances all G streams at once.

        Each sample b has its own scale KV (uniform ``num_scale`` frames) and a
        history-special block of length ``n_hist_b`` that VARIES per sample; the
        specials are right-padded to ``max_n_hist`` and the injected padding region
        [n_hist_b*6, max_n_hist*6) is masked out at attention time (kv_cache
        '_special_pad_lo'/'_special_pad_hi'). Per-sample temporal offsets
        ``f_start_list`` are handed to the aggregator via ``agg._batched_f_start`` so
        3D-RoPE places each stream's frames at the right absolute slots. Used for
        both the window replay (f_start = k-W+1) and the goal warm-up (f_start =
        max(num_scale, m-warm+1)).

          caches       : list of G dicts (from ``MemNavNet._load_cache``): scale_k/v
                         [L,H,num_scale,P,d], anchor_k/v [L,H,n_hist_b,6,d].
          n_hist_list  : G history-frame counts (specials injected per sample).
          f_start_list : G absolute temporal offsets of the FIRST streamed frame.
        Returns max_n_hist (int).
        """
        G = len(caches)
        n_hist = [max(0, int(x)) for x in n_hist_list]
        max_n_hist = max(n_hist)

        self.model.clean_kv_cache()
        kv = self.agg.kv_cache
        for i in range(self.depth):
            kv[f"k_{i}"] = torch.stack([caches[b]["scale_k"][i] for b in range(G)], 0)  # [G,H,S,P,d]
            kv[f"v_{i}"] = torch.stack([caches[b]["scale_v"][i] for b in range(G)], 0)
            if max_n_hist > 0:
                ak, av = [], []
                for b in range(G):
                    a = caches[b]["anchor_k"][i, :, :n_hist[b]]   # [H, n_hist_b, 6, d]
                    v = caches[b]["anchor_v"][i, :, :n_hist[b]]
                    pad = max_n_hist - n_hist[b]
                    if pad > 0:                                    # right-pad the frame axis
                        a = torch.nn.functional.pad(a, (0, 0, 0, 0, 0, pad))
                        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
                    ak.append(a); av.append(v)
                kv[f"k_{i}_special"] = torch.stack(ak, 0)          # [G,H,max_n_hist,6,d]
                kv[f"v_{i}_special"] = torch.stack(av, 0)

        dev = kv[f"k_0"].device
        self.agg._batched_f_start = torch.tensor([int(x) for x in f_start_list],
                                                 device=dev, dtype=torch.long)
        if max_n_hist > 0:
            kv["_special_pad_lo"] = torch.tensor([n_hist[b] * 6 for b in range(G)],
                                                 device=dev, dtype=torch.long)
            kv["_special_pad_hi"] = torch.tensor(max_n_hist * 6, device=dev, dtype=torch.long)
        return max_n_hist

    @torch.no_grad()
    def window_forward_batched(self, caches, window_imgs, ks, return_multilayer=False):
        """Batched analogue of :meth:`window_forward`. Streams the W window frames
        for G samples at once (batch dim), collapsing the per-sample Python loop
        into the aggregator's batch axis.

          caches      : list of G cache dicts (see :meth:`_inject_batched`)
          window_imgs : [G, W, 3, H, W] (each sample's frames [k-W+1 .. k], ordered)
          ks          : G current-frame indices
        Returns the current (last window frame) tokens [G, P, 2C]; if
        return_multilayer, also (cur_agg, patch_start_idx) with cur_agg a list of
        [G, 1, P, 2C] per selected layer.
        """
        W, S = self.window, self.num_scale
        G = len(caches)
        n_hist, f_start = self._batched_hist_and_start(
            caches, [int(ks[b]) - W + 1 for b in range(G)]
        )
        self._inject_batched(caches, n_hist, f_start)
        window_imgs = window_imgs.to(self.device)
        cur_agg = psi = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for j in range(W):
                a, psi = self.model._aggregate_features(
                    window_imgs[:, j:j + 1],
                    num_frame_for_scale=self.num_scale, num_frame_per_block=1,
                )
        cur = a[-1][:, -1]                                   # [G, P, 2C] (last frame = current state)
        if return_multilayer:
            cur_agg = [layer for layer in a]                 # each [G, 1, P, 2C]
        self._clear_batched_state()
        if return_multilayer:
            return cur, cur_agg, psi
        return cur

    def _clear_batched_state(self):
        """Drop the batched-stream markers so subsequent single-stream ops are unaffected."""
        self.agg._batched_f_start = None
        self.agg.kv_cache.pop("_special_pad_lo", None)
        self.agg.kv_cache.pop("_special_pad_hi", None)

    @torch.no_grad()
    def goal_append_warm_batched(self, goal_imgs, caches, ms, rgb_dirs, warm, return_agg=False):
        """Batched analogue of :meth:`goal_append_warm` for a LOCKSTEP group: every
        sample must warm-recompute the SAME number of frames ``L`` (the caller groups
        by L = m - max(num_scale, m-warm+1) + 1, so the batched stream + sliding-window
        eviction advance in lockstep). Per-sample variation — different start frame,
        history length, and temporal offset — is handled by :meth:`_inject_batched`'s
        per-sample RoPE offsets and padding mask.

          goal_imgs : [G, 3, H, W] goal frames (LingBot-preprocessed)
          caches    : list of G cache dicts
          ms        : G anchor indices (goal streamed at m+1)
          rgb_dirs  : G trajectory frame dirs
        Returns the goal tokens [G, P, 2C]; if return_agg, also a list (len G) of the
        per-sample agg lists (each a list of [1, 1, P, 2C]) for ``camera_pose``.
        """
        S = self.num_scale
        G = len(caches)
        starts = [max(S, int(ms[b]) - warm + 1) for b in range(G)]
        Ls = [int(ms[b]) - starts[b] + 1 for b in range(G)]
        assert len(set(Ls)) == 1, f"goal_append_warm_batched needs a lockstep group, got L={Ls}"
        L = Ls[0]
        n_hist, f_start = self._batched_hist_and_start(caches, starts)
        self._inject_batched(caches, n_hist, f_start)

        # per-sample warm frames [start_b .. m_b] (L each), stacked on the batch dim
        warm_imgs = torch.stack([
            self.load_images([os.path.join(rgb_dirs[b], f"{i}.jpg")
                              for i in range(starts[b], int(ms[b]) + 1)])
            for b in range(G)], 0).to(self.device)                 # [G, L, 3, H, W]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for j in range(L):
                self.model._aggregate_features(
                    warm_imgs[:, j:j + 1],
                    num_frame_for_scale=S, num_frame_per_block=1,
                )
            # stream the goal at time m_b+1 (f_start has advanced start_b -> m_b+1)
            a, _ = self.model._aggregate_features(
                goal_imgs.to(self.device)[:, None],
                num_frame_for_scale=S, num_frame_per_block=1,
            )
        goal_tok = a[-1][:, -1]                                     # [G, P, 2C]
        agg_per_sample = None
        if return_agg:
            agg_per_sample = [[layer[b:b + 1] for layer in a] for b in range(G)]  # per-b list of [1,1,P,2C]
        self._clear_batched_state()
        if return_agg:
            return goal_tok, agg_per_sample
        return goal_tok

    @torch.no_grad()
    def depth_feature_batched(self, cur_agg, cur_imgs, patch_start_idx):
        """Batched depth-head geometry feature. ``cur_agg`` = list of [G,1,P,2C],
        ``cur_imgs`` = [G,1,3,H,W]. Returns [G, Hf*Wf, C]."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feat = self.depth_feat_head(cur_agg, cur_imgs.to(self.device), patch_start_idx)  # [G,1,C,Hf,Wf]
        feat = feat[:, 0]                                    # [G, C, Hf, Wf]
        G = feat.shape[0]
        return feat.flatten(2).transpose(1, 2).float()       # [G, Hf*Wf, C]

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

    @torch.no_grad()
    def goal_append_warm(self, goal_img, cache, m, rgb_dir, warm, return_agg=False):
        """Like goal_append's revisit path, but optionally recomputes a warm-up window
        ``[max(num_scale, m-warm+1) .. m]`` instead of just the nominal ``self.window``
        frames, before streaming the goal at ``m+1``.

        ``warm=0`` is the exact sparse-stream control: inject every cached
        keyframe through raw frame ``m`` and append the goal without adding a
        dense local replay.  Positive values implement the hybrid global-sparse /
        local-dense memory used by MemNav.

        window_forward's cold start at the nominal window boundary starves the goal's pose
        estimate — the first live-recomputed frame (and everything causally downstream of
        it, including the goal) has no real predecessors, only the injected specials-only
        history. Empirically (scripts/diag_lingbot_pose_accuracy.py's goal-insertion test,
        comparing against a true continuous-stream oracle and the goal's real GT position)
        ``warm=64`` closes this gap almost entirely — matches oracle to within noise, while
        ``warm=32`` (the nominal window) leaves ~30% avoidable error on the table and
        ``warm=128`` buys nothing further. Cost is fixed at `warm` frames regardless of how
        deep `m` is, unlike replaying the whole trajectory.
        """
        warm = int(warm)
        if warm < 0:
            raise ValueError(f"goal warm-up must be non-negative, got {warm}")
        start = m + 1 if warm == 0 else max(self.num_scale, m - warm + 1)
        anchor_indices = cache.get("anchor_frame_indices")
        if anchor_indices is None:
            self._inject(
                cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
                n_hist=start - self.num_scale, total_frames=start,
            )
        else:
            self._inject(
                cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
                anchor_frame_indices=anchor_indices, raw_start=start,
            )
        if warm:
            imgs = self.load_images(
                [os.path.join(rgb_dir, f"{i}.jpg") for i in range(start, m + 1)]
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for j in range(len(imgs)):
                    self.model._aggregate_features(
                        imgs[j:j + 1][None].to(self.device),
                        num_frame_for_scale=self.num_scale, num_frame_per_block=1,
                    )
        return self._stream_one(goal_img, return_agg=return_agg)   # goal at time m+1

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
