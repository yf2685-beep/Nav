#!/usr/bin/env python
"""Precompute frozen LingBot-Map features for the `memnav` project.

LingBot-Map (Geometric Context Transformer) is used as a *frozen* streaming
memory backbone.  It needs torch 2.8 (and optionally FlashInfer), which is
incompatible with the InternNav training env, so we run it once here, offline,
in the ``lingbot-map`` conda env and cache two per-frame tensors next to every
trajectory.  Training (in the InternNav env) then only ever loads these caches.

For each trajectory we save ``videos/chunk-000/lingbot_cache.npz`` holding the
LingBot **KV cache** (not output tokens) so training can reload the streamed
history and run only the local sliding window live (memnav v1 design):

  * ``scale_k`` / ``scale_v``  [L, H, scale, P, d]  — full K/V of the `scale`
        anchor frames across all L=depth global blocks (P = 6 special + patches).
        Loaded into ``k_{i}`` / ``v_{i}`` at train time (scale tier, never evicted).

  * ``anchor_k`` / ``anchor_v`` [N-scale, L, H, 6, d]  — the 6 special-token K/V
        (camera + 4 register + scale) of every subsequent frame, captured the
        step it is the current streaming frame (write-once: read from the cache's
        newest slot before eviction can compress it). Loaded into the
        ``k_{i}_special`` stream for frames older than the live window.

  * ``dino_cls``  [N, 1024]  — context-free DINOv2 CLS token (``x_norm_clstoken``),
        the **symmetric match key** (same encoder for goal and history).

  * ``meta``  [scale, num_special, depth, heads, head_dim]  — shape header.

The streaming is driven exactly like ``GCTStream.inference_streaming`` (phase-1
scale block, then causal frame-by-frame), so the captured K/V are bit-for-bit
what the live eval server produces; we skip all prediction heads.

NOTE: K/V are per-layer, so the scale cache is heavy (~135 MB/frame → the 8
scale frames are ~1.1 GB/traj). The anchor stream is cheap (~0.59 MB/frame).

Run (in the lingbot-map / torch-2.8 env). Smoke-test on one trajectory first:

    python precompute_lingbot_features.py \
        --root_dirs /home/asus/Research/datasets/InternData-N1 \
        --image_size 518 --num_scale_frames 8 --use_sdpa --limit 1
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from internnav.model.basemodel.memnav.cache_schema import (
    CACHE_SCHEMA_VERSION,
    DEFAULT_KEYFRAME_BUDGET,
    FLOW_KEYFRAME_INTERVAL_SENTINEL,
    KEYFRAME_POLICY,
    KEYFRAME_POLICY_FLOW,
    auto_keyframe_interval,
    validate_cache_files,
)


# --------------------------------------------------------------------------- #
# Trajectory discovery (mirror NavDP_Base_Datset.__init__ directory walk)
# --------------------------------------------------------------------------- #
def find_trajectories(root_dirs):
    """Yield (traj_dir, rgb_dir, sorted rgb paths) for every trajectory.

    Layout (same as internnav/dataset/navdp_dataset_lerobot.py):
        root_dirs/<group>/<scene>/<traj>/videos/chunk-000/observation.images.rgb/<i>.jpg
    """
    trajectories = []
    for group_dir in sorted(os.listdir(root_dirs)):
        group_path = os.path.join(root_dirs, group_dir)
        if not os.path.isdir(group_path):
            continue
        for scene_dir in sorted(os.listdir(group_path)):
            scene_path = os.path.join(group_path, scene_dir)
            if not os.path.isdir(scene_path):
                continue
            for traj_dir in sorted(os.listdir(scene_path)):
                entire_task_dir = os.path.join(scene_path, traj_dir)
                rgb_dir = os.path.join(
                    entire_task_dir, "videos/chunk-000/observation.images.rgb/"
                )
                if not os.path.isdir(rgb_dir):
                    continue
                n = len([p for p in os.listdir(rgb_dir) if p.endswith(".jpg")])
                if n == 0:
                    continue
                rgb_paths = [os.path.join(rgb_dir, "%d.jpg" % i) for i in range(n)]
                if not all(os.path.exists(p) for p in rgb_paths):
                    continue
                trajectories.append((entire_task_dir, rgb_dir, rgb_paths))
    return trajectories


def select_trajectories(trajectories, root_dirs, trajectory_list=""):
    """Select an explicit, ordered subset from a newline-delimited manifest.

    Each non-empty, non-comment line is an episode path relative to ``root_dirs``.
    A tab-delimited diagnostic suffix is allowed so the strict-coverage report can
    be passed directly.  Unknown or duplicate episodes are rejected rather than
    silently ignored.
    """
    if not trajectory_list:
        return trajectories

    by_relpath = {
        os.path.relpath(item[0], root_dirs): item
        for item in trajectories
    }
    requested = []
    seen = set()
    duplicates = set()
    with open(trajectory_list, encoding="utf-8") as manifest:
        for line_number, raw_line in enumerate(manifest, start=1):
            episode = raw_line.split("\t", 1)[0].strip().rstrip("/")
            if not episode or episode.startswith("#"):
                continue
            normalized = os.path.normpath(episode)
            if os.path.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
                raise ValueError(
                    f"{trajectory_list}:{line_number}: episode must be relative to "
                    f"root_dirs, got {episode!r}"
                )
            if normalized in seen:
                duplicates.add(normalized)
            seen.add(normalized)
            requested.append(normalized)

    if not requested:
        raise ValueError(f"No episode paths found in {trajectory_list}")
    if duplicates:
        raise ValueError(
            f"Duplicate episode(s) in {trajectory_list}: {sorted(duplicates)[:5]}"
        )
    missing = [episode for episode in requested if episode not in by_relpath]
    if missing:
        raise ValueError(
            f"{len(missing)} episode(s) from {trajectory_list} were not found under "
            f"{root_dirs}: {missing[:5]}"
        )
    return [by_relpath[episode] for episode in requested]


def validate_frame_capacity(trajectories, max_frame_num):
    """Fail before GPU work if temporal RoPE cannot represent an episode."""
    too_long = [
        (traj_dir, len(rgb_paths))
        for traj_dir, _rgb_dir, rgb_paths in trajectories
        if len(rgb_paths) > max_frame_num
    ]
    if too_long:
        examples = "; ".join(
            f"{traj_dir} ({length} frames)" for traj_dir, length in too_long[:5]
        )
        raise ValueError(
            f"max_frame_num={max_frame_num} is too small for {len(too_long)} "
            f"trajectory/trajectories; temporal RoPE would be truncated. Examples: "
            f"{examples}. Set --max_frame_num to at least "
            f"{max(length for _traj_dir, length in too_long)}."
        )


def _atomic_savez(path, **arrays):
    """np.savez to a sibling .tmp then os.replace(path) so the final file is never
    partially written. Writing to an open handle avoids numpy's auto ".npz" suffixing."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _sha256_file(path, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", os.fspath(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_keyframe_interval(args, num_frames):
    if args.flow_threshold > 0:
        # Flow-gated selection is causal and per-frame; there is no interval.
        # The sentinel is stored in the cache so validation knows the indices
        # are irregular by design, not a corrupted modular cache.
        return FLOW_KEYFRAME_INTERVAL_SENTINEL
    if args.auto_keyframe_interval:
        return auto_keyframe_interval(num_frames, args.keyframe_budget)
    if args.keyframe_interval < 1:
        raise ValueError(
            f"--keyframe_interval must be positive, got {args.keyframe_interval}"
        )
    return int(args.keyframe_interval)


def _precompute_provenance(args):
    """Immutable configuration shared by both files of every generated pair."""
    weights_sha256 = _sha256_file(args.weights) if args.weights else "none"
    lingbot_revision = _git_revision(args.lingbot_repo)
    if lingbot_revision == "unknown":
        raise RuntimeError(
            f"LINGBOT_REPO must be an auditable git checkout: {args.lingbot_repo}"
        )
    internnav_revision = _git_revision(PROJECT_ROOT)
    if internnav_revision == "unknown":
        raise RuntimeError(f"InternNav must be inside an auditable git checkout: {PROJECT_ROOT}")
    flow_gated = args.flow_threshold > 0
    if flow_gated:
        keyframe_interval_mode = (
            f"flow_thr{args.flow_threshold:g}px_gap{args.max_non_keyframe_gap}"
        )
    elif args.auto_keyframe_interval:
        keyframe_interval_mode = f"auto_budget_{args.keyframe_budget}"
    else:
        keyframe_interval_mode = f"fixed_{args.keyframe_interval}"
    config = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "keyframe_policy": KEYFRAME_POLICY_FLOW if flow_gated else KEYFRAME_POLICY,
        "internnav_revision": internnav_revision,
        "precompute_script_sha256": _sha256_file(__file__),
        "cache_schema_sha256": _sha256_file(
            os.path.join(
                PROJECT_ROOT,
                "internnav/model/basemodel/memnav/cache_schema.py",
            )
        ),
        "keyframe_interval_mode": keyframe_interval_mode,
        "flow_threshold": args.flow_threshold,
        "max_non_keyframe_gap": args.max_non_keyframe_gap,
        "lingbot_revision": lingbot_revision,
        "weights_sha256": weights_sha256,
        "image_size": args.image_size,
        "patch_size": args.patch_size,
        "num_scale_frames": args.num_scale_frames,
        "kv_cache_sliding_window": args.kv_cache_sliding_window,
        "enable_3d_rope": bool(args.enable_3d_rope),
        "max_frame_num": args.max_frame_num,
        "camera_num_iterations": args.camera_num_iterations,
        "use_sdpa": bool(args.use_sdpa),
        "preprocess_mode": args.preprocess_mode,
        "dtype": args.dtype,
        "skip_ground_scale": bool(args.skip_ground_scale),
        "ground_stride": args.ground_stride,
    }
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return payload, signature


# --------------------------------------------------------------------------- #
# Model construction (mirror demo.py:load_model)
# --------------------------------------------------------------------------- #
def build_model(args, device):
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=args.enable_3d_rope,       # temporal 3D RoPE — LingBot's intended mode (demo default True)
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"LingBot checkpoint mismatch: missing={len(missing)} "
                f"unexpected={len(unexpected)}"
            )
        print("  loaded weights: exact key match")

    # --- 3D-RoPE frame-table extension --------------------------------------- #
    # WanRotaryPosEmbed precomputes an ANALYTIC (untrained) frequency table of shape
    # [max_seq_len, ...]; forward() slices it by global frame index, so a trajectory
    # with more than max_seq_len frames slices past the end and silently returns too
    # few rows -> shape-mismatch crash. The aggregator sizes its table to max_frame_num,
    # but the camera head HARDCODES max_seq_len=1024 (ignores max_frame_num). Rebuild
    # every table shorter than max_frame_num up to max_frame_num.
    #   * Safe to load: the table is a plain attribute, not a registered buffer, so it is
    #     not in the checkpoint (load is 0-missing/0-unexpected regardless of size).
    #   * Backward-compatible: row i depends only on i, so extending only APPENDS rows
    #     >= old_len; every <=old_len trajectory reads identical rows. We ASSERT the
    #     overlap is bit-identical (proves theta==10000, the value both call sites use)
    #     so short-episode caches are guaranteed unchanged.
    from lingbot_map.layers.rope import WanRotaryPosEmbed, get_1d_rotary_pos_embed
    n_ext = 0
    for mod in model.modules():
        if isinstance(mod, WanRotaryPosEmbed) and mod.max_seq_len < args.max_frame_num:
            t_dim, h_dim, w_dim = mod.fhw_dim
            old = mod.freqs
            new = torch.cat([get_1d_rotary_pos_embed(
                                d, args.max_frame_num, 10000.0, use_real=False,
                                repeat_interleave_real=False, freqs_dtype=torch.float64)
                             for d in (t_dim, h_dim, w_dim)], dim=1)
            assert torch.allclose(new[:old.shape[0]].to(old.dtype), old.to(new.dtype).to(old.dtype),
                                  atol=1e-6), "RoPE table rebuild changed the overlap region"
            mod.freqs = new
            mod.max_seq_len = args.max_frame_num
            n_ext += 1
    if n_ext:
        print(f"  extended {n_ext} WanRotaryPosEmbed table(s) -> {args.max_frame_num} frames "
              f"(overlap verified bit-identical)")
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Per-trajectory KV-cache + CLS capture
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_trajectory(
    model,
    images,
    scale_frames,
    dino_capture,
    cam_only=False,
    skip_scale=False,
    keyframe_interval=1,
    ground_helpers=None,
    ground_stride=1,
    flow_threshold=0.0,
    max_non_keyframe_gap=30,
):
    """Stream `images` [1, S, 3, H, W] through the aggregator + camera head,
    capturing per-frame KV caches (and the DINOv2 CLS token).

    Aggregator cache (skipped when `cam_only`):
      * ``scale_k`` / ``scale_v``  [L, H, scale, P, d]   full K/V of the `scale` block.
      * ``anchor_k`` / ``anchor_v`` [K, L, H, 6, d] selected later-frame specials.
      * ``anchor_frame_indices`` [K] raw frame indices represented by those KVs.
      * ``dino_cls`` [S, D']  context-free DINOv2 CLS token (symmetric match key).

    Camera-head cache (always — the pose-specialized feature for revisit/aux-pose):
      * ``cam_k`` / ``cam_v`` [scale+K, NI, TD, H, d] the selected camera-token K/V across
            num_iterations × trunk_depth, captured the step each frame is current.
      * ``cam_frame_indices`` [scale+K] raw indices represented by those KVs.
      * ``cam_pose_enc`` [S, 9]  the head's NATIVE streaming pose (absT, quaR, FoV)
            per frame — empirically decodes as cam-to-world (despite the VGGT-derived
            w2c docstring). Used for metric calibration (per-traj monocular scale /
            map alignment) and as exact anchor poses in the revisit sweep.

    The camera head is run via a direct ``camera_head(...)`` call right after each
    ``_aggregate_features`` (it manages its own frame_idx + KV cache); this avoids the
    depth/world-point heads that the full ``forward`` would also run.

    Ground-anchored metric scale (``ground_helpers`` = the pair
    ``(ground_frame_heights, ground_h_est_from_heights)`` from internnav's
    lingbot_stream module): additionally run the FULL depth head on each streamed
    frame's tokens (every ``ground_stride``-th frame; ~10-15 ms/frame on top of a
    stream that already runs the aggregator anyway), unproject with the frame's own
    just-captured pose, and pool a WHOLE-EPISODE floor-height histogram. Saves
      * ``ground_h_est``  scalar — estimated camera-to-floor distance in lingbot map
            units (median of per-frame deepest-peak estimates — robust to
            multi-level scenes / balcony views / furniture frames / scale drift;
            NaN if too few frames saw a confident floor). Training turns it into
            the metric scale via ground_scale_from_h_est(h_est, camera_height_m).
      * ``ground_dbg``    [n_points, n_frames, n_valid, h_iqr] float32.
    """
    B, S = images.shape[0], images.shape[1]
    assert B == 1
    scale = min(scale_frames, S)
    keyframe_interval = int(keyframe_interval)
    flow_gated = float(flow_threshold) > 0.0
    if flow_gated:
        if keyframe_interval != FLOW_KEYFRAME_INTERVAL_SENTINEL:
            raise ValueError(
                "flow-gated selection replaces keyframe_interval; pass the "
                f"sentinel {FLOW_KEYFRAME_INTERVAL_SENTINEL}, got {keyframe_interval}"
            )
        if int(max_non_keyframe_gap) < 1:
            raise ValueError(
                f"max_non_keyframe_gap must be positive, got {max_non_keyframe_gap}"
            )
        if getattr(model, "depth_head", None) is None:
            raise RuntimeError(
                "flow-gated keyframe selection needs the LingBot depth head "
                "(flow is reprojection parallax from predicted pose + depth)"
            )
        # Upstream's flow metric (gct_stream_window): back-project the current
        # frame's predicted depth, reproject into the last keyframe's camera,
        # mean pixel displacement.  Rotation-dominant and scale-invariant.
        from lingbot_map.models.gct_stream_window import _compute_flow_magnitude
    else:
        if keyframe_interval < 1:
            raise ValueError(
                f"keyframe_interval must be positive, got {keyframe_interval}"
            )
        if keyframe_interval > 1 and not hasattr(model, "_set_skip_append"):
            raise RuntimeError(
                "sparse keyframe precompute requires LingBot GCTStream._set_skip_append; "
                "update LINGBOT_REPO before generating caches"
            )

    model.clean_kv_cache()
    dev = next(model.parameters()).device
    agg = model.aggregator
    L = agg.depth                       # number of global blocks (24)
    kv = agg.kv_cache                   # SDPA dict cache: k_{i}/v_{i} = [B,H,S_frames,P,d]
    ch = model.camera_head
    NI, TD = ch.num_iterations, ch.trunk_depth      # 4 x 4

    cls_list = []

    def pop_cls(n_frames):
        out = dino_capture[0]; dino_capture[0] = None
        cls = out["x_norm_clstoken"] if isinstance(out, dict) else out
        return cls.reshape(n_frames, -1).float().cpu()

    def read_cache_full(frame_slice):
        ks = torch.stack([kv[f"k_{i}"][0, :, frame_slice].to(torch.float16).cpu() for i in range(L)])
        vs = torch.stack([kv[f"v_{i}"][0, :, frame_slice].to(torch.float16).cpu() for i in range(L)])
        return ks, vs

    def read_cache_anchor_newest(n_special):
        ks = torch.stack([kv[f"k_{i}"][0, :, -1, :n_special].to(torch.float16).cpu() for i in range(L)])
        vs = torch.stack([kv[f"v_{i}"][0, :, -1, :n_special].to(torch.float16).cpu() for i in range(L)])
        return ks, vs

    def read_cam_newest(n_new):
        """Newest `n_new` frames' camera-token K/V -> [n_new, NI, TD, H, d] fp16."""
        cc = ch.kv_cache                                          # list[NI] of {k_j/v_j: [B,H,F,1,d]}
        ks = torch.stack([torch.stack([cc[it][f"k_{bl}"][0, :, -n_new:, 0].to(torch.float16).cpu()
                                       for bl in range(TD)]) for it in range(NI)])   # [NI,TD,H,n_new,d]
        vs = torch.stack([torch.stack([cc[it][f"v_{bl}"][0, :, -n_new:, 0].to(torch.float16).cpu()
                                       for bl in range(TD)]) for it in range(NI)])
        return ks.permute(3, 0, 1, 2, 4).contiguous(), vs.permute(3, 0, 1, 2, 4).contiguous()

    cam_k_list, cam_v_list, cam_pose_list = [], [], []
    cam_frame_indices = list(range(scale))
    gs_heights, gs_frames, gs_off = [], [], [0]

    def pool_ground(agg_tokens, frame_imgs, pose9):
        """Depth head on the just-streamed frame(s) -> per-frame-relative floor
        candidate heights (cpu), frame indices offset to be episode-global."""
        gfh = ground_helpers[0]
        dp = model._predict_depth(agg_tokens, frame_imgs, psi)   # fp32 inside
        rel, fo = gfh(dp["depth"][0, ..., 0].float(), dp["depth_conf"][0].float(), pose9)
        gs_heights.append(rel.cpu()); gs_frames.append((fo + gs_off[0]).cpu())
        gs_off[0] += pose9.shape[0]

    # Phase 1: scale frames as a single bidirectional block.
    scale_imgs = images[:, :scale].to(dev, non_blocking=True)
    scale_agg, psi = model._aggregate_features(scale_imgs, num_frame_for_scale=scale, num_frame_per_block=scale)
    pl = ch(scale_agg, causal_inference=True, num_frame_per_block=scale, num_frame_for_scale=scale)
    cam_pose_list.append(pl[-1][0].float().cpu())            # [scale, 9]
    ck, cv = read_cam_newest(scale); cam_k_list.append(ck); cam_v_list.append(cv)
    if ground_helpers is not None:
        pool_ground(scale_agg, scale_imgs, pl[-1][0].float())
    # Flow-gate state: the reference pose starts at the last scale frame.
    last_kf_pose = pl[-1][:, -1:].float()                    # [1, 1, 9] on device
    last_kf_idx = scale - 1
    if not cam_only:
        cls_list.append(pop_cls(scale))
        if not skip_scale:
            scale_k, scale_v = read_cache_full(slice(0, scale))  # [L,H,scale,P,d]

    # Phase 2: causal streaming, one frame at a time.
    anchor_k_list, anchor_v_list, anchor_frame_indices = [], [], []
    for i in range(scale, S):
        frame = images[:, i:i + 1].to(dev, non_blocking=True)
        if flow_gated:
            # Commit-then-drop: forward in normal append mode (one pass — the
            # frame's own attention is identical whether or not its KV persists),
            # decide from its predicted pose+depth, and undo the cache writes if
            # it is not a keyframe.  The undo is O(1): every SDPA dict-cache
            # update (append AND sliding-window eviction) builds new tensors via
            # torch.cat/slicing and reassigns dict entries — nothing is mutated
            # in place — so restoring the pre-forward dict references restores
            # the exact pre-forward cache state.
            saved_agg_kv = dict(kv)
            saved_cam_kv = [dict(dd) for dd in ch.kv_cache]
            saved_total = agg.total_frames_processed
            agg_tok, _ = model._aggregate_features(
                frame, num_frame_for_scale=scale, num_frame_per_block=1
            )
            pl = ch(
                agg_tok,
                causal_inference=True,
                num_frame_per_block=1,
                num_frame_for_scale=scale,
            )
            cur_pose = pl[-1][:, -1:].float()                # [1, 1, 9]
            if i == scale:
                # First post-scale frame always anchors the gate (official rule;
                # cache_schema._flow_gate_indices asserts it).
                is_keyframe = True
            else:
                depth = model._predict_depth(
                    agg_tok, images=frame, patch_start_idx=psi
                )["depth"].float()                           # [1, 1, H, W, 1]
                flow = _compute_flow_magnitude(
                    cur_pose, last_kf_pose, depth, tuple(depth.shape[2:4])
                )
                is_keyframe = (
                    flow > float(flow_threshold)
                    or (i - last_kf_idx) >= int(max_non_keyframe_gap)
                )
            if is_keyframe:
                last_kf_pose, last_kf_idx = cur_pose, i
            else:
                kv.clear(); kv.update(saved_agg_kv)
                for live, saved in zip(ch.kv_cache, saved_cam_kv):
                    live.clear(); live.update(saved)
                # Aggregator time counts memory WRITES (compressed keyframe
                # timeline) -> rewind.  The camera head's frame_idx counts
                # frames SEEN (raw timeline) -> deliberately not restored.
                agg.total_frames_processed = saved_total
        else:
            is_keyframe = (
                keyframe_interval <= 1
                or (i - scale) % keyframe_interval == 0
            )
            if not is_keyframe:
                model._set_skip_append(True)
            try:
                agg_tok, _ = model._aggregate_features(
                    frame, num_frame_for_scale=scale, num_frame_per_block=1
                )
                pl = ch(
                    agg_tok,
                    causal_inference=True,
                    num_frame_per_block=1,
                    num_frame_for_scale=scale,
                )
            finally:
                if not is_keyframe:
                    model._set_skip_append(False)
        cam_pose_list.append(pl[-1][0].float().cpu())        # [1, 9]
        if ground_helpers is not None and (i - scale) % ground_stride == 0:
            # Dense per-frame output like cam_pose_enc — pooled regardless of
            # keyframe status (only the KV memory is sparsified).
            pool_ground(agg_tok, frame, pl[-1][0].float())
        if is_keyframe:
            ck, cv = read_cam_newest(1)
            cam_k_list.append(ck)
            cam_v_list.append(cv)
            cam_frame_indices.append(i)
        if not cam_only:
            cls_list.append(pop_cls(1))
            if is_keyframe:
                ak, av = read_cache_anchor_newest(psi)
                anchor_k_list.append(ak)
                anchor_v_list.append(av)
                anchor_frame_indices.append(i)

    model.clean_kv_cache()

    Hh, d = ck.shape[3], ck.shape[-1]
    cam_k = torch.cat(cam_k_list, 0).numpy()                     # [S, NI, TD, H, d]
    cam_v = torch.cat(cam_v_list, 0).numpy()
    out = {"cam_k": cam_k, "cam_v": cam_v,
           "cam_pose_enc": torch.cat(cam_pose_list, 0).numpy(),   # [S, 9]
           "cam_frame_indices": np.asarray(cam_frame_indices, dtype=np.int64),
           "cam_meta": np.array([NI, TD, Hh, d], dtype=np.int64)}
    if ground_helpers is not None:
        h_est, gdbg = ground_helpers[1](torch.cat(gs_heights), torch.cat(gs_frames))
        out["ground_h_est"] = np.float32(h_est if h_est is not None else np.nan)
        out["ground_dbg"] = np.array([gdbg["n_points"], gdbg["n_frames"], gdbg["n_valid"],
                                      gdbg["h_iqr"] if gdbg["h_iqr"] is not None else np.nan],
                                     dtype=np.float32)
    if cam_only:
        return out

    if anchor_k_list:
        anchor_k = torch.stack(anchor_k_list, 0).numpy()         # [S-scale, L, H, psi, d]
        anchor_v = torch.stack(anchor_v_list, 0).numpy()
    else:
        anchor_k = np.zeros((0, L, Hh, psi, d), np.float16)
        anchor_v = np.zeros((0, L, Hh, psi, d), np.float16)
    out.update({
        "dino_cls": torch.cat(cls_list, dim=0).numpy(),          # [S, D']
        "anchor_k": anchor_k, "anchor_v": anchor_v,              # [S-scale, L, H, psi, d]
        "anchor_frame_indices": np.asarray(anchor_frame_indices, dtype=np.int64),
        "meta": np.array([scale, psi, L, Hh, d], dtype=np.int64),
    })
    if not skip_scale:
        out["scale_k"] = scale_k.numpy()                          # [L, H, scale, P, d]
        out["scale_v"] = scale_v.numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dirs", required=True,
                    help="InternData-N1 root (same path passed to NavDP dataset).")
    ap.add_argument("--lingbot_repo", default="/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map",
                    help="Path to the (vendored) lingbot-map repo (added to sys.path).")
    ap.add_argument("--weights",
                    default="/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt")
    ap.add_argument("--image_size", type=int, default=518)
    ap.add_argument("--patch_size", type=int, default=14)
    ap.add_argument("--num_scale_frames", type=int, default=8)
    ap.add_argument("--kv_cache_sliding_window", type=int, default=8)
    keyframes = ap.add_mutually_exclusive_group()
    keyframes.add_argument(
        "--keyframe_interval", type=int, default=1,
        help="Append post-scale KVs every N raw frames (1 keeps the legacy dense cache).",
    )
    keyframes.add_argument(
        "--auto_keyframe_interval", action="store_true",
        help="Use LingBot's per-trajectory ceil(num_frames / keyframe_budget) policy.",
    )
    ap.add_argument(
        "--keyframe_budget", type=int, default=DEFAULT_KEYFRAME_BUDGET,
        help="Temporal-view budget used by --auto_keyframe_interval (official default: 320).",
    )
    ap.add_argument(
        "--flow_threshold", type=float, default=0.0,
        help="Enable flow-gated keyframe selection (>0, pixels): a frame becomes a "
             "keyframe when its mean reprojection flow vs the last keyframe exceeds "
             "this threshold. Causal (no dependence on episode length), rotation-"
             "sensitive, scale-invariant. Mutually exclusive with interval flags.",
    )
    ap.add_argument(
        "--max_non_keyframe_gap", type=int, default=30,
        help="Flow mode only: force a keyframe after this many consecutive "
             "non-keyframes (backstop for feature-poor / stationary stretches).",
    )
    ap.add_argument("--enable_3d_rope", action="store_true", default=True,
                    help="Temporal 3D RoPE (LingBot's intended mode). Needed for goal time-index placement.")
    ap.add_argument("--max_frame_num", type=int, default=4096,
                    help="Max frames for 3D RoPE; must cover the longest selected trajectory.")
    ap.add_argument("--camera_num_iterations", type=int, default=4)
    ap.add_argument("--use_sdpa", action="store_true", default=False,
                    help="Use SDPA attention (no FlashInfer dependency). Recommended.")
    ap.add_argument("--preprocess_mode", default="pad", choices=["pad", "crop"],
                    help="LingBot image preprocess mode (square pad to image_size).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_name", default="lingbot_cache.npz",
                    help="Output filename written next to each trajectory's videos/chunk-000/.")
    ap.add_argument("--out_root", default="",
                    help="If set, write caches under out_root mirroring each trajectory's path "
                         "relative to root_dirs (instead of beside the frames). Required when "
                         "root_dirs is a read-only squashfs mount.")
    ap.add_argument("--cam_out_name", default="lingbot_cam_cache.npz",
                    help="Camera-head KV cache filename (cam_k/cam_v/cam_meta).")
    ap.add_argument("--cam_only", action="store_true",
                    help="Capture ONLY the camera-head cache (keep the existing aggregator npz untouched).")
    ap.add_argument("--skip_scale", action="store_true",
                    help="Skip writing scale_k/scale_v (~1.08 GB/traj). Training recomputes them "
                         "on the fly from the first num_scale RGB frames via "
                         "LingBotStream.get_scale_kv (LRU-cached).")
    ap.add_argument("--skip_ground_scale", action="store_true",
                    help="Skip the whole-episode ground-anchored scale estimate "
                         "(ground_h_est in the cam cache; adds one depth-head forward "
                         "per ground_stride-th frame, ~10-15 ms each).")
    ap.add_argument("--ground_stride", type=int, default=1,
                    help="Pool floor heights from every Nth streamed frame.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if the output npz already exists.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N trajectories (0 = all). For smoke tests.")
    ap.add_argument("--trajectory_list", default="",
                    help="Optional newline-delimited episode paths relative to root_dirs. "
                         "Tab-delimited suffixes are ignored, so a strict-coverage report "
                         "can be used directly.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Divide the traj list into this many interleaved shards.")
    ap.add_argument("--shard", type=int, default=0,
                    help="Which shard [0, num_shards) to process. Combine with SLURM array to parallelize.")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    if args.keyframe_interval < 1:
        ap.error("--keyframe_interval must be positive")
    if args.keyframe_budget < 1:
        ap.error("--keyframe_budget must be positive")
    if args.flow_threshold < 0:
        ap.error("--flow_threshold must be non-negative")
    if args.max_non_keyframe_gap < 1:
        ap.error("--max_non_keyframe_gap must be positive")
    flow_mode = args.flow_threshold > 0
    if flow_mode:
        if args.auto_keyframe_interval or args.keyframe_interval != 1:
            ap.error(
                "--flow_threshold replaces interval-based selection; drop "
                "--keyframe_interval/--auto_keyframe_interval"
            )
        if args.cam_only:
            ap.error(
                "--cam_only cannot reproduce a flow-gated selection (the keyframe "
                "set is decided during the joint stream); run a full precompute"
            )

    sys.path.insert(0, args.lingbot_repo)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    ground_helpers = None
    if not args.skip_ground_scale:
        # Load the shared floor-histogram helpers from the module FILE (not the
        # internnav package — its __init__ pulls training deps this torch-2.8
        # lingbot env doesn't have). The helpers only need torch/np here;
        # lingbot_map.utils.rotation is imported lazily at call time (on sys.path).
        import importlib.util
        _ls_path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../internnav/model/basemodel/memnav/lingbot_stream.py"))
        _spec = importlib.util.spec_from_file_location("_lingbot_stream_helpers", _ls_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        ground_helpers = (_mod.ground_frame_heights, _mod.ground_h_est_from_heights)

    all_trajectories = find_trajectories(args.root_dirs)
    total = len(all_trajectories)
    trajectories = select_trajectories(
        all_trajectories, args.root_dirs, args.trajectory_list
    )
    selected = len(trajectories)
    if args.num_shards > 1:
        assert 0 <= args.shard < args.num_shards, f"--shard {args.shard} out of [0, {args.num_shards})"
        trajectories = trajectories[args.shard::args.num_shards]  # interleaved for load balance
    if args.limit:
        trajectories = trajectories[:args.limit]
    validate_frame_capacity(trajectories, args.max_frame_num)
    print(f"Found {total} trajectories under {args.root_dirs}; "
          f"selected {selected}; "
          f"processing {len(trajectories)} on shard {args.shard}/{args.num_shards}")

    device = torch.device(args.device)
    autocast_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    model = build_model(args, device)
    if (
        args.auto_keyframe_interval or args.keyframe_interval > 1
    ) and not hasattr(model, "_set_skip_append"):
        raise RuntimeError(
            "sparse keyframe precompute requires LingBot GCTStream._set_skip_append"
        )
    if flow_mode and getattr(model, "depth_head", None) is None:
        raise RuntimeError(
            "flow-gated keyframe selection requires the LingBot depth head"
        )

    precompute_config_json, precompute_signature = _precompute_provenance(args)
    print(f"precompute_signature={precompute_signature}")
    print(f"precompute_config={precompute_config_json}")

    # Forward hook on the DINOv2 patch embedder to capture the context-free
    # descriptor as it is computed inside _embed_images.
    dino_capture = [None]

    def hook(_module, _inp, out):
        dino_capture[0] = out

    handle = model.aggregator.patch_embed.register_forward_hook(hook)

    n_done, n_skip, n_err = 0, 0, 0
    for traj_dir, rgb_dir, rgb_paths in tqdm(trajectories, desc="trajectories"):
        chunk_dir = os.path.dirname(rgb_dir.rstrip("/"))
        if args.out_root:
            rel = os.path.relpath(chunk_dir, args.root_dirs)
            dst_dir = os.path.join(args.out_root, rel)
            os.makedirs(dst_dir, exist_ok=True)
        else:
            dst_dir = chunk_dir
        out_path = os.path.join(dst_dir, args.out_name)
        cam_path = os.path.join(dst_dir, args.cam_out_name)
        gate_path = cam_path if args.cam_only else out_path
        keyframe_interval = _resolve_keyframe_interval(args, len(rgb_paths))
        if os.path.exists(gate_path) and not args.overwrite:
            if not os.path.isfile(out_path) or not os.path.isfile(cam_path):
                raise RuntimeError(
                    f"partial cache pair at {dst_dir}; remove only after review or "
                    "rerun with --overwrite"
                )
            layout = validate_cache_files(
                out_path,
                cam_path,
                expected_num_frames=len(rgb_paths),
                expected_num_scale_frames=min(args.num_scale_frames, len(rgb_paths)),
                expected_sliding_window=args.kv_cache_sliding_window,
                require_versioned=True,
            )
            if layout.precompute_signature != precompute_signature:
                raise RuntimeError(
                    f"existing cache signature differs at {dst_dir}: "
                    f"{layout.precompute_signature} != {precompute_signature}; "
                    "use a new out_root rather than mixing precompute runs"
                )
            expected_policy = KEYFRAME_POLICY_FLOW if flow_mode else KEYFRAME_POLICY
            if layout.keyframe_policy != expected_policy:
                raise RuntimeError(
                    f"existing cache keyframe policy differs at {dst_dir}: "
                    f"{layout.keyframe_policy} != {expected_policy}"
                )
            if layout.keyframe_interval != keyframe_interval:
                raise RuntimeError(
                    f"existing cache interval differs at {dst_dir}: "
                    f"{layout.keyframe_interval} != {keyframe_interval}"
                )
            n_skip += 1
            continue
        try:
            if args.cam_only:
                if not os.path.isfile(out_path):
                    raise RuntimeError(
                        f"--cam_only requires an existing versioned aggregator cache: {out_path}"
                    )
                with np.load(out_path, allow_pickle=False) as existing:
                    required = {
                        "cache_schema_version", "precompute_signature",
                        "keyframe_interval", "num_frames",
                    }
                    missing = sorted(required - set(existing.files))
                    if missing:
                        raise RuntimeError(
                            f"--cam_only cannot pair with legacy/incomplete aggregator "
                            f"{out_path}; missing={missing}; run a full precompute"
                        )
                    if str(existing["precompute_signature"].item()) != precompute_signature:
                        raise RuntimeError(
                            "--cam_only precompute signature differs from aggregator; "
                            "run a full precompute into a new out_root"
                        )
                    if int(existing["keyframe_interval"].item()) != keyframe_interval:
                        raise RuntimeError("--cam_only keyframe interval differs from aggregator")
                    if int(existing["num_frames"].item()) != len(rgb_paths):
                        raise RuntimeError("--cam_only frame count differs from aggregator")
            images = load_and_preprocess_images(
                rgb_paths,
                mode=args.preprocess_mode,
                image_size=args.image_size,
                patch_size=args.patch_size,
            )  # [N, 3, H, W] in [0, 1]
            images = images.unsqueeze(0)  # [1, N, 3, H, W]
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(args.dtype != "fp32")):
                feats = extract_trajectory(
                    model, images, args.num_scale_frames, dino_capture,
                    cam_only=args.cam_only, skip_scale=args.skip_scale,
                    keyframe_interval=keyframe_interval,
                    ground_helpers=ground_helpers, ground_stride=args.ground_stride,
                    flow_threshold=args.flow_threshold,
                    max_non_keyframe_gap=args.max_non_keyframe_gap,
                )
            if flow_mode:
                n_kf = int(len(feats["anchor_frame_indices"]))
                n_streamed = max(0, len(rgb_paths) - min(args.num_scale_frames, len(rgb_paths)))
                tqdm.write(
                    f"  flow-gated keyframes: {n_kf}/{n_streamed} "
                    f"({(100.0 * n_kf / n_streamed) if n_streamed else 0.0:.1f}%) "
                    f"at thr={args.flow_threshold:g}px gap<={args.max_non_keyframe_gap} "
                    f"for {os.path.relpath(traj_dir, args.root_dirs)}"
                )
            for name in ("cam_k", "cam_v", "cam_pose_enc"):
                assert np.isfinite(feats[name]).all(), f"non-finite {name}"
            shared_metadata = dict(
                cache_schema_version=np.array([CACHE_SCHEMA_VERSION], dtype=np.int64),
                keyframe_policy=np.array(
                    [KEYFRAME_POLICY_FLOW if flow_mode else KEYFRAME_POLICY]
                ),
                flow_threshold=np.array([float(args.flow_threshold)], dtype=np.float64),
                max_non_keyframe_gap=np.array(
                    [int(args.max_non_keyframe_gap)], dtype=np.int64
                ),
                num_frames=np.array([len(rgb_paths)], dtype=np.int64),
                num_scale_frames=np.array(
                    [min(args.num_scale_frames, len(rgb_paths))], dtype=np.int64
                ),
                keyframe_interval=np.array([keyframe_interval], dtype=np.int64),
                kv_cache_sliding_window=np.array(
                    [args.kv_cache_sliding_window], dtype=np.int64
                ),
                precompute_signature=np.array([precompute_signature]),
                precompute_config_json=np.array([precompute_config_json]),
            )
            # Camera-head cache (always) — small; np.savez (ZIP_STORED) avoids slow deflate.
            # ATOMIC write: savez into a .tmp *file handle* (writing to a handle skips numpy's
            # ".npz" suffix munging), fsync, then os.replace. A crash mid-write (node death,
            # timeout, OOM) leaves only a .tmp the skip-if-exists gate ignores — never a
            # truncated final cache that would be silently treated as "done".
            _atomic_savez(cam_path, cam_k=feats["cam_k"], cam_v=feats["cam_v"],
                          cam_pose_enc=feats["cam_pose_enc"],
                          cam_frame_indices=feats["cam_frame_indices"],
                          cam_meta=feats["cam_meta"],
                          **({"ground_h_est": feats["ground_h_est"],
                              "ground_dbg": feats["ground_dbg"]}
                             if "ground_h_est" in feats else {}),
                          **shared_metadata)
            if not args.cam_only:
                assert np.isfinite(feats["dino_cls"]).all(), "non-finite dino_cls"
                assert np.isfinite(feats["anchor_k"]).all(), "non-finite anchor_k"
                assert np.isfinite(feats["anchor_v"]).all(), "non-finite anchor_v"
                save_kwargs = dict(
                    dino_cls=feats["dino_cls"].astype(np.float16),
                    anchor_k=feats["anchor_k"], anchor_v=feats["anchor_v"],
                    anchor_frame_indices=feats["anchor_frame_indices"],
                    meta=feats["meta"],
                    **shared_metadata,
                )
                if not args.skip_scale:
                    assert np.isfinite(feats["scale_k"]).all(), "non-finite scale_k"
                    save_kwargs["scale_k"] = feats["scale_k"]
                    save_kwargs["scale_v"] = feats["scale_v"]
                # out_path (gate file) written LAST + atomically, so it appears only once complete.
                _atomic_savez(out_path, **save_kwargs)
            layout = validate_cache_files(
                out_path,
                cam_path,
                expected_num_frames=len(rgb_paths),
                expected_num_scale_frames=min(args.num_scale_frames, len(rgb_paths)),
                expected_sliding_window=args.kv_cache_sliding_window,
                require_versioned=True,
            )
            if (
                layout.precompute_signature != precompute_signature
                or layout.keyframe_interval != keyframe_interval
                or layout.keyframe_policy
                != (KEYFRAME_POLICY_FLOW if flow_mode else KEYFRAME_POLICY)
            ):
                raise RuntimeError("post-write cache metadata validation failed")
            n_done += 1
        except Exception as e:  # noqa: BLE001 — keep going, report at the end
            n_err += 1
            print(f"[ERROR] {traj_dir}: {e}")
            traceback.print_exc()

    handle.remove()
    print(f"Done. computed={n_done} skipped={n_skip} errors={n_err}")
    if n_err:
        raise RuntimeError(
            f"LingBot precomputation failed for {n_err} trajectory/trajectories; "
            "see the tracebacks above. Partial successful outputs are retained and "
            "a rerun will skip them."
        )


if __name__ == "__main__":
    main()
