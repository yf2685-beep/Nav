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
import os
import sys

import numpy as np
import torch
from tqdm import tqdm


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
        print(f"  loaded weights: {len(missing)} missing, {len(unexpected)} unexpected keys")
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Per-trajectory KV-cache + CLS capture
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_trajectory(model, images, scale_frames, dino_capture, cam_only=False, skip_scale=False):
    """Stream `images` [1, S, 3, H, W] through the aggregator + camera head,
    capturing per-frame KV caches (and the DINOv2 CLS token).

    Aggregator cache (skipped when `cam_only`):
      * ``scale_k`` / ``scale_v``  [L, H, scale, P, d]   full K/V of the `scale` block.
      * ``anchor_k`` / ``anchor_v`` [S-scale, L, H, 6, d] each later frame's specials.
      * ``dino_cls`` [S, D']  context-free DINOv2 CLS token (symmetric match key).

    Camera-head cache (always — the pose-specialized feature for revisit/aux-pose):
      * ``cam_k`` / ``cam_v`` [S, NI, TD, H, d]  the single camera token's K/V across
            num_iterations × trunk_depth, captured the step each frame is current.
            Injected [0..m] at train/eval time so the camera head relocalizes the goal.
      * ``cam_pose_enc`` [S, 9]  the head's NATIVE streaming pose (absT, quaR, FoV)
            per frame — empirically decodes as cam-to-world (despite the VGGT-derived
            w2c docstring). Used for metric calibration (per-traj monocular scale /
            map alignment) and as exact anchor poses in the revisit sweep.

    The camera head is run via a direct ``camera_head(...)`` call right after each
    ``_aggregate_features`` (it manages its own frame_idx + KV cache); this avoids the
    depth/world-point heads that the full ``forward`` would also run.
    """
    B, S = images.shape[0], images.shape[1]
    assert B == 1
    scale = min(scale_frames, S)

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

    # Phase 1: scale frames as a single bidirectional block.
    scale_imgs = images[:, :scale].to(dev, non_blocking=True)
    scale_agg, psi = model._aggregate_features(scale_imgs, num_frame_for_scale=scale, num_frame_per_block=scale)
    pl = ch(scale_agg, causal_inference=True, num_frame_per_block=scale, num_frame_for_scale=scale)
    cam_pose_list.append(pl[-1][0].float().cpu())            # [scale, 9]
    ck, cv = read_cam_newest(scale); cam_k_list.append(ck); cam_v_list.append(cv)
    if not cam_only:
        cls_list.append(pop_cls(scale))
        if not skip_scale:
            scale_k, scale_v = read_cache_full(slice(0, scale))  # [L,H,scale,P,d]

    # Phase 2: causal streaming, one frame at a time.
    anchor_k_list, anchor_v_list = [], []
    for i in range(scale, S):
        frame = images[:, i:i + 1].to(dev, non_blocking=True)
        agg_tok, _ = model._aggregate_features(frame, num_frame_for_scale=scale, num_frame_per_block=1)
        pl = ch(agg_tok, causal_inference=True, num_frame_per_block=1, num_frame_for_scale=scale)
        cam_pose_list.append(pl[-1][0].float().cpu())        # [1, 9]
        ck, cv = read_cam_newest(1); cam_k_list.append(ck); cam_v_list.append(cv)
        if not cam_only:
            cls_list.append(pop_cls(1))
            ak, av = read_cache_anchor_newest(psi)
            anchor_k_list.append(ak); anchor_v_list.append(av)

    model.clean_kv_cache()

    Hh, d = ck.shape[3], ck.shape[-1]
    cam_k = torch.cat(cam_k_list, 0).numpy()                     # [S, NI, TD, H, d]
    cam_v = torch.cat(cam_v_list, 0).numpy()
    out = {"cam_k": cam_k, "cam_v": cam_v,
           "cam_pose_enc": torch.cat(cam_pose_list, 0).numpy(),   # [S, 9]
           "cam_meta": np.array([NI, TD, Hh, d], dtype=np.int64)}
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
    ap.add_argument("--enable_3d_rope", action="store_true", default=True,
                    help="Temporal 3D RoPE (LingBot's intended mode). Needed for goal time-index placement.")
    ap.add_argument("--max_frame_num", type=int, default=1024,
                    help="Max frames for 3D RoPE (>= longest trajectory; dataset max is 342).")
    ap.add_argument("--camera_num_iterations", type=int, default=4)
    ap.add_argument("--use_sdpa", action="store_true", default=False,
                    help="Use SDPA attention (no FlashInfer dependency). Recommended.")
    ap.add_argument("--preprocess_mode", default="pad", choices=["pad", "crop"],
                    help="LingBot image preprocess mode (square pad to image_size).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_name", default="lingbot_cache.npz",
                    help="Output filename written next to each trajectory's videos/chunk-000/.")
    ap.add_argument("--cam_out_name", default="lingbot_cam_cache.npz",
                    help="Camera-head KV cache filename (cam_k/cam_v/cam_meta).")
    ap.add_argument("--cam_only", action="store_true",
                    help="Capture ONLY the camera-head cache (keep the existing aggregator npz untouched).")
    ap.add_argument("--skip_scale", action="store_true",
                    help="Skip writing scale_k/scale_v (~1.08 GB/traj). Training recomputes them "
                         "on the fly from the first num_scale RGB frames via "
                         "LingBotStream.get_scale_kv (LRU-cached).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Recompute even if the output npz already exists.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Process at most N trajectories (0 = all). For smoke tests.")
    ap.add_argument("--num_shards", type=int, default=1,
                    help="Divide the traj list into this many interleaved shards.")
    ap.add_argument("--shard", type=int, default=0,
                    help="Which shard [0, num_shards) to process. Combine with SLURM array to parallelize.")
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    args = ap.parse_args()

    sys.path.insert(0, args.lingbot_repo)
    from lingbot_map.utils.load_fn import load_and_preprocess_images

    device = torch.device(args.device)
    autocast_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]

    model = build_model(args, device)

    # Forward hook on the DINOv2 patch embedder to capture the context-free
    # descriptor as it is computed inside _embed_images.
    dino_capture = [None]

    def hook(_module, _inp, out):
        dino_capture[0] = out

    handle = model.aggregator.patch_embed.register_forward_hook(hook)

    trajectories = find_trajectories(args.root_dirs)
    total = len(trajectories)
    if args.num_shards > 1:
        assert 0 <= args.shard < args.num_shards, f"--shard {args.shard} out of [0, {args.num_shards})"
        trajectories = trajectories[args.shard::args.num_shards]  # interleaved for load balance
    if args.limit:
        trajectories = trajectories[:args.limit]
    print(f"Found {total} trajectories under {args.root_dirs}; "
          f"processing {len(trajectories)} on shard {args.shard}/{args.num_shards}")

    n_done, n_skip, n_err = 0, 0, 0
    for traj_dir, rgb_dir, rgb_paths in tqdm(trajectories, desc="trajectories"):
        chunk_dir = os.path.dirname(rgb_dir.rstrip("/"))
        out_path = os.path.join(chunk_dir, args.out_name)
        cam_path = os.path.join(chunk_dir, args.cam_out_name)
        gate_path = cam_path if args.cam_only else out_path
        if os.path.exists(gate_path) and not args.overwrite:
            n_skip += 1
            continue
        try:
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
                )
            assert np.isfinite(feats["cam_k"]).all(), "non-finite cam_k"
            # Camera-head cache (always) — small; np.savez (ZIP_STORED) avoids slow deflate.
            np.savez(cam_path, cam_k=feats["cam_k"], cam_v=feats["cam_v"],
                     cam_pose_enc=feats["cam_pose_enc"], cam_meta=feats["cam_meta"])
            if not args.cam_only:
                assert np.isfinite(feats["dino_cls"]).all(), "non-finite dino_cls"
                save_kwargs = dict(
                    dino_cls=feats["dino_cls"].astype(np.float16),
                    anchor_k=feats["anchor_k"], anchor_v=feats["anchor_v"],
                    meta=feats["meta"],
                )
                if not args.skip_scale:
                    assert np.isfinite(feats["scale_k"]).all(), "non-finite scale_k"
                    save_kwargs["scale_k"] = feats["scale_k"]
                    save_kwargs["scale_v"] = feats["scale_v"]
                np.savez(out_path, **save_kwargs)
            n_done += 1
        except Exception as e:  # noqa: BLE001 — keep going, report at the end
            n_err += 1
            print(f"[ERROR] {traj_dir}: {e}")

    handle.remove()
    print(f"Done. computed={n_done} skipped={n_skip} errors={n_err}")


if __name__ == "__main__":
    main()
