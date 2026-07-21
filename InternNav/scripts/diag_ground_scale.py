"""Validate ground-anchored per-trajectory metric scale recovery.

Compares, per validate_gated episode:
  s_ground — LingBotStream.compute_metric_scale: FULL frozen depth head on the
             num_scale scale-frames, unproject to the map frame with the
             continuous-stream poses, floor = dominant height-histogram peak below
             the cameras, s = camera_height_m / est_camera_to_floor
             (VGP-Nav arXiv:2606.09268 sec III-E).
  s_gt     — the Umeyama Sim(3) scale fitted between the SAME continuous-stream
             positions (cam_pose_enc) and the episode's GT parquet extrinsics —
             the true lingbot-units -> meters factor for that trajectory.

If s_ground/s_gt ~= 1 across episodes, the recovery (and the +y-down gravity-axis
assumption for this checkpoint's map frame) is validated; a consistent sign/axis
error would show up as garbage ratios or all-None scales.

It then validates the quantity the policy actually consumes: for sampled frame
pairs (k, g) it computes the RELATIVE translation t_rel = R_k^T (t_g - t_k) from
cam_pose_enc (exactly RevisitMerge._relative_pose), rescales it by s_ground, and
compares its planar norm |t_rel_xz| against the GT pair's metric displacement
|t_gt[g] - t_gt[k]| — per-pair error in meters, no Sim(3) fit involved.

Usage (memnav conda env, from InternNav/):
  python scripts/diag_ground_scale.py
  python scripts/diag_ground_scale.py --episodes mp3d_2leg/17DRP5sb8fy/episode_0000 \
      --skip_precompute /tmp/lingbot_pose_eval_out_XXXX
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from diag_lingbot_pose_accuracy import (  # noqa: E402  (same scripts/ dir)
    DEFAULT_ROOT, LINGBOT_REPO, WEIGHTS,
    run_precompute, load_gt_positions, load_official_positions, umeyama,
)
from internnav.model.basemodel.memnav.lingbot_stream import (  # noqa: E402
    LingBotStream, ground_scale_from_h_est)

DEFAULT_EPISODES = ",".join(
    f"{group}/{scene}/episode_{i:04d}"
    for group in ("mp3d_2leg",)
    for scene in ("17DRP5sb8fy", "1LXtFkjw3qL")
    for i in range(3)
)


def quat_to_mat_np(q):
    """xyzw (non-unit ok) -> rotation matrix, batched [N,4] -> [N,3,3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def relpose_check(cam_pose_enc, gt_pos, s_ground, n_pairs=200, min_gap=50, max_gap=300,
                  seed=0):
    """Rescaled-relative-translation vs GT displacement over random (k, g) pairs.
    Lingbot side uses the PLANAR norm |t_rel_{x,z}| (y = height axis, and the fitted
    axis conversion maps (z, -x) -> data (x, y)); GT motion is planar so its 3D
    displacement norm is the matching metric quantity."""
    S = min(len(cam_pose_enc), len(gt_pos))
    rng = np.random.default_rng(seed)
    ks = rng.integers(0, S - min_gap - 1, n_pairs)
    gs = np.minimum(ks + rng.integers(min_gap, max_gap, n_pairs), S - 1)
    t = cam_pose_enc[:S, :3].astype(np.float64)
    R = quat_to_mat_np(cam_pose_enc[:S, 3:7].astype(np.float64))
    t_rel = np.einsum("nji,nj->ni", R[ks], t[gs] - t[ks])       # R_k^T (t_g - t_k)
    lb_norm = np.linalg.norm(t_rel[:, [0, 2]], axis=-1)          # planar (x,z)
    gt_norm = np.linalg.norm(gt_pos[gs] - gt_pos[ks], axis=-1)
    keep = gt_norm > 0.5                                         # skip near-zero pairs
    err_m = np.abs(s_ground * lb_norm[keep] - gt_norm[keep])
    rel_err = err_m / gt_norm[keep]
    return dict(n=int(keep.sum()),
                gt_med=float(np.median(gt_norm[keep])),
                err_med=float(np.median(err_m)), err_p90=float(np.percentile(err_m, 90)),
                rel_med=float(np.median(rel_err)), rel_p90=float(np.percentile(rel_err, 90)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--episodes", default=DEFAULT_EPISODES)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--num_scale", type=int, default=8)
    ap.add_argument("--max_frame_num", type=int, default=2048)
    ap.add_argument("--scale_frames", type=int, default=64,
                    help="frames pooled into the height histogram (compute_metric_scale n_frames)")
    ap.add_argument("--peak_thresh", type=float, default=0.3,
                    help="deepest-peak significance threshold (fraction of max bin; 1.0 = argmax)")
    ap.add_argument("--camera_height", type=float, default=None,
                    help="override; default reads gen_meta.json camera_height_m (0.5 fallback)")
    ap.add_argument("--skip_precompute", default=None,
                    help="reuse an existing out_root from a prior run (also of diag_lingbot_pose_accuracy)")
    args = ap.parse_args()
    episodes = args.episodes.split(",")

    if args.skip_precompute:
        out_root = args.skip_precompute
    else:
        out_root = run_precompute(args.root, episodes, args.window, args.num_scale,
                                  args.max_frame_num, overwrite=False)

    lb = LingBotStream(lingbot_repo=LINGBOT_REPO, weights=WEIGHTS, window=args.window,
                       num_scale=args.num_scale, max_frame_num=args.max_frame_num)

    rows = []
    for ep in episodes:
        group, scene, traj = ep.split("/")
        traj_dir = os.path.join(args.root, ep)
        rgb_dir = os.path.join(traj_dir, "videos/chunk-000/observation.images.rgb")
        cam_cache_path = os.path.join(out_root, group, scene, traj,
                                      "videos/chunk-000/lingbot_cam_cache.npz")
        cc = np.load(cam_cache_path)
        cam_pose_enc = cc["cam_pose_enc"]
        # whole-episode estimate stored by the new precompute (the actual train-time path)
        h_cache = float(cc["ground_h_est"]) if "ground_h_est" in cc.files else None
        if h_cache is not None and not np.isfinite(h_cache):
            h_cache = None

        # GT scale: Umeyama fit of the trusted continuous-stream positions vs parquet GT
        gt = load_gt_positions(traj_dir)
        official = load_official_positions(cam_cache_path)
        S = min(len(gt), len(official))
        s_gt, _R, _t = umeyama(official[:S], gt[:S])

        cam_h = args.camera_height
        if cam_h is None:
            meta = json.load(open(os.path.join(traj_dir, "meta/gen_meta.json")))
            cam_h = float(meta.get("camera_height_m", 0.5))

        paths = [os.path.join(rgb_dir, f"{i}.jpg") for i in range(args.scale_frames)]
        with torch.no_grad():
            s_ground, dbg = lb.compute_metric_scale(paths, cam_pose_enc, cam_h,
                                                    n_frames=args.scale_frames,
                                                    peak_thresh=args.peak_thresh,
                                                    return_debug=True)
        ratio = (s_ground / s_gt) if s_ground else float("nan")
        cache_note = ""
        if h_cache is not None:
            s_cache = ground_scale_from_h_est(h_cache, cam_h)
            cache_note = (f" | CACHE h_est={h_cache:.4f} s_cache="
                          f"{('%.3f' % s_cache) if s_cache else 'None'} "
                          f"ratio_cache={(s_cache / s_gt) if s_cache else float('nan'):.3f}")
        print(f"[{ep}] cam_h={cam_h}m s_gt={s_gt:.3f} s_ground="
              f"{('%.3f' % s_ground) if s_ground else 'None'} ratio={ratio:.3f} | "
              f"h_est={dbg['h_est']} n_valid={dbg['n_valid']}/{dbg['n_frames']} "
              f"h_iqr={dbg['h_iqr']}{cache_note}")
        if s_ground:
            # end-to-end: the policy's t_rel, rescaled, vs GT metric displacement.
            # The s_gt row is the ORACLE-scale baseline — its residual error is
            # LingBot pose noise alone, so (s_ground row - s_gt row) isolates the
            # error attributable to the recovered scale.
            for name, sc in (("s_ground", s_ground), ("s_gt (oracle)", s_gt)):
                rc = relpose_check(cam_pose_enc, gt, sc)
                print(f"    t_rel x {name:14s} vs GT, {rc['n']} pairs "
                      f"(median |GT|={rc['gt_med']:.2f}m): err median={rc['err_med']:.3f}m "
                      f"p90={rc['err_p90']:.3f}m | rel err median={rc['rel_med']:.1%} "
                      f"p90={rc['rel_p90']:.1%}")
        rows.append((ep, s_gt, s_ground, ratio))

    ok = [r for _, _, s, r in rows if s]
    print(f"\n=== {len(ok)}/{len(rows)} episodes recovered a scale ===")
    if ok:
        ok = np.array(ok)
        print(f"ratio s_ground/s_gt: median={np.median(ok):.3f} "
              f"mean={ok.mean():.3f} std={ok.std():.3f} "
              f"range=[{ok.min():.3f}, {ok.max():.3f}]")
    print(f"(caches at {out_root} — pass --skip_precompute {out_root} to rerun instantly)")


if __name__ == "__main__":
    main()
