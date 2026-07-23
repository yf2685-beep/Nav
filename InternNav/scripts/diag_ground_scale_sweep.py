"""CPU-only, cache-driven validation of the baked-in ground_h_est over EVERY episode.

For each cached episode this reads:
  * cam cache (host feat tree): cam_pose_enc [S,9] (t + quat_xyzw + fov) and the
    whole-episode ground_h_est written by precompute_lingbot_features.py.
  * GT parquet (raw tree / sqf overlay): action[:, :3, 3] cam-to-world translation.

Then, with NO model and NO GPU, it computes per episode:
  s_gt      Umeyama Sim(3) scale between cam_pose_enc positions and the GT parquet
            positions -> the TRUE lingbot-units -> meters factor for that trajectory.
  s_naive   0.5 / ground_h_est                (the user's plain scale)
  s_corr    ground_scale_from_h_est(h_est)    (bias-corrected + range-gated)
  ratio     s_* / s_gt                        (== 1 iff the estimate is perfect)
  relerr    median | s_* * |t_rel_xz|  -  |GT displacement| | / |GT| over random
            frame pairs (k,g) -- the exact quantity the policy consumes, from
            t_rel = R_k^T (t_g - t_k), planar (x,z) norm.

"Consistent across episodes" == the ratio distribution is tight (low std / IQR).
We also split within-scene vs across-scene spread: if a per-scene depth bias
dominates, within-scene is tight but scene medians spread.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd

# replicate internnav.model.basemodel.memnav.lingbot_stream constants (standalone/CPU)
GROUND_BIAS_CORRECTION = 1.15
GROUND_SCALE_RANGE = (0.8, 4.0)
CAMERA_HEIGHT_M = 0.5


def ground_scale_from_h_est(h_est, camera_height_m=CAMERA_HEIGHT_M,
                            bias_correction=GROUND_BIAS_CORRECTION,
                            scale_range=GROUND_SCALE_RANGE):
    if h_est is None or not np.isfinite(h_est) or h_est <= 1e-6:
        return None
    s = float(bias_correction * camera_height_m / h_est)
    if not (scale_range[0] <= s <= scale_range[1]):
        return None
    return s


def umeyama(src, dst):
    """Sim(3) fit dst ~= s R src + t. Returns (s, R, t)."""
    src = np.asarray(src, float); dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = src - mu_s; D = dst - mu_d
    cov = D.T @ S / len(src)
    U, d, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = float(d.sum() / var_s)
    t = mu_d - s * R @ mu_s
    return s, R, t


def quat_to_mat_np(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def load_gt_positions(traj_dir):
    df = pd.read_parquet(os.path.join(traj_dir, "data/chunk-000/episode_000000.parquet"))
    traj = np.array([np.stack(f) for f in df["action"]], dtype=np.float64).reshape(-1, 4, 4)
    return traj[:, :3, 3]


def relpose_check(cam_pose_enc, gt_pos, s, n_pairs=200, min_gap=50, max_gap=300, seed=0):
    S = min(len(cam_pose_enc), len(gt_pos))
    if S <= min_gap + 2:
        return None
    rng = np.random.default_rng(seed)
    ks = rng.integers(0, S - min_gap - 1, n_pairs)
    gs = np.minimum(ks + rng.integers(min_gap, max_gap, n_pairs), S - 1)
    t = cam_pose_enc[:S, :3].astype(np.float64)
    R = quat_to_mat_np(cam_pose_enc[:S, 3:7].astype(np.float64))
    t_rel = np.einsum("nji,nj->ni", R[ks], t[gs] - t[ks])
    lb_norm = np.linalg.norm(t_rel[:, [0, 2]], axis=-1)
    gt_norm = np.linalg.norm(gt_pos[gs] - gt_pos[ks], axis=-1)
    keep = gt_norm > 0.5
    if keep.sum() < 10:
        return None
    rel_err = np.abs(s * lb_norm[keep] - gt_norm[keep]) / gt_norm[keep]
    return float(np.median(rel_err))


def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return {}
    q1, q3 = np.percentile(x, [25, 75])
    return dict(n=len(x), median=float(np.median(x)), mean=float(x.mean()),
                std=float(x.std()), iqr=float(q3 - q1),
                lo=float(x.min()), hi=float(x.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_root", default="/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data")
    ap.add_argument("--raw_root", default="/mp3d_revisit_v0/vln_n1/traj_data",
                    help="where the GT parquets live (sqf overlay path by default)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    caches = sorted(glob.glob(os.path.join(args.feat_root, "*/*/*/videos/chunk-000/lingbot_cam_cache.npz")))
    if args.limit:
        caches = caches[:args.limit]
    print(f"found {len(caches)} cam caches under {args.feat_root}")

    rows = []
    n_no_h, n_no_gt, n_gated = 0, 0, 0
    for cpath in caches:
        rel = os.path.relpath(cpath, args.feat_root)          # group/scene/ep/videos/chunk-000/...
        group, scene, ep = rel.split(os.sep)[:3]
        try:
            cc = np.load(cpath)
        except Exception:
            continue
        if "ground_h_est" not in cc.files:
            n_no_h += 1; continue
        h = float(cc["ground_h_est"])
        if not np.isfinite(h) or h <= 1e-6:
            n_no_h += 1; continue
        cam_pose_enc = cc["cam_pose_enc"]

        traj_dir = os.path.join(args.raw_root, group, scene, ep)
        pq = os.path.join(traj_dir, "data/chunk-000/episode_000000.parquet")
        if not os.path.exists(pq):
            n_no_gt += 1; continue
        try:
            gt = load_gt_positions(traj_dir)
        except Exception:
            n_no_gt += 1; continue

        official = cam_pose_enc[:, :3].astype(np.float64)
        S = min(len(gt), len(official))
        if S < 20:
            continue
        s_gt, _, _ = umeyama(official[:S], gt[:S])
        if not np.isfinite(s_gt) or s_gt <= 1e-6:
            continue

        s_naive = 0.5 / h
        s_corr = ground_scale_from_h_est(h)
        if s_corr is None:
            n_gated += 1
        rows.append(dict(
            group=group, scene=scene, ep=ep, S=int(S), h=h, s_gt=float(s_gt),
            s_naive=float(s_naive), s_corr=(float(s_corr) if s_corr else None),
            ratio_naive=float(s_naive / s_gt),
            ratio_corr=(float(s_corr / s_gt) if s_corr else None),
            relerr_naive=relpose_check(cam_pose_enc, gt, s_naive),
            relerr_corr=(relpose_check(cam_pose_enc, gt, s_corr) if s_corr else None),
        ))

    print(f"episodes used={len(rows)}  (skipped: no_h={n_no_h} no_gt={n_no_gt})  "
          f"corr-gated-to-None={n_gated}")
    if not rows:
        return

    rn = [r["ratio_naive"] for r in rows]
    rc = [r["ratio_corr"] for r in rows if r["ratio_corr"] is not None]
    en = [r["relerr_naive"] for r in rows if r["relerr_naive"] is not None]
    ec = [r["relerr_corr"] for r in rows if r["relerr_corr"] is not None]

    def show(name, d):
        if not d:
            print(f"  {name:22s}  (none)"); return
        print(f"  {name:22s}  n={d['n']:4d}  median={d['median']:.3f}  mean={d['mean']:.3f}  "
              f"std={d['std']:.3f}  IQR={d['iqr']:.3f}  range=[{d['lo']:.3f},{d['hi']:.3f}]")

    print("\n=== ratio = s_est / s_gt  (1.0 = perfect; tight spread = consistent) ===")
    show("naive 0.5/h_est", stats(rn))
    show("corrected (bias+gate)", stats(rc))
    print("\n=== relpose planar rel-err (policy-consumed quantity) ===")
    show("naive 0.5/h_est", stats(en))
    show("corrected (bias+gate)", stats(ec))

    # within-scene vs across-scene spread of the naive ratio
    by_scene = {}
    for r in rows:
        by_scene.setdefault((r["group"], r["scene"]), []).append(r["ratio_naive"])
    scene_meds = {k: float(np.median(v)) for k, v in by_scene.items()}
    within = np.concatenate([np.asarray(v) / np.median(v) for v in by_scene.values() if len(v) > 1])
    print(f"\n=== per-scene structure of naive ratio ({len(by_scene)} scenes) ===")
    sm = np.array(list(scene_meds.values()))
    print(f"  across-scene medians: std={sm.std():.3f} range=[{sm.min():.3f},{sm.max():.3f}]")
    print(f"  within-scene (ratio/scene-median): std={within.std():.3f}  "
          f"-> {'per-scene bias dominates' if within.std() < sm.std() else 'per-episode noise dominates'}")
    print("  scene medians (sorted):")
    for (g, sc), m in sorted(scene_meds.items(), key=lambda kv: kv[1]):
        print(f"    {m:.3f}  {g}/{sc}  (n={len(by_scene[(g, sc)])})")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(rows=rows, scene_medians={f"{g}/{s}": m for (g, s), m in scene_meds.items()}), f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
