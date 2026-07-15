"""Compare GT vs LingBot-predicted camera trajectories on generate_twoleg episodes.

Three trajectories per episode:
  (1) GT       — camera-to-world extrinsics from the episode's parquet.
  (2) official — `cam_pose_enc` captured by precompute_lingbot_features.extract_trajectory's
                 continuous causal stream: scale block, then one frame at a time, through the
                 SAME GCTStream aggregator + camera head (same built-in scale/sliding-window/
                 specials eviction as GCTStream.inference_streaming — see attention.py's
                 _apply_kv_cache_eviction_causal). This is the trusted, unbroken-stream reference.
  (3) ours     — internnav.model.basemodel.memnav.lingbot_stream.LingBotStream.window_forward +
                 .camera_pose, driven at a strided set of steps k against the precomputed cache —
                 the exact reconstruction-from-snapshot path MemNavNet.encode_memory uses in
                 production (recompute the live window fresh from raw pixels on top of the
                 injected scale + specials-only history).

pose9's absT is used directly as the camera position, NOT inverted via a world-to-camera
convention: precompute_lingbot_features.py's extract_trajectory docstring notes cam_pose_enc
"empirically decodes as cam-to-world (despite the VGGT-derived w2c docstring)" for this
checkpoint, and RevisitMerge/_pose7 already treats it the same way.

LingBot's map frame has an arbitrary origin/rotation/scale vs. the dataset's world frame, so a
raw overlay is meaningless — trajectories are aligned to GT via a closed-form Sim(3) (Umeyama)
fit BEFORE plotting/scoring. The Sim(3) is fit ONCE on official-vs-GT (the trusted reference)
and the SAME transform is then applied to `ours`, so a real bug in the windowed reconstruction
shows up as elevated ours-vs-GT error rather than being silently absorbed by a separate best-fit.

Usage (from InternNav/, memnav conda env, needs pandas/torch/matplotlib):
  python scripts/diag_lingbot_pose_accuracy.py \
      --episodes mp3d_2leg/17DRP5sb8fy/episode_0000,mp3d_2leg/1LXtFkjw3qL/episode_0000
"""
import argparse
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

DEFAULT_ROOT = "/home/asus/Research/Nav/memnav_viz/validate_gated"
DEFAULT_EPISODES = "mp3d_2leg/17DRP5sb8fy/episode_0000,mp3d_2leg/1LXtFkjw3qL/episode_0000"
LINGBOT_REPO = "/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map"
WEIGHTS = os.path.join(LINGBOT_REPO, "weights/lingbot-map-long.pt")
PRECOMPUTE_SCRIPT = os.path.join(os.path.dirname(__file__), "dataset_converters", "precompute_lingbot_features.py")
OUT_DIR = "/home/asus/Research/Nav/memnav_viz/lingbot_pose_eval"


# --------------------------------------------------------------------------- #
# (1) precompute — real production cache, restricted to the chosen episodes
# --------------------------------------------------------------------------- #
def run_precompute(root, episodes, window, num_scale, max_frame_num, overwrite):
    """Symlink the chosen episodes into a scratch root (group/scene/traj layout, what
    find_trajectories expects) and run the real precompute CLI on just those, with the
    same settings production training uses (see precompute_mp3d_pt1.sbatch)."""
    tmp_root = tempfile.mkdtemp(prefix="lingbot_pose_eval_src_")
    out_root = tempfile.mkdtemp(prefix="lingbot_pose_eval_out_")
    for ep in episodes:
        group, scene, traj = ep.split("/")
        dst_dir = os.path.join(tmp_root, group, scene)
        os.makedirs(dst_dir, exist_ok=True)
        os.symlink(os.path.join(root, ep), os.path.join(dst_dir, traj))
    cmd = [
        sys.executable, PRECOMPUTE_SCRIPT,
        "--root_dirs", tmp_root, "--out_root", out_root,
        "--lingbot_repo", LINGBOT_REPO, "--weights", WEIGHTS,
        "--kv_cache_sliding_window", str(window), "--num_scale_frames", str(num_scale),
        "--max_frame_num", str(max_frame_num),
        "--skip_scale", "--use_sdpa",
    ]
    if overwrite:
        cmd.append("--overwrite")
    print("[precompute]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return out_root


# --------------------------------------------------------------------------- #
# (2) trajectory loaders
# --------------------------------------------------------------------------- #
def load_gt_positions(traj_dir):
    df = pd.read_parquet(os.path.join(traj_dir, "data/chunk-000/episode_000000.parquet"))
    traj = np.array([np.stack(f) for f in df["action"]], dtype=np.float64).reshape(-1, 4, 4)
    return traj[:, :3, 3]                                   # cam-to-world translation per frame [T,3]


def load_official_positions(cam_cache_path):
    c = np.load(cam_cache_path)
    return c["cam_pose_enc"].astype(np.float64)[:, :3]      # [S,3] absT, cam-to-world (empirical)


def build_cache_dict(lb, cache_path, cam_cache_path, rgb_dir):
    c = np.load(cache_path)
    if "scale_k" in c.files and "scale_v" in c.files:
        sk, sv, ak, av = lb._cache_to_layered(c["scale_k"], c["scale_v"], c["anchor_k"], c["anchor_v"], lb.device)
    else:
        sk, sv = lb.get_scale_kv(rgb_dir)
        ak = torch.as_tensor(c["anchor_k"], device=lb.device, dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
        av = torch.as_tensor(c["anchor_v"], device=lb.device, dtype=torch.bfloat16).permute(1, 2, 0, 3, 4).contiguous()
    cc = np.load(cam_cache_path)
    ck, cv = lb._cam_to_device(cc["cam_k"], cc["cam_v"], lb.device)
    return dict(scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av, cam_k=ck, cam_v=cv)


@torch.no_grad()
def load_ours_positions(lb, cache, rgb_dir, ks, gt=None, official=None, verbose=False):
    """window_forward + camera_pose at each k — the exact MemNavNet.encode_memory path."""
    W = lb.window
    out = []
    for k in ks:
        win_idx = list(range(k - W + 1, k + 1))
        win_img = lb.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in win_idx]).to(lb.device)
        _, cur_agg, _ = lb.window_forward(cache, win_img, k, return_multilayer=True)
        pose = lb.camera_pose(cache["cam_k"], cache["cam_v"], k, cur_agg)[-1]
        p = pose[:3].float().cpu().numpy()
        out.append(p)
        if verbose:
            n_hist = (k - W + 1) - lb.num_scale
            msg = f"  k={k:4d} n_hist={n_hist:4d} raw_pos={p.round(3).tolist()}"
            if official is not None:
                msg += f"  official_raw={official[k].round(3).tolist()}  raw_dist={np.linalg.norm(p - official[k]):.3f}"
            print(msg)
    return np.array(out)


@torch.no_grad()
def warm_forward(lb, cache, rgb_dir, k, warm):
    """Adapted from MemNavData/analysis/revisit_sweep_eval.py:warm_forward — recompute
    live starting `warm` frames before k (not just the nominal window), so k gets genuine
    local full-KV context instead of starting cold off the injected specials. Requires
    lb's underlying model kv_cache_sliding_window >= warm (+ scale) so nothing evicts
    mid-warmup — caller must construct `lb` with a large-enough `window` for this.
    Returns the agg list for frame k only (for camera_pose)."""
    start = max(lb.num_scale, k - warm + 1)
    n_hist = start - lb.num_scale
    lb._inject(cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
              n_hist=n_hist, total_frames=start)
    imgs = lb.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(start, k + 1)])
    a = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(len(imgs)):
            a, _ = lb.model._aggregate_features(
                imgs[j:j + 1][None].to(lb.device),
                num_frame_for_scale=lb.num_scale, num_frame_per_block=1)
    return [layer for layer in a]


@torch.no_grad()
def load_ours_warm_positions(lb, cache, rgb_dir, ks, warm, official=None, verbose=False):
    """warm_forward + camera_pose at each k — tests whether a deep-warmup live recompute
    (instead of window_forward's cold-start-at-the-window-boundary) closes the gap to official."""
    out = []
    for k in ks:
        agg = warm_forward(lb, cache, rgb_dir, k, warm)
        pose = lb.camera_pose(cache["cam_k"], cache["cam_v"], k, agg)[-1]
        p = pose[:3].float().cpu().numpy()
        out.append(p)
        if verbose:
            msg = f"  [warm={warm}] k={k:4d} raw_pos={p.round(3).tolist()}"
            if official is not None:
                msg += f"  official_raw={official[k].round(3).tolist()}  raw_dist={np.linalg.norm(p - official[k]):.3f}"
            print(msg)
    return np.array(out)


# --------------------------------------------------------------------------- #
# goal-insertion accuracy: the quantity that actually matters for RevisitMerge.
# Every frame in [m-W+1..m] is real and already has an exact cached pose (cam_pose_enc)
# — window_forward/warm_forward is NEVER needed to recover their poses. Its only remaining
# job is building aggregator context so the newly-inserted GOAL image (no cache entry)
# gets a reasonable camera_pose() estimate. These three functions measure exactly that,
# against the goal's true GT position (gen_meta.json goals[i]['pos']).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def production_goal_pose(lb, cache, rgb_dir, m, goal_img_path, window):
    """Current production behavior: goal_append's cold-started window_forward([m-W+1..m])
    then stream the goal at m+1. Equivalent to warm_goal_pose(..., warm=window)."""
    return warm_goal_pose(lb, cache, rgb_dir, m, window, goal_img_path)


@torch.no_grad()
def warm_goal_pose(lb, cache, rgb_dir, m, warm, goal_img_path):
    """warm_forward's context-building up to m, then stream the ACTUAL goal image at m+1
    and return camera_pose()'s estimate for it (not for m itself)."""
    start = max(lb.num_scale, m - warm + 1)
    n_hist = start - lb.num_scale
    lb._inject(cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
              n_hist=n_hist, total_frames=start)
    imgs = lb.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(start, m + 1)])
    goal_img = lb.load_images([goal_img_path])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(len(imgs)):
            lb.model._aggregate_features(imgs[j:j + 1][None].to(lb.device),
                                         num_frame_for_scale=lb.num_scale, num_frame_per_block=1)
        a, _ = lb.model._aggregate_features(goal_img[None].to(lb.device),
                                            num_frame_for_scale=lb.num_scale, num_frame_per_block=1)
    agg = [layer for layer in a]
    pose = lb.camera_pose(cache["cam_k"], cache["cam_v"], m + 1, agg)[-1]
    return pose[:3].float().cpu().numpy()


@torch.no_grad()
def oracle_goal_pose(lb_big, cache, rgb_dir, m, goal_img_path):
    """ORACLE upper bound: a true, unbroken continuous stream [num_scale..m] (lb_big must be
    constructed with kv_cache_sliding_window large enough that nothing evicts — i.e. this is
    exactly what precompute's extract_trajectory does, zero information loss), then insert
    the goal at m+1. Isolates whatever error remains even with perfect context — e.g. a
    render/real domain gap in the goal image itself, or the model's own scale ambiguity."""
    lb_big._inject(cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
                   n_hist=0, total_frames=lb_big.num_scale)
    imgs = lb_big.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(lb_big.num_scale, m + 1)])
    goal_img = lb_big.load_images([goal_img_path])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for j in range(len(imgs)):
            lb_big.model._aggregate_features(imgs[j:j + 1][None].to(lb_big.device),
                                             num_frame_for_scale=lb_big.num_scale, num_frame_per_block=1)
        a, _ = lb_big.model._aggregate_features(goal_img[None].to(lb_big.device),
                                                num_frame_for_scale=lb_big.num_scale, num_frame_per_block=1)
    agg = [layer for layer in a]
    pose = lb_big.camera_pose(cache["cam_k"], cache["cam_v"], m + 1, agg)[-1]
    return pose[:3].float().cpu().numpy()


# --------------------------------------------------------------------------- #
# (3) Sim(3) alignment (Umeyama, closed form) + ATE
# --------------------------------------------------------------------------- #
def umeyama(src, dst):
    """Fit dst ~= s * (R @ src) + t (least squares). Returns (s, R, t)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    cov = (dst_c.T @ src_c) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = (src_c ** 2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_src
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def apply_sim3(s, R, t, pts):
    return s * (pts @ R.T) + t


def ate(aligned, gt):
    return float(np.sqrt(((aligned - gt) ** 2).sum(-1).mean()))


# --------------------------------------------------------------------------- #
def process_episode(root, ep, lb, out_root, window, stride, max_k, lb_warm=None, warm=None):
    group, scene, traj = ep.split("/")
    traj_dir = os.path.join(root, ep)
    rgb_dir = os.path.join(traj_dir, "videos/chunk-000/observation.images.rgb")
    cache_dir = os.path.join(out_root, group, scene, traj, "videos/chunk-000")
    cache_path = os.path.join(cache_dir, "lingbot_cache.npz")
    cam_cache_path = os.path.join(cache_dir, "lingbot_cam_cache.npz")

    gt = load_gt_positions(traj_dir)
    official = load_official_positions(cam_cache_path)
    S = min(len(gt), len(official))
    gt, official = gt[:S], official[:S]

    anchor_margin = lb.num_scale + window - 1
    hi = S - 1 if max_k is None else min(S - 1, max_k)
    ks = list(range(anchor_margin, hi + 1, stride))
    print(f"[{ep}] S={S} frames, sampling {len(ks)} k's (stride={stride}) for 'ours'")

    cache = build_cache_dict(lb, cache_path, cam_cache_path, rgb_dir)
    ours = load_ours_positions(lb, cache, rgb_dir, ks, gt=gt, official=official, verbose=True)

    gt_ks, official_ks = gt[ks], official[ks]

    # Fit ONE Sim(3) on the trusted reference (official -> GT), apply it to both.
    s, R, t = umeyama(official_ks, gt_ks)
    official_aligned = apply_sim3(s, R, t, official_ks)
    ours_aligned = apply_sim3(s, R, t, ours)

    ate_official = ate(official_aligned, gt_ks)
    ate_ours = ate(ours_aligned, gt_ks)
    ate_ours_vs_official = ate(ours_aligned, official_aligned)
    print(f"[{ep}] fitted scale={s:.4f} | ATE official-vs-GT={ate_official:.3f}m "
          f"ours-vs-GT={ate_ours:.3f}m ours-vs-official={ate_ours_vs_official:.3f}m")

    result = dict(episode=ep, scale=s, ate_official=ate_official, ate_ours=ate_ours,
                  ate_ours_vs_official=ate_ours_vs_official)

    ours_warm_aligned = None
    if lb_warm is not None and warm is not None:
        cache_warm = build_cache_dict(lb_warm, cache_path, cam_cache_path, rgb_dir)
        ours_warm = load_ours_warm_positions(lb_warm, cache_warm, rgb_dir, ks, warm,
                                             official=official, verbose=True)
        ours_warm_aligned = apply_sim3(s, R, t, ours_warm)          # SAME fitted transform
        ate_ours_warm = ate(ours_warm_aligned, gt_ks)
        ate_ours_warm_vs_official = ate(ours_warm_aligned, official_aligned)
        print(f"[{ep}] [warm={warm}] ATE ours_warm-vs-GT={ate_ours_warm:.3f}m "
              f"ours_warm-vs-official={ate_ours_warm_vs_official:.3f}m")
        result.update(ate_ours_warm=ate_ours_warm, ate_ours_warm_vs_official=ate_ours_warm_vs_official)

    # official at full density (all S frames), aligned with the same fitted transform, for a smooth reference line
    official_full_aligned = apply_sim3(s, R, t, official)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(gt[:, 0], gt[:, 1], "-", color="black", lw=2, label="GT")
    ax.plot(official_full_aligned[:, 0], official_full_aligned[:, 1], "--", color="tab:blue", lw=1.5,
            label=f"official (ATE={ate_official:.2f}m)")
    ax.scatter(ours_aligned[:, 0], ours_aligned[:, 1], color="tab:red", s=18, zorder=5,
               label=f"ours, no warm (ATE={ate_ours:.2f}m)")
    if ours_warm_aligned is not None:
        ax.scatter(ours_warm_aligned[:, 0], ours_warm_aligned[:, 1], color="tab:green", marker="^", s=28, zorder=6,
                   label=f"ours, warm={warm} (ATE={result['ate_ours_warm']:.2f}m)")
    ax.scatter(*gt[0, :2], marker="^", color="black", s=80, zorder=7, label="start")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{group}/{scene}/{traj}  (scale fit={s:.3f}, ours-vs-official={ate_ours_vs_official:.2f}m)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{group}_{scene}_{traj}_bev.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{ep}] saved {out_path}")
    result["out_path"] = out_path
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--episodes", default=DEFAULT_EPISODES,
                    help="comma-separated group/scene/traj (relative to --root)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--num_scale", type=int, default=8)
    ap.add_argument("--max_frame_num", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=8, help="k-sampling stride for the 'ours' path")
    ap.add_argument("--max_k", type=int, default=None, help="cap the last k sampled (debug/speed)")
    ap.add_argument("--warm", type=int, default=None,
                    help="if set, also test warm_forward (deep live-recompute before k, "
                         "MemNavData/analysis/revisit_sweep_eval.py-style) at this depth")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip_precompute", default=None,
                    help="reuse an existing --out_root from a prior run instead of recomputing")
    args = ap.parse_args()

    episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]

    if args.skip_precompute:
        out_root = args.skip_precompute
    else:
        out_root = run_precompute(args.root, episodes, args.window, args.num_scale,
                                  args.max_frame_num, args.overwrite)

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # InternNav/ on path
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
    lb = LingBotStream(lingbot_repo=LINGBOT_REPO, weights=WEIGHTS, window=args.window,
                       num_scale=args.num_scale, max_frame_num=args.max_frame_num, device="cuda")

    lb_warm = None
    if args.warm is not None:
        # kv_cache_sliding_window must be >= warm (+ margin) so nothing evicts mid-warmup
        # (see MemNavData/analysis/revisit_sweep_eval.py: window=max(window,warm)+8).
        lb_warm = LingBotStream(lingbot_repo=LINGBOT_REPO, weights=WEIGHTS,
                                window=max(args.window, args.warm) + 8,
                                num_scale=args.num_scale, max_frame_num=args.max_frame_num, device="cuda")

    results = []
    for ep in episodes:
        results.append(process_episode(args.root, ep, lb, out_root, args.window, args.stride, args.max_k,
                                       lb_warm=lb_warm, warm=args.warm))

    print("\n=== summary ===")
    for r in results:
        line = (f"{r['episode']}: scale={r['scale']:.3f} ATE(official)={r['ate_official']:.3f}m "
                f"ATE(ours)={r['ate_ours']:.3f}m ATE(ours-vs-official)={r['ate_ours_vs_official']:.3f}m")
        if "ate_ours_warm" in r:
            line += (f" | ATE(ours_warm)={r['ate_ours_warm']:.3f}m "
                     f"ATE(ours_warm-vs-official)={r['ate_ours_warm_vs_official']:.3f}m")
        print(line + f" -> {r['out_path']}")
    print(f"\n(cache written to {out_root} — pass --skip_precompute {out_root} to rerun scoring without recomputing)")


if __name__ == "__main__":
    main()
