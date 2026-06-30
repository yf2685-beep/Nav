"""Visualize a MemNav (current -> goal) action label, especially the SEEN/revisit case.

It reconstructs the trajectory straight from the parquet `action` column (SE(3)
camera-to-world poses) and reuses the dataset's OWN `process_actions` /
`xyz_to_xyt`, so the local-frame waypoints + headings shown are byte-for-byte the
label the model trains on.

Left panel  : world-frame BEV of the full trajectory, with current step k and
              goal step k_goal marked and the supervised segment highlighted.
Right panel : the local-frame action label (origin = robot's current pose) with
              heading arrows. For the seen case you can see the waypoints fall
              *behind* the robot (the "turn-around" lives in geometry, not in an
              explicit rotation command).

Usage:
    python viz_memnav_trajectory.py --parquet /path/episode_000000.parquet --mode seen
    python viz_memnav_trajectory.py --parquet ... --k 40 --k_goal 5     # force steps
    python viz_memnav_trajectory.py --parquet ... --mode unseen --out cmp.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# repo root so `internnav` imports resolve
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset


def add_dir_arrows(ax, pts, color, n=5, lw=1.8):
    """Draw n arrowheads along polyline `pts` (in travel order) to show direction."""
    if len(pts) < 2:
        return
    samp = pts[np.linspace(0, len(pts) - 1, n + 1).astype(int)]
    for j in range(len(samp) - 1):
        ax.annotate("", xy=samp[j + 1], xytext=samp[j],
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=16))


def make_helper(predict_size=24):
    """A MemNav_Dataset with ONLY the attrs the pose-math needs (skips __init__)."""
    ds = MemNav_Dataset.__new__(MemNav_Dataset)
    ds.predict_size = predict_size
    return ds


def build_label(ds, seg, base_extrinsic, pred_digit=4):
    """Replicates _build_actions but also returns the world-frame seg for BEV.

    seg: [L,4,4] ordered current(0) -> goal(L-1).  The label path
    (`local_label_points`) is deterministic — the random rotation augmentation
    only affects `local_augment_points`, which we discard.
    """
    L = seg.shape[0]
    local_pts, _aug, world_pts, _augw, idx = ds.process_actions(
        seg, base_extrinsic, 0, L - 1, pred_digit=pred_digit)
    init_vector = local_pts[1] - local_pts[0]
    xyt = ds.xyz_to_xyt(local_pts, init_vector)          # [L-1, 3]  full path-to-goal
    pred_xyt = xyt[idx]                                   # [predict_size+1, 3]  the actual label
    init_angle = np.arctan2(init_vector[1], init_vector[0])
    horizon = ds.predict_size * pred_digit               # steps the label can cover
    truncated = (L - 1) > horizon
    return pred_xyt, init_angle, xyt, horizon, truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--mode", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--k", type=int, default=None, help="current step (override)")
    ap.add_argument("--k_goal", type=int, default=None, help="goal step (override)")
    ap.add_argument("--pred_digit", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="memnav_traj.png")
    args = ap.parse_args()
    np.random.seed(args.seed)

    ds = make_helper()
    # process_data_parquet only needs trajectory_data_dir[index]; fake index 0.
    ds.trajectory_data_dir = [args.parquet]
    _intr, base_extrinsic, extrinsics, T = ds.process_data_parquet(0)
    print(f"trajectory length T = {T}, extrinsics shape = {extrinsics.shape}")

    # pick steps
    k, k_goal = args.k, args.k_goal
    g = 4
    if k is None or k_goal is None:
        k_min = min(8 + 8, T - 1)
        if args.mode == "seen":
            k = np.random.randint(max(k_min, g + 1), T)
            k_goal = np.random.randint(0, k - g + 1)
        else:
            k = np.random.randint(k_min, T - 1 - g + 1)
            k_goal = np.random.randint(k + g, T)
    is_seen = k_goal < k
    print(f"mode={'seen' if is_seen else 'unseen'}  k(current)={k}  k_goal={k_goal}")

    # ordered current -> goal segment (matches __getitem__)
    if is_seen:
        seg = extrinsics[k_goal:k + 1][::-1].copy()   # reversed: seg[0]==current
    else:
        seg = extrinsics[k:k_goal + 1].copy()
    print(f"seg shape = {seg.shape}  (seg[0]=current pose @ step {k})")

    pred_xyt, init_angle, full_xyt, horizon, truncated = build_label(
        ds, seg, base_extrinsic, args.pred_digit)
    if truncated:
        print(f"NOTE: seg has {seg.shape[0]-1} steps but label horizon is only "
              f"{horizon} steps -> label is TRUNCATED, does NOT reach the goal.")

    # ---------------- plot ----------------
    fig, (axw, axl) = plt.subplots(1, 2, figsize=(13, 6))

    # Left panel: full trajectory transformed into the CURRENT robot's local frame
    # using the SAME relative_pose the label uses -> shares orientation/scale/aspect
    # with the right panel, so the GT segment overlaps the label's path-to-goal line.
    Rk, Tk = extrinsics[k, 0:3, 0:3], extrinsics[k, 0:3, 3]
    _, local_all = ds.relative_pose(Rk, Tk, np.eye(3), extrinsics[:, 0:3, 3], base_extrinsic)
    lxy = local_all[:, 0:2]
    lo, hi = sorted((k_goal, k))
    seg_l = lxy[lo:hi + 1]
    axw.plot(lxy[:, 0], lxy[:, 1], "-", color="0.7", lw=1, label="full trajectory")
    axw.plot(seg_l[:, 0], seg_l[:, 1], "-", color="tab:blue", lw=2.5,
             label="supervised segment (GT)")
    # original travel direction along the path actually driven: start (0) -> current (k)
    add_dir_arrows(axw, lxy[0:k + 1], color="tab:purple", n=6)
    axw.plot([], [], color="tab:purple", lw=1.8,
             label="original travel dir (start→current)")
    axw.scatter(0, 0, c="tab:green", s=120, zorder=5, label=f"current k={k}")
    axw.scatter(*lxy[k_goal], c="tab:red", marker="*", s=260, zorder=5,
                label=f"goal k_goal={k_goal}")
    axw.scatter(*lxy[0], c="k", marker="s", s=40, zorder=4, label="traj start")
    axw.annotate("", xy=(0.5, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2))
    axw.set_title("full trajectory in current-robot local frame")
    axw.set_aspect("equal"); axw.legend(fontsize=8); axw.grid(alpha=0.3)
    axw.axhline(0, color="0.85", lw=0.5); axw.axvline(0, color="0.85", lw=0.5)

    # Local action label: origin = current robot pose
    # faint = the FULL path to the true goal; bold = the actual 24-waypoint label
    axl.plot(full_xyt[:, 0], full_xyt[:, 1], "-", color="0.75", lw=1,
             label="full path to goal")
    axl.scatter(full_xyt[-1, 0], full_xyt[-1, 1], c="tab:red", marker="*", s=260,
                zorder=5, label="true goal")
    px, py, th = pred_xyt[:, 0], pred_xyt[:, 1], pred_xyt[:, 2]
    axl.plot(px, py, "-o", color="tab:blue", ms=3, lw=1.5,
             label=f"label waypoints (≤{horizon} steps)")
    # heading arrows: absolute angle = init_angle + theta (theta is rel. to init_vector)
    head = init_angle + th
    axl.quiver(px, py, np.cos(head), np.sin(head), color="tab:orange",
               angles="xy", scale=18, width=0.005, label="heading θ")
    axl.scatter(0, 0, c="tab:green", s=120, zorder=5, label="current (origin)")
    # robot's physical forward axis (+x of body frame) for reference
    axl.annotate("", xy=(0.5, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color="tab:green", lw=2))
    axl.text(0.52, 0.0, "body +x (forward)", color="tab:green", fontsize=8)
    ttl = "SEEN / revisit — goal behind ⇒ turn-around is geometric (waypoints at −x)" \
        if is_seen else "unseen / forward — waypoints continue ahead (+x)"
    if truncated:
        ttl += "\n[label TRUNCATED at horizon — does not reach true goal]"
    axl.set_title(ttl, fontsize=9)
    axl.set_aspect("equal"); axl.legend(fontsize=8, loc="best"); axl.grid(alpha=0.3)
    axl.axhline(0, color="0.85", lw=0.5); axl.axvline(0, color="0.85", lw=0.5)

    # identical limits + aspect on both panels (full-trajectory scale, so the start
    # position is visible) -> an overlay still lines the GT path up.
    xs = np.concatenate([lxy[:, 0], full_xyt[:, 0]])
    ys = np.concatenate([lxy[:, 1], full_xyt[:, 1]])
    pad = 0.3
    xlim = (xs.min() - pad, xs.max() + pad)
    ylim = (ys.min() - pad, ys.max() + pad)
    for ax in (axw, axl):
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
