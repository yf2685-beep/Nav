"""BEV occupancy + trajectory for an InternData-N1 trajectory, for comparison with our
generated episodes. No mesh/navmesh available for MP3D, so (like iPlanner's data_generation)
we accumulate the per-frame DEPTH point cloud and build a height-classified top-down map.

Frame: parquet `action` = camera-to-world 4x4 (data Z-up: x,y ground, z height). Depth is
uint16 PNG / depth_scale -> metres. We unproject in OpenCV optical (x right, y down, z fwd),
convert to the action camera frame (y up, z back -> flip y,z), transform to world, and bin.
"""
import argparse, os, glob
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="trajectory dir (contains data/, videos/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth_scale", type=float, default=10000.0)
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--frame_stride", type=int, default=3)
    ap.add_argument("--px_stride", type=int, default=5)
    ap.add_argument("--obs_lo", type=float, default=0.30); ap.add_argument("--obs_hi", type=float, default=1.8)
    ap.add_argument("--dmax", type=float, default=6.0)
    ap.add_argument("--min_obs_pts", type=int, default=3, help="min pts in cell to call it obstacle")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(args.traj, "data/chunk-000/episode_000000.parquet"))
    P = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in df["action"]])       # cam->world
    K = np.array(df["observation.camera_intrinsic"].iloc[0].tolist(), float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xy = P[:, :2, 3]; fwd = P[:, :2, 0]; yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    floor_z = float(np.median(P[:, 2, 3]))              # camera height ~ constant

    rgb_dir = os.path.join(args.traj, "videos/chunk-000/observation.images.depth")
    dpaths = sorted(glob.glob(rgb_dir + "/*.png"), key=lambda p: int(os.path.basename(p)[:-4]))
    pts = []
    for i in range(0, min(len(dpaths), len(P)), args.frame_stride):
        d = np.array(Image.open(dpaths[i]), np.float32) / args.depth_scale
        Hh, Ww = d.shape
        us = np.arange(0, Ww, args.px_stride); vs = np.arange(0, Hh, args.px_stride)
        uu, vv = np.meshgrid(us, vs)
        dd = d[vv, uu]
        m = (dd > 0.1) & (dd < args.dmax)
        if m.sum() == 0:
            continue
        uu, vv, dd = uu[m], vv[m], dd[m]
        # OpenCV optical -> action camera frame (flip y,z) -> world
        xc = (uu - cx) / fx * dd; yc = (vv - cy) / fy * dd; zc = dd
        cam = np.stack([xc, -yc, -zc], axis=1)          # y up, z back
        R = P[i, :3, :3]; t = P[i, :3, 3]
        pts.append(cam @ R.T + t)
    pts = np.concatenate(pts, axis=0)

    # occupancy over the trajectory bounds + margin
    x0, x1 = xy[:, 0].min() - 1.0, xy[:, 0].max() + 1.0
    y0, y1 = xy[:, 1].min() - 1.0, xy[:, 1].max() + 1.0
    nx = int((x1 - x0) / args.res) + 1; ny = int((y1 - y0) / args.res) + 1
    seen_cnt = np.zeros((ny, nx), int); obs_cnt = np.zeros((ny, nx), int)
    ix = ((pts[:, 0] - x0) / args.res).astype(int); iy = ((pts[:, 1] - y0) / args.res).astype(int)
    good = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(seen_cnt, (iy[good], ix[good]), 1)
    # obstacle = height ABOVE THE OBSERVED FLOOR in [obs_lo, obs_hi] (scene-independent: the
    # floor's absolute z differs per MP3D scene, so anchor to a low percentile of the cloud).
    floor_world = float(np.percentile(pts[:, 2], 3))
    band = good & (pts[:, 2] > floor_world + args.obs_lo) & (pts[:, 2] < floor_world + args.obs_hi)
    np.add.at(obs_cnt, (iy[band], ix[band]), 1)
    seen = seen_cnt >= 1; obs = obs_cnt >= args.min_obs_pts

    # render: white free (seen & !obs), dark obstacle, light-grey unseen
    img = np.full((ny, nx), 0.65)                        # unseen grey
    img[seen] = 1.0                                       # seen free -> white
    img[obs] = 0.15                                       # obstacle -> dark
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(img, origin="lower", extent=[x0, x1, y0, y1], cmap="gray", vmin=0, vmax=1,
              interpolation="nearest", zorder=0)
    ax.plot(xy[:, 0], xy[:, 1], "-", color="royalblue", lw=1.8, label="N1 trajectory", zorder=3)
    for i in range(0, len(xy), 12):
        ax.arrow(xy[i, 0], xy[i, 1], 0.13 * np.cos(yaw[i]), 0.13 * np.sin(yaw[i]),
                 head_width=0.05, color="k", alpha=0.6, zorder=4)
    ax.scatter(*xy[0], c="lime", s=100, ec="k", zorder=6, label="start")
    ax.scatter(*xy[-1], c="k", s=140, marker="*", zorder=6, label="goal")
    ax.legend(loc="upper right", fontsize=9); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"InternData-N1 {os.path.basename(args.traj)}  n={len(xy)}  "
                 f"(occupancy from accumulated depth: white=seen-free dark=obstacle grey=unseen)")
    plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print(f"saved {args.out} | pts={len(pts)} occ {img.shape} obstacle_cells={obs.sum()} seen={seen.sum()}")


if __name__ == "__main__":
    main()
