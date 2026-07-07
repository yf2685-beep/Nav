"""Characterize InternData-N1 trajectory motion: step displacement, heading, angular rate,
turn-in-place presence, curvature — to decide how to smooth our generated paths."""
import numpy as np, pandas as pd, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

base = "/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i"
pqs = sorted(glob.glob(f"{base}/*/*/data/chunk-000/episode_000000.parquet"))[:6]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
allstep, allturn = [], []
for ax, pq in zip(axes.ravel(), pqs):
    df = pd.read_parquet(pq)
    A = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in df["action"]])
    pos = A[:, :3, 3]                       # world xyz (z const per traj)
    xy = pos[:, :2]
    # forward axis of camera pose = which column? try col0 projected to ground
    fwd = A[:, :2, 0]                        # col0 xy
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dyaw = np.abs((np.diff(yaw) + np.pi) % (2 * np.pi) - np.pi)
    allstep.append(step); allturn.append(np.degrees(dyaw))
    # turn-in-place = big heading change with tiny displacement
    tip = ((step < 0.01) & (np.degrees(dyaw) > 2)).sum()
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=1)
    ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), s=8, cmap="viridis")
    ax.scatter(*xy[0], c="g", s=80, marker="o"); ax.scatter(*xy[-1], c="r", s=80, marker="*")
    # draw heading arrows every 10 frames
    for i in range(0, len(xy), 10):
        ax.arrow(xy[i, 0], xy[i, 1], 0.15*np.cos(yaw[i]), 0.15*np.sin(yaw[i]),
                 head_width=0.05, color="k", alpha=0.5)
    ax.set_title(f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pq))))[:10]} "
                 f"n={len(xy)} step med={np.median(step):.3f}m turn-in-place={tip}")
    ax.axis("equal")
plt.tight_layout(); plt.savefig("/home/asus/Research/Nav/memnav_viz/n1_motion.png", dpi=100)

s = np.concatenate(allstep); t = np.concatenate(allturn)
print(f"STEP displacement (m/frame): median={np.median(s):.3f} mean={s.mean():.3f} "
      f"p90={np.percentile(s,90):.3f} max={s.max():.3f} frac<1cm={100*(s<0.01).mean():.1f}%")
print(f"TURN rate (deg/frame): median={np.median(t):.2f} mean={t.mean():.2f} "
      f"p90={np.percentile(t,90):.2f} max={t.max():.2f}")
print(f"turn-in-place frames (step<1cm & turn>2deg): {((s<0.01)&(t>2)).sum()} / {len(s)}")
print("saved /home/asus/Research/Nav/memnav_viz/n1_motion.png")
