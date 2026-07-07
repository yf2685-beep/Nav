"""Reverse-engineer InternData-N1 controller params: min turning radius + effective lookahead.

turning radius:  R_i = ds_i / |dtheta_i|   (arc length per unit heading change)
                 min turning radius = tight percentile of R over turning frames.
lookahead (eff): at frame i, march forward along the path; the lookahead ~ the arc distance
                 at which the chord from p_i first makes the observed steer angle with heading,
                 i.e. how far ahead the heading 'aims'. Reported as a distribution (effective,
                 since N1 may not be literal pure-pursuit).
"""
import numpy as np, pandas as pd, glob, os

base = "/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i"
pqs = sorted(glob.glob(f"{base}/*/*/data/chunk-000/episode_000000.parquet"))

def smooth(a, k=5):
    ker = np.ones(k) / k
    return np.convolve(np.concatenate([[a[0]]*(k//2), a, [a[-1]]*(k//2)]), ker, "valid")

R_all, Rsm_all, look_all, speed_all, curv_all = [], [], [], [], []
for pq in pqs:
    df = pd.read_parquet(pq)
    A = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in df["action"]])
    xy = A[:, :2, 3]
    if len(xy) < 12:
        continue
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    yaw = np.unwrap(np.arctan2(A[:, 1, 0], A[:, 0, 0]))
    dth = np.abs(np.diff(yaw))
    speed_all.append(ds)
    # curvature radius (raw), only on turning + moving frames
    mv = (ds > 0.005) & (dth > np.deg2rad(0.3))
    R = ds[mv] / dth[mv]
    R_all.append(R); curv_all.append(dth[mv] / ds[mv])
    # smoothed heading radius (removes per-frame quantization noise) -> true sustained tightness
    yss = smooth(yaw, 5); dth_s = np.abs(np.diff(yss)); dss = smooth(ds, 5)
    mv2 = (dss > 0.01) & (dth_s > np.deg2rad(0.2))
    Rsm_all.append((dss[mv2] / dth_s[mv2]))
    # effective lookahead: for each frame, arc distance ahead until the path bends by ~15 deg
    arc = np.concatenate([[0], np.cumsum(ds)])
    for i in range(0, len(yaw) - 1, 3):
        j = i + 1
        while j < len(yaw) and abs(yaw[j] - yaw[i]) < np.deg2rad(15):
            j += 1
        if j < len(yaw):
            look_all.append(arc[min(j, len(arc)-1)] - arc[i])

R = np.concatenate(R_all); Rsm = np.concatenate(Rsm_all); look = np.array(look_all)
sp = np.concatenate(speed_all); curv = np.concatenate(curv_all)
print(f"trajectories used: {len(R_all)}")
print(f"\nSPEED  m/frame: median={np.median(sp):.4f}  (=> {np.median(sp)*30:.2f} m/s @30fps)")
print(f"\nTURNING RADIUS R=ds/dθ (per-frame, raw):")
print(f"   p1={np.percentile(R,1):.2f}  p5={np.percentile(R,5):.2f}  p50={np.percentile(R,50):.2f}  p95={np.percentile(R,95):.2f} m")
print(f"TURNING RADIUS (5-frame smoothed heading -> sustained):")
print(f"   p1={np.percentile(Rsm,1):.2f}  p5={np.percentile(Rsm,5):.2f}  p50={np.percentile(Rsm,50):.2f} m")
print(f"   => MIN TURNING RADIUS ~ {np.percentile(Rsm,1):.2f}-{np.percentile(Rsm,5):.2f} m (tightest sustained turns)")
print(f"\nEFFECTIVE LOOKAHEAD (arc dist to a 15deg bend):")
print(f"   p25={np.percentile(look,25):.2f}  median={np.median(look):.2f}  p75={np.percentile(look,75):.2f} m")
print(f"\nmax per-frame turn: {np.degrees(np.percentile(np.concatenate([np.abs(np.diff(np.unwrap(np.arctan2(np.stack([np.array(a.tolist(),float).reshape(4,4) for a in pd.read_parquet(pq)['action']])[:,1,0], np.stack([np.array(a.tolist(),float).reshape(4,4) for a in pd.read_parquet(pq)['action']])[:,0,0])))) for pq in pqs[:1]]),99)):.2f} deg (ep0 p99)")
