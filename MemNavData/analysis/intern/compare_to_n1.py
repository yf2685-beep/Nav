import numpy as np, pandas as pd, json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def load(pq):
    A = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in pd.read_parquet(pq)["action"]])
    xy = A[:, :2, 3]; yaw = np.unwrap(np.arctan2(A[:, 1, 0], A[:, 0, 0]))
    return xy, yaw

def stats(pqs):
    R, sp, tp = [], [], []
    for pq in pqs:
        xy, yaw = load(pq)
        if len(xy) < 8: continue
        ds = np.linalg.norm(np.diff(xy, axis=0), axis=1); dth = np.abs(np.diff(yaw))
        sp.append(ds); tp.append(np.degrees(dth))
        mv = (ds > 0.005) & (dth > np.deg2rad(0.3)); R.append(ds[mv] / dth[mv])
    return np.concatenate(R), np.concatenate(sp), np.concatenate(tp)

n1 = sorted(glob.glob("/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i/*/*/data/chunk-000/episode_000000.parquet"))
ours = sorted(glob.glob("/home/asus/Research/Nav/memnav_viz/twoleg_pp/episode_*/data/chunk-000/episode_000000.parquet"))
Rn, spn, tpn = stats(n1); Ro, spo, tpo = stats(ours)

fig = plt.figure(figsize=(20, 9))
# top row: 3 example paths from ours with U-turn zoom
for c, pq in enumerate(ours[:3]):
    ep = os.path.dirname(os.path.dirname(os.path.dirname(pq)))
    m = json.load(open(ep + "/meta/gen_meta.json")); sw = m["switch_idx"]
    xy, yaw = load(pq)
    ax = fig.add_subplot(2, 4, c + 1)
    ax.plot(xy[:sw, 0], xy[:sw, 1], "b-", lw=1.3, label="leg A")
    ax.plot(xy[sw:, 0], xy[sw:, 1], "r-", lw=1.3, label="leg B (return)")
    for i in range(0, len(xy), 12):
        ax.arrow(xy[i,0], xy[i,1], .1*np.cos(yaw[i]), .1*np.sin(yaw[i]), head_width=.04, color="k", alpha=.35)
    ax.scatter(*xy[sw], c="m", s=70, zorder=5, label="A (U-turn)")
    ax.set_title(f"ep{c} div={m['legB_vs_revlegA_mean_div_m']:.2f}"); ax.axis("equal"); ax.legend(fontsize=6)
# step+yaw vs frame for ep0 (check U-turn coupling)
xy0, yaw0 = load(ours[0]); m0 = json.load(open(os.path.dirname(os.path.dirname(os.path.dirname(ours[0]))) + "/meta/gen_meta.json"))
sw0 = m0["switch_idx"]; step0 = np.concatenate([[0], np.linalg.norm(np.diff(xy0, axis=0), axis=1)])
ax = fig.add_subplot(2, 4, 4)
ax.plot(step0, label="step m/frame"); ax.axvline(sw0, color="m", ls=":", label="U-turn/switch")
ax.axhline(0.0376, color="g", ls="--", alpha=.5, label="N1 speed"); ax.set_ylim(0, .05)
ax.set_title("ep0 step vs frame (U-turn = slow, not 0)"); ax.legend(fontsize=6)
# distributions
ax = fig.add_subplot(2, 4, 5)
bins = np.linspace(0, 6, 60)
ax.hist(Rn, bins, density=True, alpha=.5, label="N1"); ax.hist(Ro, bins, density=True, alpha=.5, label="ours")
ax.axvline(0.4, color="k", ls=":", label="r_min 0.4"); ax.set_title("turning radius R=ds/dθ (m)"); ax.legend(fontsize=7); ax.set_xlim(0, 6)
ax = fig.add_subplot(2, 4, 6)
ax.hist(spn, np.linspace(0, .06, 40), density=True, alpha=.5, label="N1")
ax.hist(spo, np.linspace(0, .06, 40), density=True, alpha=.5, label="ours"); ax.set_title("step displacement (m/frame)"); ax.legend(fontsize=7)
ax = fig.add_subplot(2, 4, 7)
ax.hist(tpn, np.linspace(0, 6, 40), density=True, alpha=.5, label="N1")
ax.hist(tpo, np.linspace(0, 6, 40), density=True, alpha=.5, label="ours"); ax.set_title("turn rate (deg/frame)"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("/home/asus/Research/Nav/memnav_viz/pp_compare.png", dpi=95)
print("=== turning radius R=ds/dθ (m) ===")
for nm, R in [("N1  ", Rn), ("OURS", Ro)]:
    print(f"  {nm}: p5={np.percentile(R,5):.2f} p25={np.percentile(R,25):.2f} p50={np.percentile(R,50):.2f} p75={np.percentile(R,75):.2f}")
print(f"=== step m/frame: N1 med={np.median(spn):.4f}  OURS med={np.median(spo):.4f}")
print(f"=== turn deg/frame: N1 med={np.median(tpn):.2f} max={tpn.max():.1f}  OURS med={np.median(tpo):.2f} max={tpo.max():.1f}")
print(f"=== ep0 U-turn: min step in [{sw0-5},{sw0+30}] = {step0[sw0-5:sw0+30].min():.4f} m (0 would be a frozen spin)")
print("saved /home/asus/Research/Nav/memnav_viz/pp_compare.png")
