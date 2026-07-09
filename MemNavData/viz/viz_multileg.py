"""Multi-leg BEV: colour each leg, mark start/A/goals + goal-image orientation arrow +
the covisibility-argmax history frame each revisit goal matched. Handles 2- and 3-leg."""
import argparse, json, os, numpy as np, pandas as pd, habitat_sim
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True); ap.add_argument("--episode", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--agent_radius", type=float, default=0.30)
args = ap.parse_args()

m = json.load(open(os.path.join(args.episode, "meta/gen_meta.json")))
sw = m["switches"]; n = m["n_frames"]
A = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in
              pd.read_parquet(os.path.join(args.episode, "data/chunk-000/episode_000000.parquet"))["action"]])
xy = A[:, :2, 3]; yaw = np.unwrap(np.arctan2(-A[:, 1, 2], -A[:, 0, 2])); haby = float(np.median(A[:, 2, 3]))
# heading = camera forward (-Z col) projected to the data ground plane, NOT the x-axis (1st col)

bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = args.scene; bk.enable_physics = False
sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [habitat_sim.agent.AgentConfiguration()]))
pf = sim.pathfinder
ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = args.agent_radius; sim.recompute_navmesh(pf, ns)
x0, x1 = xy[:, 0].min()-1, xy[:, 0].max()+1; y0, y1 = xy[:, 1].min()-1, xy[:, 1].max()+1; res = 0.06
xs = np.arange(x0, x1, res); ys = np.arange(y0, y1, res); occ = np.zeros((len(ys), len(xs)))
# haby is the CAMERA height; snap_point returns the FLOOR (~cam_h below), and the floor itself varies
# along the path -> compare the snapped floor to the reference floor (haby - cam_h), not to haby, or a
# lower/sloping floor is wrongly marked non-navigable (black). cam_h=0.5 matches generation; band 0.8m.
CAM_H = 0.5
for iy, gy in enumerate(ys):
    for ix, gx in enumerate(xs):
        q = pf.snap_point([gx, haby, -gy])
        occ[iy, ix] = 1 if (pf.is_navigable(q) and abs(q[0]-gx) < res and abs(q[2]+gy) < res
                            and abs(q[1] - (haby - CAM_H)) < 0.8) else 0
sim.close()

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(occ, origin="lower", extent=[x0, x1, y0, y1], cmap="gray", vmin=-0.5, vmax=1, zorder=0)
bounds = [0]+sw+[n]; cols = ["royalblue", "red", "darkorange"]; labs = ["leg1 start->A", "leg2 ->B", "leg3 ->C"]
for k in range(len(bounds)-1):
    seg = xy[bounds[k]:bounds[k+1]]; ax.plot(seg[:, 0], seg[:, 1], "-", color=cols[k], lw=1.8, label=labs[k], zorder=3)
for i in range(0, n, 16): ax.arrow(xy[i, 0], xy[i, 1], .1*np.cos(yaw[i]), .1*np.sin(yaw[i]), head_width=.04, color="dimgray", alpha=.5, zorder=4)
ax.scatter(*xy[0], c="lime", s=120, ec="k", zorder=6, label="start")
ax.scatter(xy[sw[0]-1, 0], xy[sw[0]-1, 1], c="cyan", s=110, ec="k", zorder=6, label="A")
for g, c in zip(m["goals"], ["magenta", "gold"]):
    p = g["pos"]; ax.scatter(p[0], p[1], s=210, marker="*", ec="k", c=c, zorder=7,
                             label=f"{g['name']} [{g['kind']}] covis={g['covis']:.2f}")
    gy = g["yaw_habitat"]  # habitat yaw -> data-frame forward = (-sin, cos)
    ax.arrow(p[0], p[1], .35*-np.sin(gy), .35*np.cos(gy), head_width=.11, color="k", lw=2, zorder=8)
    ai = g.get("covis_argmax", -1)
    if ai is not None and ai >= 0:
        ax.scatter(xy[ai, 0], xy[ai, 1], s=90, marker="D", ec="k", c=c, zorder=7)   # matched history frame
ax.legend(fontsize=9, loc="upper right"); ax.set_aspect("equal")
ax.set_title(f"{os.path.basename(args.episode)}  n_legs={m['n_legs']} n={n} switches={sw}  "
             f"(black arrow=goal orientation, diamond=matched history frame)")
plt.tight_layout(); plt.savefig(args.out, dpi=115); print("saved", args.out)
