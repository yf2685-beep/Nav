"""Overlay the navmesh free-space (BEV occupancy) behind a generated two-leg trajectory.

The generated parquet stores poses in the Z-up "data" frame (data = M_W @ habitat,
M_W = [[1,0,0],[0,0,-1],[0,1,0]]), so the plot's (x,y) = (hab_x, -hab_z). We sample
navigability on a grid IN THAT SAME (x,y) frame (convert each cell back to habitat via
M_W^-1 and query the pathfinder), so the occupancy aligns exactly with the trajectory.

Run (habitat env):
  python viz_bev.py --scene .../apartment_1.glb --navmesh .../apartment_1.navmesh \
     --episode /home/asus/Research/Nav/memnav_viz/twoleg_pp/episode_0001 \
     --out /home/asus/Research/Nav/memnav_viz/ep1_bev.png
"""
import argparse, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--navmesh", default="")
    ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.05, help="occupancy grid res (m)")
    ap.add_argument("--margin", type=float, default=0.8)
    ap.add_argument("--arrow_every", type=int, default=12)
    ap.add_argument("--agent_radius", type=float, default=0.30, help="match generator inflation")
    args = ap.parse_args()

    import pandas as pd, habitat_sim, magnum as mn
    mpath = os.path.join(args.episode, "meta/gen_meta.json")
    m = json.load(open(mpath)) if os.path.isfile(mpath) else {}   # InternData-N1 has no gen_meta
    A = np.stack([np.array(a.tolist(), float).reshape(4, 4)
                  for a in pd.read_parquet(os.path.join(args.episode, "data/chunk-000/episode_000000.parquet"))["action"]])
    sw = m.get("switch_idx", len(A))                # no switch -> whole thing is one leg
    xy = A[:, :2, 3]                                    # data-frame ground plane (x, y=-hab_z)
    yaw = np.unwrap(np.arctan2(A[:, 1, 0], A[:, 0, 0]))
    hab_y = float(np.median(A[:, 2, 3]))               # data_z == hab_y (camera height)

    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = args.scene; bk.enable_physics = False
    s = habitat_sim.CameraSensorSpec(); s.uuid = "c"; s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = [64, 64]; s.position = mn.Vector3(0, 0, 0)
    ac = habitat_sim.agent.AgentConfiguration(); ac.sensor_specifications = [s]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ac]))
    pf = sim.pathfinder
    if args.navmesh:
        pf.load_nav_mesh(args.navmesh)
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = args.agent_radius
    ns.agent_height = 1.5; sim.recompute_navmesh(pf, ns)   # match generator's inflated navmesh

    x0, x1 = xy[:, 0].min() - args.margin, xy[:, 0].max() + args.margin
    y0, y1 = xy[:, 1].min() - args.margin, xy[:, 1].max() + args.margin
    xs = np.arange(x0, x1, args.res); ys = np.arange(y0, y1, args.res)
    occ = np.zeros((len(ys), len(xs)), np.uint8)       # 1 = free/navigable
    for iy, gy in enumerate(ys):
        for ix, gx in enumerate(xs):
            # data (gx, gy) -> habitat: hab = M_W^-1 @ [gx, gy, hab_y]; M_W^-1 = [[1,0,0],[0,0,1],[0,-1,0]]
            q = pf.snap_point([gx, hab_y, -gy])
            if pf.is_navigable(q) and np.hypot(q[0] - gx, q[2] - (-gy)) < args.res:
                occ[iy, ix] = 1
    sim.close()

    fig, ax = plt.subplots(figsize=(13, 11))
    # free space white, obstacle/void dark grey (convention)
    ax.imshow(occ, origin="lower", extent=[x0, x1, y0, y1], cmap="gray",
              vmin=-0.5, vmax=1.0, interpolation="nearest", zorder=0)
    if sw < len(xy):                                    # two-leg (our generated) episode
        ax.plot(xy[:sw, 0], xy[:sw, 1], "-", color="royalblue", lw=1.6, label="leg A", zorder=3)
        ax.plot(xy[sw:, 0], xy[sw:, 1], "-", color="red", lw=1.6, label="leg B (return)", zorder=3)
        ax.scatter(*xy[sw], c="magenta", s=130, ec="k", zorder=6, label="A / switch (U-turn)")
    else:                                               # single-leg (InternData-N1) trajectory
        ax.plot(xy[:, 0], xy[:, 1], "-", color="royalblue", lw=1.6, label="trajectory", zorder=3)
    ax.scatter(*xy[0], c="lime", s=100, ec="k", zorder=6, label="start")
    ax.scatter(*xy[-1], c="k", s=140, marker="*", zorder=6, label="goal")
    for i in range(0, len(xy), args.arrow_every):
        ax.arrow(xy[i, 0], xy[i, 1], 0.13 * np.cos(yaw[i]), 0.13 * np.sin(yaw[i]),
                 head_width=0.05, color="k", alpha=0.6, zorder=4)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    div = m.get("legB_vs_revlegA_mean_div_m")
    ax.set_title(f"{os.path.basename(args.episode)}  free-space (light) vs walls (dark)  "
                 f"n={len(xy)} switch={sw}" + (f" div={div:.2f}" if div is not None else " (InternData-N1)"))
    plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print("saved", args.out, "| occ grid", occ.shape, "| free frac", round(occ.mean(), 2))


if __name__ == "__main__":
    main()
