"""BEV occupancy + RAW geodesic (before smoothing) vs SMOOTHED pursuit trajectory (after),
with collision / wall-clearance diagnostics.

- raw geodesic = replanned start->A->B from meta (navmesh shortest path; collision-free by
  construction on the navmesh).
- smoothed = the stored parquet poses (pursuit controller output).
We check each smoothed point AND each segment midpoint against the navmesh, and report
clearance to the navmesh boundary (real wall is ~agent_radius beyond that).

Run (habitat env):
  python viz_before_after.py --scene .../apartment_1.glb --navmesh .../apartment_1.navmesh \
     --episode .../episode_0001 --out .../ep1_beforeafter.png
"""
import argparse, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

MW = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)   # data = MW @ habitat


def hab_to_data(p):
    return MW @ np.asarray(p, float)


def geodesic(pf, a, b):
    import habitat_sim
    sp = habitat_sim.ShortestPath(); sp.requested_start = a; sp.requested_end = b
    ok = pf.find_path(sp)
    return [np.array(x, float) for x in sp.points] if ok else [np.array(a), np.array(b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--navmesh", default="")
    ap.add_argument("--episode", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--res", type=float, default=0.05); ap.add_argument("--margin", type=float, default=0.8)
    ap.add_argument("--agent_radius", type=float, default=0.30, help="match generator inflation")
    args = ap.parse_args()
    import pandas as pd, habitat_sim, magnum as mn

    m = json.load(open(os.path.join(args.episode, "meta/gen_meta.json")))
    sw = m["switch_idx"]
    A = np.stack([np.array(a.tolist(), float).reshape(4, 4)
                  for a in pd.read_parquet(os.path.join(args.episode, "data/chunk-000/episode_000000.parquet"))["action"]])
    xy = A[:, :2, 3]; hab_y = float(np.median(A[:, 2, 3]))

    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = args.scene; bk.enable_physics = False
    s = habitat_sim.CameraSensorSpec(); s.uuid = "c"; s.sensor_type = habitat_sim.SensorType.COLOR
    s.resolution = [64, 64]; s.position = mn.Vector3(0, 0, 0)
    ac = habitat_sim.agent.AgentConfiguration(); ac.sensor_specifications = [s]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ac])); pf = sim.pathfinder
    if args.navmesh:
        pf.load_nav_mesh(args.navmesh)
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults(); ns.agent_radius = args.agent_radius
    ns.agent_height = 1.5; sim.recompute_navmesh(pf, ns)   # match generator's inflated navmesh

    # raw geodesics (before smoothing), from meta start/A/B (habitat coords)
    gA = np.array([hab_to_data(p)[:2] for p in geodesic(pf, m["start"], m["A"])])
    gB = np.array([hab_to_data(p)[:2] for p in geodesic(pf, m["A"], m["B"])])

    def nav_and_clear(px, py):
        q = pf.snap_point([px, hab_y, -py])
        on = pf.is_navigable(q) and np.hypot(q[0] - px, q[2] + py) < 0.06
        clr = pf.distance_to_closest_obstacle(q) if on else -1.0
        return on, clr

    # collision + clearance diagnostics on the smoothed trajectory
    pt_off, seg_off, clears = [], [], []
    for i in range(len(xy)):
        on, clr = nav_and_clear(*xy[i])
        if not on:
            pt_off.append(i)
        else:
            clears.append(clr)
        if i > 0:
            mid = (xy[i] + xy[i - 1]) / 2
            if not nav_and_clear(*mid)[0]:
                seg_off.append(i)
    clears = np.array(clears)

    # occupancy grid (data frame)
    x0, x1 = xy[:, 0].min() - args.margin, xy[:, 0].max() + args.margin
    y0, y1 = xy[:, 1].min() - args.margin, xy[:, 1].max() + args.margin
    xs = np.arange(x0, x1, args.res); ys = np.arange(y0, y1, args.res)
    occ = np.zeros((len(ys), len(xs)), np.uint8)
    for iy, gy in enumerate(ys):
        for ix, gx in enumerate(xs):
            occ[iy, ix] = 1 if nav_and_clear(gx, gy)[0] else 0
    sim.close()

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.imshow(occ, origin="lower", extent=[x0, x1, y0, y1], cmap="gray",
              vmin=-0.5, vmax=1.0, interpolation="nearest", zorder=0)
    ax.plot(gA[:, 0], gA[:, 1], "--", color="deepskyblue", lw=2, label="raw geodesic A (before)", zorder=2)
    ax.plot(gB[:, 0], gB[:, 1], "--", color="orange", lw=2, label="raw geodesic B (before)", zorder=2)
    ax.scatter(gA[:, 0], gA[:, 1], c="deepskyblue", s=18, zorder=2)
    ax.scatter(gB[:, 0], gB[:, 1], c="orange", s=18, zorder=2)
    ax.plot(xy[:sw, 0], xy[:sw, 1], "-", color="royalblue", lw=1.5, label="smoothed A (after)", zorder=3)
    ax.plot(xy[sw:, 0], xy[sw:, 1], "-", color="red", lw=1.5, label="smoothed B (after)", zorder=3)
    if pt_off:
        ax.scatter(xy[pt_off, 0], xy[pt_off, 1], marker="x", c="red", s=90, lw=2.5,
                   label=f"OFF-navmesh pts ({len(pt_off)})", zorder=7)
    if seg_off:
        mids = np.array([(xy[i] + xy[i - 1]) / 2 for i in seg_off])
        ax.scatter(mids[:, 0], mids[:, 1], marker="+", c="magenta", s=110, lw=2.5,
                   label=f"seg crosses wall ({len(seg_off)})", zorder=7)
    ax.scatter(*xy[sw], c="magenta", s=120, ec="k", zorder=6, label="A / U-turn")
    ax.scatter(*xy[0], c="lime", s=100, ec="k", zorder=6); ax.scatter(*xy[-1], c="k", s=140, marker="*", zorder=6)
    ax.legend(loc="upper right", fontsize=8); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"{os.path.basename(args.episode)}  before(dashed) vs after(solid) smoothing")
    plt.tight_layout(); plt.savefig(args.out, dpi=130)
    print(f"saved {args.out}")
    print(f"smoothed pts off-navmesh: {len(pt_off)}/{len(xy)} ; segments crossing wall: {len(seg_off)}")
    print(f"navmesh clearance (dist to navmesh edge): min={clears.min():.3f} p5={np.percentile(clears,5):.3f} "
          f"median={np.median(clears):.3f} m ; pts within 5cm of edge: {(clears<0.05).sum()}")
    print(f"(real wall is ~agent_radius beyond the navmesh edge)")


if __name__ == "__main__":
    main()
