"""Stage 1/3 of the revisit-definition sweep: generate synthetic (leg-A, goal-B) pairs.

Question this experiment answers: when a goal image B is inserted into the LingBot
stream after its best-match frame X on leg A (the memnav revisit path), over what
envelope of (relative pose X->B, covisibility B<->X) does LingBot still relocalize B
correctly? The envelope becomes the operational definition of "revisit".

This stage (habitat env, no GPU torch needed):
  1. roll a leg-A trajectory per episode (same geodesic + ElasticBands + pure-pursuit
     stack as generate_twoleg.py, so frames match the training distribution).
  2. pick `--anchors` anchor frames X in the valid match range
     [num_scale + window - 1 .. n-2] (the memnav_policy clamp range).
  3. for each X, place goal cameras B on a grid RELATIVE TO X's pose:
     forward x lateral x heading-offset (defaults below), snapped to the navmesh.
  4. render B (RGB + depth) and compute GROUND-TRUTH covisibility by occlusion-checked
     reprojection of B's depth pixels: covis(B->F) = fraction of B's surface points
     co-observed by frame F (|z_reproj - depth_F| <= tol). Recorded vs the anchor,
     vs the reverse direction, and vs every `--covis_stride`-th leg frame (max/argmax).
  5. sample `--n_neg` NEGATIVE controls: navigable viewpoints whose max covisibility
     with the whole leg is < `--neg_covis` (never-seen content) — these measure the
     false-positive behaviour of any revisit gate downstream.

Output layout (precompute_lingbot_features.py-compatible, so stage 1.5 runs unchanged):
  <out>/<group>/<scene>/trajectory_XXX/
      videos/chunk-000/observation.images.{rgb,depth}/   leg-A frames
      data/chunk-000/episode_000000.parquet              leg-A poses (Z-up data frame)
      meta/gen_meta.json
      sweep/goal_SSSS.jpg                                goal images (grid + negatives)
      sweep/sweep_meta.json                              per-sample GT records (below)

Per-sample record: kind grid|neg, anchor_idx, requested (fwd, lat, dyaw_deg), actual
post-snap (fwd_m, lat_m, dyaw_deg), goal cam-to-world (habitat 4x4), covisibilities,
goal file. Trajectory record adds every leg frame's cam-to-world (habitat 4x4) so
stage 2 never re-derives frames from the parquet.

Run (habitat env):
  python revisit_sweep_gen.py \
      --scene /home/asus/Research/datasets/mp3d/17DRP5sb8fy.glb \
      --navmesh /home/asus/Research/datasets/mp3d/17DRP5sb8fy.navmesh \
      --out /home/asus/Research/datasets/memnav_sweep --n 4 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generate_twoleg as g2  # noqa: E402  (lives in MemNavData/, one level up)
from generate_twoleg import (CX, CY, FX, FY, H, W, K, M_W,  # noqa: E402
                             build_esdf, cam_to_world_hab, densify, geodesic,
                             render, roll_leg, save_traj, yaw_facing)

DEF_FWD = [0.0, 0.5, 1.0, 2.0, 4.0, 6.0]
DEF_LAT = [0.0, -0.75, 0.75]
DEF_DYAW = [0, 30, -30, 60, -60, 90, -90, 180]


# --------------------------------------------------------------------------- #
# ground-truth covisibility (vectorized reprojection with occlusion check)
# --------------------------------------------------------------------------- #
def backproject(depth, stride=6, d_min=0.15, d_max=10.0):
    """depth [H,W] -> surface points in the CAMERA frame (habitat optical: -Z fwd,
    +Y up), subsampled on a `stride` pixel grid. [Np, 3]."""
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    d = depth[vs, us].astype(float)
    m = (d > d_min) & (d < d_max)
    u, v, d = us[m].astype(float), vs[m].astype(float), d[m]
    x = (u - CX) / FX * d
    y = (v - CY) / FY * d          # OpenCV: +Y down, +Z fwd
    return np.stack([x, -y, -d], axis=1)   # -> habitat optical


def to_world(p_cam, T_wc):
    return p_cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def covis_frac(p_world, T_wc, depth, tol=0.3):
    """Fraction of surface points `p_world` co-observed by the camera (T_wc, depth):
    inside the frustum AND the rendered depth agrees with the reprojected range
    (|z - depth[v,u]| <= tol), i.e. the camera sees THE SAME surface, not past/into it."""
    if len(p_world) == 0:
        return 0.0
    pc = (p_world - T_wc[:3, 3]) @ T_wc[:3, :3]      # R^T (p - t), habitat optical
    x, y, z = pc[:, 0], -pc[:, 1], -pc[:, 2]         # -> OpenCV
    m = z > 0.05
    zs = np.maximum(z, 1e-6)
    u = FX * x / zs + CX
    v = FY * y / zs + CY
    m &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.clip(u.astype(int), 0, W - 1)
    vi = np.clip(v.astype(int), 0, H - 1)
    d = depth[vi, ui]
    good = m & (np.abs(z - d) <= tol)
    return float(good.sum()) / len(p_world)


def covis_curve(p_world, poses, depths, idxs, tol=0.3):
    return {int(i): covis_frac(p_world, poses[i], depths[i], tol) for i in idxs}


# --------------------------------------------------------------------------- #
# goal placement
# --------------------------------------------------------------------------- #
def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def yaw_dirs(yaw):
    """(forward, left) unit vectors in the habitat ground plane for heading `yaw`
    (rotation about +Y; camera forward = -Z)."""
    fwd = np.array([-np.sin(yaw), 0.0, -np.cos(yaw)])
    left = np.array([-np.cos(yaw), 0.0, np.sin(yaw)])
    return fwd, left


def place_goal(pf, anchor_pos, anchor_yaw, fwd, lat, dyaw_deg, cam_h, max_snap=0.5):
    """Grid-sample a goal camera relative to the anchor. Returns
    (cam_pos, yaw, actual fwd, actual lat) or None if not navigable / snapped too far."""
    f, l = yaw_dirs(anchor_yaw)
    floor = np.array([anchor_pos[0], anchor_pos[1] - cam_h, anchor_pos[2]])
    target = floor + fwd * f + lat * l
    q = np.array(pf.snap_point(target), float)
    if not pf.is_navigable(q):
        return None
    if np.hypot(q[0] - target[0], q[2] - target[2]) > max_snap:
        return None
    if abs(q[1] - floor[1]) > 0.5:          # snapped onto another storey
        return None
    cam = q + np.array([0.0, cam_h, 0.0])
    delta = cam - anchor_pos
    return cam, anchor_yaw + np.deg2rad(dyaw_deg), float(delta @ f), float(delta @ l)


def actual_rel(anchor_pos, anchor_yaw, cam, yaw):
    f, l = yaw_dirs(anchor_yaw)
    d = cam - anchor_pos
    return float(d @ f), float(d @ l), float(wrap_deg(np.degrees(yaw - anchor_yaw)))


# --------------------------------------------------------------------------- #
# one trajectory = leg A + goal sweep
# --------------------------------------------------------------------------- #
def make_leg(sim, rng, args, esdf_cache):
    """Roll one leg-A trajectory (start->A geodesic), confined to A's floor (MP3D is multi-floor).
    Returns frames [(pos,yaw)] or None."""
    pf = sim.pathfinder
    ftol = getattr(args, "floor_tol", 0.8)
    eb = dict(iters=60, kc=0.5, kr=0.8, rho0=0.6, step=0.04, res=args.esdf_res)
    for _ in range(40):
        A = pf.get_random_navigable_point()
        if pf.distance_to_closest_obstacle(A) < 0.4:
            continue
        floor_y = float(A[1])                                   # A defines the floor
        E = g2._get_esdf(esdf_cache, pf, floor_y, args)         # per-floor ESDF (cached)
        cp = dict(v_max=0.0376, L=0.7, r_min=0.40, v_min_frac=0.48,
                  max_turn_deg=4.5, cam_h=args.cam_h, floor_y=floor_y)
        start = None
        for _ in range(30):
            s = pf.get_random_navigable_point()
            ok, gd, pts = geodesic(pf, s, A)
            if ok and args.dA_min <= gd <= args.dA_max and g2._geo_on_floor(pts, floor_y, ftol):
                start = s
                break
        if start is None:
            continue
        ok, gd, pts = geodesic(pf, start, A)
        if not ok or not g2._geo_on_floor(pts, floor_y, ftol):
            continue
        frames, ok = roll_leg(pts, pf, E, eb, cp, None, None, None)
        if ok and len(frames) >= args.min_frames:
            return frames
    return None


def sweep_trajectory(sim, rng, args, esdf_cache, traj_dir):
    pf = sim.pathfinder
    frames = make_leg(sim, rng, args, esdf_cache)
    if frames is None:
        return False
    n = len(frames)
    lo = args.num_scale + args.window - 1                       # memnav valid-match low bound
    if n < lo + 30:
        return False

    # ---- render leg A ----
    rgbs, depths, poses = [], [], []
    for pos, yaw in frames:
        c, d = render(sim, pos, yaw)
        rgbs.append(c)
        depths.append(d)
        poses.append(cam_to_world_hab(pos, yaw))

    # ---- anchors, evenly spaced in the valid range ----
    a0, a1 = lo + 5, n - 10
    anchors = sorted(set(int(round(a)) for a in np.linspace(a0, a1, args.anchors)))

    covis_idxs = list(range(0, n, args.covis_stride))
    sweep_dir = os.path.join(traj_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    records, sid, n_drop = [], 0, 0

    def eval_goal(cam, yaw):
        """Render + covisibility bundle for a goal camera. Returns (rgb, rec_fields)."""
        rgb, dep = render(sim, cam, yaw)
        p_world = to_world(backproject(dep, args.px_stride), cam_to_world_hab(cam, yaw))
        curve = covis_curve(p_world, poses, depths, covis_idxs, args.covis_tol)
        best = max(curve, key=curve.get) if curve else -1
        return rgb, dep, dict(
            covis_per_frame={str(k): round(v, 4) for k, v in curve.items()},
            covis_max=round(max(curve.values()), 4) if curve else 0.0,
            covis_argmax=int(best),
        )

    # ---- grid goals around each anchor ----
    for m in anchors:
        a_pos, a_yaw = frames[m]
        a_pts = to_world(backproject(depths[m], args.px_stride), poses[m])
        for fwd in args.fwd:
            for lat in (args.lat if fwd > 0 else [0.0]):
                for dyaw in args.dyaw:
                    placed = place_goal(pf, a_pos, a_yaw, fwd, lat, dyaw, args.cam_h)
                    if placed is None:
                        n_drop += 1
                        continue
                    cam, yaw, fwd_a, lat_a = placed
                    rgb, dep, cov = eval_goal(cam, yaw)
                    g_pts = to_world(backproject(dep, args.px_stride), cam_to_world_hab(cam, yaw))
                    fname = f"goal_{sid:04d}.jpg"
                    g2.Image.fromarray(rgb).save(os.path.join(sweep_dir, fname), quality=95)
                    fa, la, dya = actual_rel(a_pos, a_yaw, cam, yaw)
                    records.append(dict(
                        sid=sid, kind="grid", goal_file=fname, anchor_idx=int(m),
                        req=dict(fwd=fwd, lat=lat, dyaw_deg=dyaw),
                        act=dict(fwd=round(fa, 3), lat=round(la, 3), dyaw_deg=round(dya, 2)),
                        gt_T_wc_hab=cam_to_world_hab(cam, yaw).tolist(),
                        covis_goal_in_anchor=round(covis_frac(g_pts, poses[m], depths[m], args.covis_tol), 4),
                        covis_anchor_in_goal=round(covis_frac(a_pts, cam_to_world_hab(cam, yaw), dep, args.covis_tol), 4),
                        **cov,
                    ))
                    sid += 1

    # ---- covis-stratified random goals per anchor (kind="rand"): random navigable
    # viewpoints + random yaw in a disk around the anchor, quota-balanced across
    # covisibility bins. Decouples the covisibility axis from grid placement bias
    # (long-fwd cells only exist where sightlines are long).
    rand_bins = [0.0, 0.05, 0.2, 0.5, 1.01]
    for m in anchors:
        a_pos, a_yaw = frames[m]
        a_pts = to_world(backproject(depths[m], args.px_stride), poses[m])
        floor = np.array([a_pos[0], a_pos[1] - args.cam_h, a_pos[2]])
        quota = [args.rand_per_bin] * (len(rand_bins) - 1)
        tries = 0
        while sum(quota) > 0 and tries < args.rand_per_bin * 80:
            tries += 1
            rr = rng.uniform(0.5, args.rand_radius)
            th = rng.uniform(-np.pi, np.pi)
            q = np.array(pf.snap_point(floor + np.array([rr * np.cos(th), 0.0, rr * np.sin(th)])), float)
            if not pf.is_navigable(q) or abs(q[1] - floor[1]) > 0.5:
                continue
            cam = q + np.array([0.0, args.cam_h, 0.0])
            yaw = float(rng.uniform(-np.pi, np.pi))
            rgb, dep, cov = eval_goal(cam, yaw)
            g_pts = to_world(backproject(dep, args.px_stride), cam_to_world_hab(cam, yaw))
            c_anchor = covis_frac(g_pts, poses[m], depths[m], args.covis_tol)
            b = min(max(int(np.searchsorted(rand_bins, c_anchor, side="right")) - 1, 0), len(quota) - 1)
            if quota[b] <= 0:
                continue
            quota[b] -= 1
            fname = f"goal_{sid:04d}.jpg"
            g2.Image.fromarray(rgb).save(os.path.join(sweep_dir, fname), quality=95)
            fa, la, dya = actual_rel(a_pos, a_yaw, cam, yaw)
            records.append(dict(
                sid=sid, kind="rand", goal_file=fname, anchor_idx=int(m),
                req=None,
                act=dict(fwd=round(fa, 3), lat=round(la, 3), dyaw_deg=round(dya, 2)),
                gt_T_wc_hab=cam_to_world_hab(cam, yaw).tolist(),
                covis_goal_in_anchor=round(c_anchor, 4),
                covis_anchor_in_goal=round(covis_frac(a_pts, cam_to_world_hab(cam, yaw), dep, args.covis_tol), 4),
                **cov,
            ))
            sid += 1

    # ---- negative controls: navigable viewpoints never covisible with leg A ----
    n_neg, tries = 0, 0
    while n_neg < args.n_neg and tries < args.n_neg * 40:
        tries += 1
        q = np.array(pf.snap_point(pf.get_random_navigable_point()), float)
        if not pf.is_navigable(q) or abs(q[1] - (frames[0][0][1] - args.cam_h)) > 0.5:
            continue
        cam = q + np.array([0.0, args.cam_h, 0.0])
        yaw = float(rng.uniform(-np.pi, np.pi))
        rgb, dep, cov = eval_goal(cam, yaw)
        if cov["covis_max"] >= args.neg_covis:
            continue
        fname = f"goal_{sid:04d}.jpg"
        g2.Image.fromarray(rgb).save(os.path.join(sweep_dir, fname), quality=95)
        records.append(dict(
            sid=sid, kind="neg", goal_file=fname, anchor_idx=-1,
            req=None, act=None,
            gt_T_wc_hab=cam_to_world_hab(cam, yaw).tolist(),
            covis_goal_in_anchor=0.0, covis_anchor_in_goal=0.0, **cov,
        ))
        sid += 1
        n_neg += 1

    if sid == 0:
        return False

    # ---- write InternData-N1 layout + sweep meta ----
    meta = dict(scene=os.path.basename(args.scene), n_frames=n, kind="revisit_sweep_legA")
    save_traj(traj_dir, rgbs, depths, poses, meta, [])
    sweep_meta = dict(
        scene=os.path.basename(args.scene), n_frames=n, anchors=anchors,
        num_scale=args.num_scale, window=args.window, cam_h=args.cam_h,
        covis_stride=args.covis_stride, covis_tol=args.covis_tol,
        grid=dict(fwd=args.fwd, lat=args.lat, dyaw=args.dyaw),
        n_grid=sum(r["kind"] == "grid" for r in records),
        n_rand=sum(r["kind"] == "rand" for r in records),
        n_neg=n_neg, n_dropped=n_drop,
        leg_T_wc_hab=[p.tolist() for p in poses],
        records=records,
    )
    with open(os.path.join(sweep_dir, "sweep_meta.json"), "w") as f:
        json.dump(sweep_meta, f)
    print(f"  frames={n} anchors={anchors} grid={sweep_meta['n_grid']} "
          f"rand={sweep_meta['n_rand']} neg={n_neg} dropped={n_drop}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--navmesh", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", default="mp3d_sweep")
    ap.add_argument("--n", type=int, default=4, help="trajectories to generate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dA_min", type=float, default=5.0)
    ap.add_argument("--dA_max", type=float, default=12.0)
    ap.add_argument("--min_frames", type=int, default=80)
    ap.add_argument("--anchors", type=int, default=4, help="anchor frames X per trajectory")
    ap.add_argument("--fwd", type=float, nargs="+", default=DEF_FWD)
    ap.add_argument("--lat", type=float, nargs="+", default=DEF_LAT)
    ap.add_argument("--dyaw", type=float, nargs="+", default=DEF_DYAW)
    ap.add_argument("--n_neg", type=int, default=12)
    ap.add_argument("--rand_per_bin", type=int, default=3,
                    help="covis-stratified random goals per (anchor, covis bin)")
    ap.add_argument("--rand_radius", type=float, default=6.0)
    ap.add_argument("--neg_covis", type=float, default=0.02)
    ap.add_argument("--covis_stride", type=int, default=3)
    ap.add_argument("--covis_tol", type=float, default=0.3)
    ap.add_argument("--px_stride", type=int, default=6)
    ap.add_argument("--cam_h", type=float, default=0.5)
    ap.add_argument("--num_scale", type=int, default=8)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--agent_radius", type=float, default=0.30)
    ap.add_argument("--esdf_res", type=float, default=0.05)
    ap.add_argument("--floor_tol", type=float, default=0.8, help="max |y-floor| to count as same floor (m)")
    ap.add_argument("--start_idx", type=int, default=0, help="first trajectory index (resume)")
    args = ap.parse_args()

    sim = g2.make_sim(args.scene, args.navmesh, agent_radius=args.agent_radius)
    assert sim.pathfinder.is_loaded, "navmesh not loaded"
    esdf_cache = {}   # per-floor ESDF cache (multi-floor MP3D; built on demand from each A.floor)
    rng = np.random.default_rng(args.seed)

    scene_stem = os.path.splitext(os.path.basename(args.scene))[0]
    scene_dir = os.path.join(args.out, args.group, scene_stem)
    made, ti = 0, args.start_idx
    while made < args.n and ti < args.start_idx + args.n * 5:
        traj_dir = os.path.join(scene_dir, f"trajectory_{ti:03d}")
        print(f"[traj {ti}]")
        if sweep_trajectory(sim, rng, args, esdf_cache, traj_dir):
            made += 1
        ti += 1
    print(f"DONE: {made}/{args.n} trajectories -> {scene_dir}")
    sim.close()


if __name__ == "__main__":
    main()
