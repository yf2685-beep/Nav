"""Genuine two-leg (start -> A -> B) episode generator in Habitat.

Prototype target: apartment_1 test scene (no MP3D license needed). Same code will
point at MP3D `<id>.glb` once the license clears.

Why two legs: the manufactured memnav "seen" case reverses leg A exactly (robot goes
to A, turns 180, retraces). Real returns differ. Here leg B = an INDEPENDENT geodesic
A->B, so the return path genuinely diverges from reversing leg A.

Per episode:
  1. sample A (navigable), start (navigable, geodesic start->A in [dA_min, dA_max]).
  2. geodesic start->A = path_A.
  3. sample B ON the start->A corridor (point of path_A at ~b_frac arc-length, snapped
     to navmesh), require geodesic(A,B) >= b_min so leg B is a real backtrack.
  4. geodesic A->B = path_B  (independent; not the reverse of path_A).
  5. roll out start->A then A->B (densify polylines to ~step_m, heading=tangent with
     turn-in-place at corners); render RGB+D at d435i intrinsics each frame.
  6. FoV check: B must project into >= min_seen leg-A frames, unoccluded (the "seen"
     premise). Reject episode otherwise.
  7. write InternData-N1 layout + goal image (B viewed at the heading robot passed it)
     + meta (switch_idx, b_idx, b_seen_frames, geo dists).

Frames: we compute/render entirely in Habitat's native Y-up frame. Poses are written to
the parquet in a Z-up camera-to-world convention (M_W below) to mirror InternData-N1's
layout. Exact axis-sign match to InternData-N1 is finalized during MP3D render-validation;
here we self-verify geometry (project B back into frames; leg-B != reversed leg-A).

Run (habitat env):
  python gen_twoleg.py --scene .../apartment_1.glb --navmesh .../apartment_1.navmesh \
     --n 5 --out /home/asus/Research/Nav/memnav_viz/twoleg_proto --seed 0 --debug
"""
import argparse, os, json
import numpy as np
import quaternion
from PIL import Image

W, H = 480, 270
FX, FY, CX, CY = 355.81464, 351.687, 240.0, 135.0
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], float)
HFOV_DEG = float(np.degrees(2 * np.arctan(CX / FX)))  # ~68.0

# Habitat(Y-up) -> stored data(Z-up) rotation:  (x,y,z)_hab -> (x,-z,y)_data  (det=+1)
M_W = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)


def make_sim(glb, navmesh, agent_radius=0.30, agent_height=1.5):
    import habitat_sim, magnum as mn
    bk = habitat_sim.SimulatorConfiguration(); bk.scene_id = glb; bk.enable_physics = False
    def cam(uuid, typ):
        s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = typ
        s.resolution = [H, W]; s.hfov = HFOV_DEG; s.position = mn.Vector3(0, 0, 0); return s
    ac = habitat_sim.agent.AgentConfiguration()
    ac.sensor_specifications = [cam("color", habitat_sim.SensorType.COLOR),
                               cam("depth", habitat_sim.SensorType.DEPTH)]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(bk, [ac]))
    if navmesh and os.path.isfile(navmesh):
        sim.pathfinder.load_nav_mesh(navmesh)
    # Re-bake the navmesh at the robot radius so the free space (and every geodesic) keeps
    # real clearance from walls and centres through doorways (PythonRobotics/iPlanner: inflate
    # by robot radius before planning). agent_radius ~0.3 matches iPlanner's robot_size.
    ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
    ns.agent_radius = agent_radius; ns.agent_height = agent_height
    ok = sim.recompute_navmesh(sim.pathfinder, ns)
    print(f"[make_sim] recompute_navmesh(agent_radius={agent_radius}) ok={ok} "
          f"navigable_area={sim.pathfinder.navigable_area:.1f} m^2")
    return sim


def detect_floors(pf, n=5000, gap=1.0):
    """MP3D scenes are multi-floor; the navmesh spans all floors. Cluster navigable-point
    heights into floors (split where the height gap > `gap` m). Returns [(floor_y, count)]
    sorted by count desc (count ~ floor area, used to weight floor choice)."""
    ys = np.sort(np.array([pf.get_random_navigable_point()[1] for _ in range(n)]))
    floors, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] > gap:
            floors.append(cur); cur = []
        cur.append(y)
    floors.append(cur)
    out = [(float(np.median(f)), len(f)) for f in floors]
    return sorted(out, key=lambda t: -t[1])


def build_esdf(pf, floor_y, res=0.05, pad=0.5, floor_tol=0.8):
    """2D ESDF (distance-to-navmesh-boundary, metres) over habitat x-z FOR ONE FLOOR: a cell is
    free only if snapping (x, floor_y, z) lands on this floor (|q.y - floor_y| < floor_tol), so
    other floors don't leak into the 2D map. = iPlanner cost map, per floor."""
    from scipy import ndimage
    lo, hi = pf.get_bounds()
    x0, z0 = float(lo[0]) - pad, float(lo[2]) - pad
    x1, z1 = float(hi[0]) + pad, float(hi[2]) + pad
    nx = int((x1 - x0) / res) + 1; nz = int((z1 - z0) / res) + 1
    free = np.zeros((nz, nx), bool)
    for iz in range(nz):
        for ix in range(nx):
            gx, gz = x0 + ix * res, z0 + iz * res
            q = pf.snap_point([gx, floor_y, gz])
            free[iz, ix] = (pf.is_navigable(q) and abs(q[0] - gx) < res and abs(q[2] - gz) < res
                            and abs(q[1] - floor_y) < floor_tol)
    dist = ndimage.distance_transform_edt(free) * res
    gzd, gxd = np.gradient(dist, res)
    return dict(dist=dist, gx=gxd, gz=gzd, x0=x0, z0=z0, res=res, nx=nx, nz=nz, floor_y=float(floor_y))


def sample_esdf(E, x, z):
    """Bilinear (clearance, grad_xz) at habitat (x,z)."""
    fx = (x - E["x0"]) / E["res"]; fz = (z - E["z0"]) / E["res"]
    ix = int(np.clip(fx, 0, E["nx"] - 1)); iz = int(np.clip(fz, 0, E["nz"] - 1))
    return float(E["dist"][iz, ix]), np.array([float(E["gx"][iz, ix]), float(E["gz"][iz, ix])])


def geodesic(pf, a, b):
    import habitat_sim
    p = habitat_sim.ShortestPath(); p.requested_start = a; p.requested_end = b
    ok = pf.find_path(p)
    return (ok, float(p.geodesic_distance), [np.array(x, float) for x in p.points])


def densify(points, step_m=0.20):
    """polyline -> list of positions spaced ~step_m along it."""
    out = []
    for i in range(len(points) - 1):
        p, q = points[i], points[i + 1]
        seg = q - p; L = np.linalg.norm(seg)
        n = max(1, int(np.ceil(L / step_m)))
        for k in range(n):
            out.append(p + seg * (k / n))
    out.append(points[-1])
    return out


def yaw_facing(delta_xz):
    """heading (Habitat, rot about +Y) so camera -Z faces horizontal direction delta."""
    dx, dz = delta_xz
    return np.arctan2(-dx, -dz)  # camera forward = -Z


def elastic_smooth(pts, pf, E, iters=60, kc=0.5, kr=0.8, rho0=0.6, step=0.04, res=0.05):
    """ElasticBands smoothing of a geodesic (PythonRobotics ElasticBands / iPlanner ESDF cost):
    each interior point feels a contraction force (smoothness, toward neighbour midpoint) and
    a repulsion force up the clearance gradient when clearance rho < rho0. Endpoints fixed;
    every update snapped back onto the navmesh so it stays feasible. Works in habitat x-z;
    returns smoothed 3D points (x, floor_y, z)."""
    P = np.array(densify(pts, res))          # dense, habitat (x,y,z)
    g = P[:, [0, 2]].astype(float)           # ground plane (x,z)
    y = float(P[0, 1])
    for _ in range(iters):
        ng = g.copy()
        for i in range(1, len(g) - 1):
            dp = g[i - 1] - g[i]; dn = g[i + 1] - g[i]
            contraction = kc * (dp / (np.linalg.norm(dp) + 1e-6) + dn / (np.linalg.norm(dn) + 1e-6))
            rho, grad = sample_esdf(E, g[i, 0], g[i, 1])
            repulsion = kr * (rho0 - rho) * grad if rho < rho0 else np.zeros(2)
            cand = g[i] + step * (contraction + repulsion)
            q = pf.snap_point([cand[0], y, cand[1]])              # keep feasible
            ng[i] = [q[0], q[2]]
        g = ng
    out = np.stack([g[:, 0], np.full(len(g), y), g[:, 1]], axis=1)
    return [out[i] for i in range(len(out))]


def pursuit_track(ref_pts, pf, init_pos=None, init_theta=None,
                  v_max=0.0376, L=0.7, r_min=0.40, v_min_frac=0.48,
                  max_turn_deg=4.5, cam_h=0.5, stop_before=0.0, floor_y=0.0):
    """Pure-pursuit unicycle tracking of a geodesic polyline, matching InternData-N1's
    controller (v≈0.0376 m/frame, lookahead 0.7 m, min radius 0.4 m). Produces smooth,
    COUPLED motion (no in-place spin): cruises with pursuit curvature (radius >= r_min);
    for a target behind (|alpha|>90°, e.g. the A->B U-turn) it rotates at the max angular
    rate while creeping (a tight arc, not a frozen pivot). Snaps to navmesh for collisions."""
    ref = np.array(densify(ref_pts, 0.05))
    refg = ref[:, [0, 2]]                                   # ground plane (x,z)
    kappa_max = 1.0 / r_min
    mturn = np.deg2rad(max_turn_deg)
    pos = (np.asarray(init_pos)[[0, 2]] if init_pos is not None else refg[0]).astype(float)
    if init_theta is not None:
        theta = float(init_theta)
    else:
        d = refg[min(5, len(refg) - 1)] - pos; theta = np.arctan2(d[1], d[0])
    ci, frames, goal = 0, [], refg[-1]
    arclen = float(np.sum(np.linalg.norm(np.diff(refg, axis=0), axis=1)))
    max_steps = int(arclen / v_max * 4) + 300     # generous vs ideal frame count
    stall = 0
    for _ in range(max_steps):
        prev = pos.copy()
        ci += int(np.argmin(np.linalg.norm(refg[ci:ci + 40] - pos, axis=1)))
        li, acc = ci, 0.0
        while li + 1 < len(refg) and acc < L:
            acc += np.linalg.norm(refg[li + 1] - refg[li]); li += 1
        to = refg[li] - pos
        alpha = (np.arctan2(to[1], to[0]) - theta + np.pi) % (2 * np.pi) - np.pi
        # Smooth, single-branch control law (no bang-bang) so v/omega/radius vary
        # continuously through a turn -> no delta-spikes, matches N1's motion:
        #  * speed eases off in turns (N1: corr(turn,speed)=-0.49, floor ~0.48*cruise)
        #  * curvature is proportional to heading error, capped at kappa_max=1/r_min,
        #    so the tightest turn (incl. the U-turn) holds radius r_min, not a spin.
        v = v_max * (v_min_frac + (1 - v_min_frac) * (1 + np.cos(alpha)) / 2)
        kappa = np.clip(2.0 * alpha / L, -kappa_max, kappa_max)
        dtheta = np.clip(kappa * v, -mturn, mturn)
        theta += dtheta
        ng = pos + v * np.array([np.cos(theta), np.sin(theta)])
        snap = pf.snap_point([ng[0], floor_y, ng[1]])
        if (not pf.is_navigable(snap)) or np.hypot(snap[0] - ng[0], snap[2] - ng[1]) > 0.25:
            ng = pos + 0.3 * v * np.array([np.cos(theta), np.sin(theta)])   # creep if blocked
            snap = pf.snap_point([ng[0], floor_y, ng[1]])
        pos = np.array([snap[0], snap[2]])
        frames.append((np.array([snap[0], snap[1] + cam_h, snap[2]]), theta))
        if stop_before > 0 and np.linalg.norm(pos - goal) < stop_before:
            return frames, True           # hand off to arrive_facing for the terminal pose
        if ci >= len(refg) - 2 and np.linalg.norm(pos - goal) < 0.15:
            return frames, True
        stall = stall + 1 if np.linalg.norm(pos - prev) < 0.004 else 0
        if stall > 40:                       # wedged against geometry -> give up
            return frames, False
    return frames, np.linalg.norm(pos - goal) < 0.3


def arrive_facing(init_pos, init_theta, goal_xz, goal_yaw, pf,
                  v_max=0.0376, r_min=0.40, v_min_frac=0.48, max_turn_deg=4.5, cam_h=0.5,
                  ka=1.0, kb=0.35, pos_tol=0.15, max_steps=320, floor_y=0.0):
    """Option D: Astolfi pose-to-pose feedback (PythonRobotics move_to_pose) — curve from
    (init_pos, init_theta) to arrive AT goal_xz FACING goal_yaw with NO in-place pivot.
    Forward-only, per-frame caps (v_max, max_turn, slows in sharp turns so it can tighten the
    alignment curve), navmesh-snapped. Returns (frames, ok)."""
    mturn = np.deg2rad(max_turn_deg)
    pos = np.asarray(init_pos, float)[[0, 2]] if len(np.asarray(init_pos)) == 3 else np.asarray(init_pos, float)
    theta = float(init_theta); frames = []; stall = 0
    for _ in range(max_steps):
        d = np.asarray(goal_xz, float) - pos
        rho = float(np.hypot(d[0], d[1]))
        alpha = (np.arctan2(d[1], d[0]) - theta + np.pi) % (2 * np.pi) - np.pi
        beta = (goal_yaw - theta - alpha + np.pi) % (2 * np.pi) - np.pi
        herr = (goal_yaw - theta + np.pi) % (2 * np.pi) - np.pi
        if rho < pos_tol and abs(herr) <= mturn:
            snap = pf.snap_point([pos[0], 0.0, pos[1]])
            frames.append((np.array([snap[0], snap[1] + cam_h, snap[2]]), float(goal_yaw)))
            return frames, True
        # speed ramps down near the goal but keeps a small floor so it can FINISH a large
        # alignment turn (a hard v->0 stalls before aligning; too big a floor orbits).
        v = v_max * float(np.clip(rho / 0.5, 0.12, 1.0))
        dtheta = np.clip(ka * alpha - kb * beta, -mturn, mturn)
        theta += dtheta
        ng = pos + v * np.array([np.cos(theta), np.sin(theta)])
        snap = pf.snap_point([ng[0], floor_y, ng[1]])
        if (not pf.is_navigable(snap)) or np.hypot(snap[0] - ng[0], snap[2] - ng[1]) > 0.25:
            ng = pos + 0.3 * v * np.array([np.cos(theta), np.sin(theta)])
            snap = pf.snap_point([ng[0], floor_y, ng[1]])
        prev = pos.copy(); pos = np.array([snap[0], snap[2]])
        frames.append((np.array([snap[0], snap[1] + cam_h, snap[2]]), theta))
        stall = stall + 1 if np.linalg.norm(pos - prev) < 0.003 else 0
        if stall > 40:
            return frames, False
    return frames, False          # ran out of budget without a clean pose arrival -> caller falls back


def agent_state(pos, yaw):
    import habitat_sim
    st = habitat_sim.agent.AgentState()
    st.position = pos
    st.rotation = quaternion.from_rotation_vector([0, yaw, 0])
    return st


def cam_to_world_hab(pos, yaw):
    """4x4 camera-to-world in Habitat frame (cam optical = OpenGL: -Z fwd, +Y up)."""
    R = quaternion.as_rotation_matrix(quaternion.from_rotation_vector([0, yaw, 0]))
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
    return T


def project_point(T_wc_hab, p_world_hab, depth_img):
    """Project world point into camera; return (u,v,z_fwd, visible, unoccluded)."""
    Rwc = T_wc_hab[:3, :3]; t = T_wc_hab[:3, 3]
    p_cam = Rwc.T @ (p_world_hab - t)            # habitat optical (-Z fwd, +Y up)
    p_cv = np.array([p_cam[0], -p_cam[1], -p_cam[2]])  # -> OpenCV (+Z fwd, +Y down)
    z = p_cv[2]
    if z <= 1e-3:
        return None
    u = FX * p_cv[0] / z + CX; v = FY * p_cv[1] / z + CY
    inb = (0 <= u < W) and (0 <= v < H)
    if not inb:
        return (u, v, z, False, False)
    d_ren = float(depth_img[int(v), int(u)])
    unocc = (d_ren <= 0) or (z <= d_ren + 0.25)   # B not clearly behind a surface
    return (u, v, z, True, unocc)


def render(sim, pos, yaw):
    sim.get_agent(0).set_state(agent_state(pos, yaw))
    o = sim.get_sensor_observations()
    return o["color"][..., :3].copy(), o["depth"].copy()


# --------------------------------------------------------------------------- #
# Ground-truth co-visibility (occlusion-checked reprojection). This is the
# operational "revisit" measure: a goal view is a revisit of a history frame if
# they co-observe enough of the same 3D surface. (Shared with revisit_sweep_gen.py.)
# --------------------------------------------------------------------------- #
def backproject(depth, stride=6, d_min=0.15, d_max=10.0):
    """depth [H,W] -> surface points in the CAMERA frame (habitat optical: -Z fwd, +Y up),
    subsampled on a `stride` pixel grid. [Np, 3]."""
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    d = depth[vs, us].astype(float)
    m = (d > d_min) & (d < d_max)
    u, v, d = us[m].astype(float), vs[m].astype(float), d[m]
    x = (u - CX) / FX * d
    y = (v - CY) / FY * d
    return np.stack([x, -y, -d], axis=1)


def to_world(p_cam, T_wc):
    return p_cam @ T_wc[:3, :3].T + T_wc[:3, 3]


def covis_frac(p_world, T_wc, depth, tol=0.3):
    """Fraction of world surface points `p_world` co-observed by camera (T_wc, depth):
    inside the frustum AND rendered depth agrees with reprojected range (not occluded)."""
    if len(p_world) == 0:
        return 0.0
    pc = (p_world - T_wc[:3, 3]) @ T_wc[:3, :3]
    x, y, z = pc[:, 0], -pc[:, 1], -pc[:, 2]
    m = z > 0.05
    zs = np.maximum(z, 1e-6)
    u = FX * x / zs + CX; v = FY * y / zs + CY
    m &= (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui = np.clip(u.astype(int), 0, W - 1); vi = np.clip(v.astype(int), 0, H - 1)
    good = m & (np.abs(z - depth[vi, ui]) <= tol)
    return float(good.sum()) / len(p_world)


def max_covis(goal_pts_world, poses, depths, stride=4, tol=0.3):
    """Max co-visibility of a goal view (its world surface points) over history frames,
    and the argmax frame index. poses/depths are the rendered leg frames (subsampled)."""
    best, bi = 0.0, -1
    for i in range(0, len(poses), stride):
        c = covis_frac(goal_pts_world, poses[i], depths[i], tol)
        if c > best:
            best, bi = c, i
    return best, bi


def covis_curve(goal_pts_world, poses, depths, tol=0.3):
    """Occlusion-aware co-visibility of a goal view vs EVERY history frame (stride 1). Index i
    aligns with global frame i (history = legs concatenated in order). This is the multi-positive
    retrieval label: the loader thresholds it into positives (>=pos_hi) / negatives (<=pos_lo) /
    ignore-band, and its argmax is the relocalization anchor."""
    return np.array([covis_frac(goal_pts_world, poses[i], depths[i], tol) for i in range(len(poses))], float)


def save_traj(out_dir, rgbs, depths, poses_hab, meta, goal_rgbs):
    import pandas as pd
    rgb_d = os.path.join(out_dir, "videos/chunk-000/observation.images.rgb")
    dep_d = os.path.join(out_dir, "videos/chunk-000/observation.images.depth")
    dat_d = os.path.join(out_dir, "data/chunk-000"); met_d = os.path.join(out_dir, "meta")
    for d in (rgb_d, dep_d, dat_d, met_d):
        os.makedirs(d, exist_ok=True)
    for i, (c, dep) in enumerate(zip(rgbs, depths)):
        Image.fromarray(c).save(os.path.join(rgb_d, f"{i}.jpg"), quality=95)
        du16 = np.clip(dep * 10000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(du16).save(os.path.join(dep_d, f"{i}.png"))
    # one goal image per stop (goal_1.jpg = B, goal_2.jpg = C, ...); goal_image.jpg = first (B)
    for k, g in enumerate(goal_rgbs, start=1):
        Image.fromarray(g).save(os.path.join(out_dir, f"goal_{k}.jpg"), quality=95)
    if goal_rgbs:
        Image.fromarray(goal_rgbs[0]).save(os.path.join(out_dir, "goal_image.jpg"), quality=95)
    # poses -> stored Z-up camera-to-world; extrinsic = identity mount (we bake full pose)
    ext = np.eye(4)
    rows = []
    for i, Tw in enumerate(poses_hab):
        Td = np.eye(4); Td[:3, :3] = M_W @ Tw[:3, :3]; Td[:3, 3] = M_W @ Tw[:3, 3]
        rows.append({"index": i,
                     "observation.camera_intrinsic": K.astype(np.float32).tolist(),
                     "observation.camera_extrinsic": ext.astype(np.float32).tolist(),
                     "action": Td.astype(np.float32).tolist()})
    pd.DataFrame(rows).to_parquet(os.path.join(dat_d, "episode_000000.parquet"))
    json.dump(meta, open(os.path.join(met_d, "gen_meta.json"), "w"), indent=2)


def align_turn(pos, yaw0, yaw1, max_turn_deg):
    """Bounded-rate in-place rotation from yaw0 to the goal-image orientation yaw1 (image-goal:
    after reaching the goal POSITION, turn so the final view matches the goal image)."""
    dy = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    step = np.deg2rad(max_turn_deg)
    n = int(abs(dy) // step)
    fr = [(pos.copy(), yaw0 + np.sign(dy) * step * k) for k in range(1, n + 1)]
    fr.append((pos.copy(), float(yaw1)))
    return fr


def heading_at_closest(frames, G):
    """(travel heading, index) of the frame closest to point G — the view the robot had when it
    passed G. Used as G's goal-image orientation (well-defined, unlike direction-to-G)."""
    pts = np.array([f[0] for f in frames])
    i = int(np.argmin(np.linalg.norm(pts - np.asarray(G, float), axis=1)))
    return float(frames[i][1]), i


def heading_at_closest_multi(legs, G):
    best = None
    for lg in legs:
        y, i = heading_at_closest(lg, G)
        d = float(np.linalg.norm(np.array(lg[i][0]) - np.asarray(G, float)))
        if best is None or d < best[0]:
            best = (d, y)
    return best[1]


def roll_leg(geo_pts, pf, E, eb, cp, init_pos, init_theta, goal=None, goal_yaw=None, arrive=False):
    """One leg: ElasticBands-smooth the geodesic -> pursuit-track (N1 dynamics) to the goal
    POSITION. NO terminal orientation alignment: the goal image is only a recognition /
    relocalization target (retrieval finds the best-match history frame X, inserts the goal to
    read its map pose), so the robot need only reach the goal position — its arrival heading is
    the natural approach heading. Keeps the whole trajectory smooth (no pivot / loop).
    (goal/goal_yaw/arrive kept for signature compatibility; unused for the trajectory.)"""
    s = elastic_smooth(geo_pts, pf, E, **eb)
    return pursuit_track(s, pf, init_pos=init_pos, init_theta=init_theta, **cp)


def _clear(pf, p, rmin):
    return pf.is_navigable(p) and pf.distance_to_closest_obstacle(p) >= rmin


def _goal_world_pts(sim, gpos_floor, gyaw, ch):
    """Render a candidate goal view; return its world surface points (for co-visibility)."""
    _, dep = render(sim, np.asarray(gpos_floor, float) + ch, gyaw)
    T = cam_to_world_hab(np.asarray(gpos_floor, float) + ch, gyaw)
    return to_world(backproject(dep, stride=6), T)


def _render_leg(sim, frames):
    poses, depths, rgbs = [], [], []
    for pos, yaw in frames:
        c, d = render(sim, pos, yaw); rgbs.append(c); depths.append(d)
        poses.append(cam_to_world_hab(pos, yaw))
    return poses, depths, rgbs


def sample_revisit(sim, pf, hist_frames, hist_poses, hist_depths, n_anchor, rng, args, ch,
                   floor_y, source=None, min_geo=0.0):
    """A perturbed REVISIT goal, ANCHOR-CENTRIC (same parameterisation as revisit_sweep_gen).
    hist_frames/poses/depths cover the FULL history the retrieval head sees at this goal's step
    (leg1 for B; leg1+leg2 for C); n_anchor = #leading frames that ARE the revisit target (leg1),
    so leg2 frames become hard NEGATIVES for C but are never sampled as the anchor.
      1. pick a random anchor frame X from the target leg ([anchor_margin, n_anchor));
      2. sample B position in a UNIFORM DISK of radius goal_jitter_pos around X, snapped to navmesh;
      3. sample B heading in a +/- head_max_deg CONE around X's heading;
      4. cheap stride gate on covis in [covis_lo, covis_hi]; on pass, compute the stride-1 covisibility
         curve over EVERY history frame -> argmax = GT relocalization anchor, curve = multi-positive label.
    Returns (pos, goal_yaw, covis, matched_frame, head_off_deg, covis_curve) or None."""
    lo = min(args.anchor_margin, max(0, n_anchor - 1))
    # long-term (implicit-memory) vs recent (in-view): for a fraction of revisits, force the matched
    # frame OUTSIDE the current window. current frame = last history frame (len(hist)-1); gap = current-X.
    hi = n_anchor
    if rng.uniform() < args.long_term_frac:
        u2 = len(hist_frames) - args.min_recall_gap          # X <= current - min_recall_gap
        if u2 > lo:
            hi = min(n_anchor, u2)
    if hi <= lo:
        hi = n_anchor                                         # long-term range empty (short leg) -> free
    R = args.goal_jitter_pos
    for _ in range(args.goal_tries):
        xi = int(rng.integers(lo, hi)) if hi > lo else int(rng.integers(0, n_anchor))
        Xp = hist_frames[xi][0]; Xyaw = float(hist_frames[xi][1])
        r = R * np.sqrt(rng.uniform()); th = rng.uniform(0, 2 * np.pi)      # uniform in the disk
        p = np.array(pf.snap_point([float(Xp[0] + r * np.cos(th)), floor_y, float(Xp[2] + r * np.sin(th))]), float)
        if not _clear(pf, p, args.r_min) or not _on_floor(p, floor_y, args.floor_tol):
            continue
        if source is not None:
            ok, gd, _ = geodesic(pf, source, p)
            if not ok or gd < min_geo:
                continue
        gyaw = Xyaw + np.deg2rad(rng.uniform(-args.head_max_deg, args.head_max_deg))   # cone around X
        gpts = _goal_world_pts(sim, p, gyaw, ch)
        cov, _ = max_covis(gpts, hist_poses, hist_depths, stride=args.covis_stride, tol=args.covis_tol)
        if not (args.covis_lo <= cov <= args.covis_hi):          # cheap subsampled GATE (reject fast)
            continue
        # On accept, the true label is the FULL stride-1 covisibility curve over every history frame:
        # position jitter means the sampling anchor X is not necessarily the best match, so the GT
        # matched frame is the curve argmax (relocalization anchor) and the curve is the retrieval label.
        curve = covis_curve(gpts, hist_poses, hist_depths, tol=args.covis_tol)
        # the relocalization anchor must be goal_append-able (>= anchor_margin: window disjoint from scale);
        # restrict the argmax to the valid range so we never store a match LingBot can't reconstruct.
        # frames < anchor_margin are ignore for retrieval (loader masks them; anchor_margin recorded in meta).
        vlo = min(args.anchor_margin, len(curve) - 1)
        ai = vlo + int(curve[vlo:].argmax()); cov = float(curve[ai])
        head_off = abs((gyaw - hist_frames[ai][1] + np.pi) % (2 * np.pi) - np.pi)       # vs GT matched frame
        if (args.covis_lo <= cov <= args.covis_hi) and head_off <= np.deg2rad(args.head_max_deg):
            return p, float(gyaw), float(cov), int(ai), float(np.degrees(head_off)), curve
    return None


def sample_novel(sim, pf, A, hist_poses, hist_depths, rng, args, ch, floor_y):
    """A NOVEL goal: on THIS floor, navigable, clearance>=r_min, geodesic(A,.)>min_dist_AB, and
    max co-visibility with history < novel_covis (retrieval target = null -> all frames negative).
    Returns (pos_floor, goal_yaw, covis, covis_curve) or None."""
    for _ in range(args.goal_tries):
        p = np.array(pf.get_random_navigable_point(), float)
        if not _clear(pf, p, args.r_min) or not _on_floor(p, floor_y, args.floor_tol):
            continue
        ok, gd, _ = geodesic(pf, A, p)
        if not ok or gd < args.min_dist_AB:
            continue
        gyaw = yaw_facing((p - np.asarray(A, float))[[0, 2]])   # view along the approach
        gpts = _goal_world_pts(sim, p, gyaw, ch)
        cov, _ai = max_covis(gpts, hist_poses, hist_depths, stride=args.covis_stride, tol=args.covis_tol)
        if cov >= args.novel_covis:                              # cheap subsampled reject
            continue
        # confirm genuinely unseen over EVERY frame: the stride gate can SKIP the one frame that
        # observed p and wrongly call it novel. The full curve doubles as the (all-negative) label.
        curve = covis_curve(gpts, hist_poses, hist_depths, tol=args.covis_tol)
        if float(curve.max()) < args.novel_covis:
            return p, float(gyaw), float(curve.max()), curve
    return None


def _on_floor(p, floor_y, tol):
    return abs(float(p[1]) - floor_y) < tol


def _geo_on_floor(pts, floor_y, tol):
    """True if every geodesic waypoint stays on the floor (no stairs to another level)."""
    return all(abs(float(w[1]) - floor_y) < tol for w in pts)


def _rand_on_floor(pf, floor_y, tol, tries=40):
    for _ in range(tries):
        p = pf.get_random_navigable_point()
        if _on_floor(p, floor_y, tol):
            return p
    return None


def _get_esdf(cache, pf, floor_y, args):
    """Per-floor ESDF, cached by height bucket (stairs are navigable so we can't pre-split
    floors by height gaps; instead each episode's floor is defined by its A, and we build/reuse
    one ESDF per floor bucket)."""
    key = round(floor_y / 0.5)
    if key not in cache:
        cache[key] = build_esdf(pf, floor_y, res=args.esdf_res, floor_tol=args.floor_tol)
    return cache[key]


def make_episode(sim, rng, args, ep_idx, esdf_cache):
    pf = sim.pathfinder
    ftol = args.floor_tol
    eb = dict(iters=args.eb_iters, kc=args.eb_kc, kr=args.eb_kr, rho0=args.eb_rho0,
              step=args.eb_step, res=args.esdf_res)
    ch = np.array([0, args.cam_h, 0])
    for _attempt in range(args.max_attempts):
        # --- A defines the episode's FLOOR (any navigable point); everything else stays on it ---
        A = pf.get_random_navigable_point()
        if not _clear(pf, A, args.r_min):
            continue
        floor_y = float(A[1])
        E = _get_esdf(esdf_cache, pf, floor_y, args)
        cp = dict(v_max=args.v, L=args.lookahead, r_min=args.r_min, v_min_frac=args.v_min_frac,
                  max_turn_deg=args.max_turn_deg, cam_h=args.cam_h, floor_y=floor_y)
        # --- start on A's floor, geodesic start->A stays on floor ---
        start = None
        for _ in range(30):
            s = _rand_on_floor(pf, floor_y, ftol)
            if s is None:
                continue
            ok, gd, pts = geodesic(pf, s, A)
            if ok and args.dA_min <= gd <= args.dA_max and _geo_on_floor(pts, floor_y, ftol):
                start = s; break
        if start is None:
            continue
        okA, gdA, g1 = geodesic(pf, start, A)
        if not okA or not _geo_on_floor(g1, floor_y, ftol):
            continue
        leg1, ok = roll_leg(g1, pf, E, eb, cp, None, None)          # A = fresh forward goal
        if not ok:
            continue
        p1, d1, r1 = _render_leg(sim, leg1)

        # --- B: 2-leg -> REVISIT on leg A ; 3-leg -> NOVEL (off leg A) ---
        # history at B's step = leg1 (n_anchor = all of leg1: the revisit target).
        if args.n_legs == 2:
            rv = sample_revisit(sim, pf, leg1, p1, d1, len(leg1), rng, args, ch, floor_y)
            if rv is None:
                continue
            B, yaw_B, covB, aiB, hoB, curveB = rv; kindB = "revisit"; arriveB = True
        else:
            nv = sample_novel(sim, pf, A, p1, d1, rng, args, ch, floor_y)
            if nv is None:
                continue
            B, yaw_B, covB, curveB = nv; aiB = -1; hoB = None; kindB = "novel"; arriveB = False
        okB, gdB, g2 = geodesic(pf, A, B)
        if not okB or gdB < args.b_min or not _geo_on_floor(g2, floor_y, ftol):
            continue
        leg2, ok = roll_leg(g2, pf, E, eb, cp, leg1[-1][0], leg1[-1][1],
                            goal=B, goal_yaw=yaw_B, arrive=arriveB)
        if not ok:
            continue
        p2, d2, r2 = _render_leg(sim, leg2)
        legs = [leg1, leg2]; R = [(p1, d1, r1), (p2, d2, r2)]
        goals = [dict(name="B", pos=B, yaw=yaw_B, kind=kindB, covis=covB, covis_argmax=aiB,
                      head_off_deg=hoB, covis_curve=curveB)]

        # --- 3-leg: C REVISITS leg A (leg1), reached from B. History at C's step = leg1+leg2, so leg2
        #     frames are hard NEGATIVES; anchor is still sampled from leg1 (n_anchor = len(leg1)). ---
        if args.n_legs >= 3:
            hist_fr = leg1 + leg2; hist_p = p1 + p2; hist_d = d1 + d2
            rv = sample_revisit(sim, pf, hist_fr, hist_p, hist_d, len(leg1), rng, args, ch, floor_y,
                                source=B, min_geo=args.c_min)
            if rv is None:
                continue
            C, yaw_C, covC, aiC, hoC, curveC = rv
            okC, gdC, g3 = geodesic(pf, B, C)
            if not okC or gdC < args.c_min or not _geo_on_floor(g3, floor_y, ftol):
                continue
            leg3, ok = roll_leg(g3, pf, E, eb, cp, leg2[-1][0], leg2[-1][1],
                                goal=C, goal_yaw=yaw_C, arrive=True)
            if not ok:
                continue
            p3, d3, r3 = _render_leg(sim, leg3)
            legs.append(leg3); R.append((p3, d3, r3))
            goals.append(dict(name="C", pos=C, yaw=yaw_C, kind="revisit", covis=covC, covis_argmax=aiC,
                              head_off_deg=hoC, covis_curve=curveC))

        # --- assemble (already rendered per leg) ---
        rgbs = [x for (_p, _d, rr) in R for x in rr]
        depths = [x for (_p, dd, _r) in R for x in dd]
        poses = [x for (pp, _d, _r) in R for x in pp]
        allframes = [f for lg in legs for f in lg]
        switches = [int(s) for s in np.cumsum([len(lg) for lg in legs])[:-1]]

        # --- segment collision backstop ---
        fy = float(allframes[0][0][1] - args.cam_h)
        def _gnav(x, z):
            q = pf.snap_point([x, fy, z])
            return pf.is_navigable(q) and abs(q[0] - x) < 0.06 and abs(q[2] - z) < 0.06
        if any(not _gnav((allframes[i - 1][0][0] + allframes[i][0][0]) / 2,
                         (allframes[i - 1][0][2] + allframes[i][0][2]) / 2)
               for i in range(1, len(allframes))):
            continue

        goal_rgbs = [render(sim, np.asarray(g["pos"], float) + ch, g["yaw"])[0] for g in goals]

        def _d(p):
            return (M_W @ np.asarray(p, float)).tolist()
        meta = dict(scene=os.path.basename(args.scene), ep_idx=ep_idx, n_frames=len(rgbs),
                    n_legs=len(legs), switch_idx=int(switches[0]), switches=switches,
                    start=_d(start), A=_d(A),
                    goals=[dict(name=g["name"], kind=g["kind"], pos=_d(g["pos"]),
                                yaw_habitat=g["yaw"], covis=round(float(g["covis"]), 4),
                                covis_argmax=int(g["covis_argmax"]),
                                head_off_deg=(round(g["head_off_deg"], 1) if g.get("head_off_deg") is not None else None),
                                # recall gap = current(=len history-1) - matched frame; large => long-term memory.
                                recall_gap=(len(g["covis_curve"]) - 1 - int(g["covis_argmax"])
                                            if int(g["covis_argmax"]) >= 0 else None),
                                # multi-positive retrieval label: covis vs every history frame [0..step-1];
                                # loader thresholds into positive/negative/ignore (pos_hi/pos_lo below).
                                covis_curve=[round(float(c), 4) for c in g["covis_curve"]])
                           for g in goals],
                    geo_startA=float(gdA),
                    covis_band=[args.covis_lo, args.covis_hi], novel_covis=args.novel_covis,
                    covis_pos_hi=args.covis_pos_hi, covis_pos_lo=args.covis_pos_lo,
                    # LingBot streaming: valid match range = [anchor_margin, step); loader masks [0,anchor_margin)
                    # from retrieval positives (goal_append can't reconstruct a match below num_scale+window-1).
                    window=args.window, num_scale=args.num_scale, anchor_margin=args.anchor_margin,
                    frame_convention="positions+parquet in data(Zup,M_W); yaw_habitat in render frame")
        return rgbs, depths, poses, meta, goal_rgbs
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True); ap.add_argument("--navmesh", default="")
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dA_min", type=float, default=3.0); ap.add_argument("--dA_max", type=float, default=9.0)
    ap.add_argument("--b_frac", type=float, nargs=2, default=(0.40, 0.60))
    ap.add_argument("--b_min", type=float, default=2.0); ap.add_argument("--b_jitter", type=float, default=0.3)
    ap.add_argument("--n_legs", type=int, default=2, choices=[2, 3],
                    help="2: start->A->B (B revisit on leg A) ; 3: ->C (B novel off leg A, C revisit on leg A)")
    ap.add_argument("--c_min", type=float, default=2.0, help="min geodesic B->C for the 3rd leg")
    # revisit / novel definition (co-visibility) + goal perturbation
    ap.add_argument("--covis_lo", type=float, default=0.20, help="revisit: min max-covisibility with history")
    ap.add_argument("--covis_hi", type=float, default=1.00, help="revisit: max max-covisibility (avoid exact copy)")
    # multi-positive retrieval label thresholds — RECORDED into meta only; the loader applies them to
    # covis_curve (positive >= pos_hi, negative <= pos_lo, ignore-band between). Not used at gen time.
    ap.add_argument("--covis_pos_hi", type=float, default=0.50, help="retrieval positive threshold on covis_curve")
    ap.add_argument("--covis_pos_lo", type=float, default=0.10, help="retrieval negative threshold on covis_curve")
    ap.add_argument("--head_max_deg", type=float, default=45.0,
                    help="revisit: max |goal yaw - matched frame yaw| (relocalizability envelope)")
    ap.add_argument("--novel_covis", type=float, default=0.10, help="novel B: max covisibility with history must be <")
    ap.add_argument("--min_dist_AB", type=float, default=3.0, help="novel B: min geodesic A->B (3-leg)")
    ap.add_argument("--goal_jitter_pos", type=float, default=1.50,
                    help="revisit: uniform-disk RADIUS (m) around the anchor frame X; covis+heading gates cap realized offset")
    ap.add_argument("--goal_jitter_yaw", type=float, default=45.0, help="(unused; heading cone = head_max_deg)")
    # LingBot goal_append recomputes a FIXED W-frame window [m-W+1..m] around the match and injects it
    # at RoPE pos total_frames=m-W+1. For that window to have W real frames AND be disjoint from the
    # scale block [0,num_scale) (else RoPE position collision), need m-W+1 >= num_scale => m >= num_scale+W-1.
    ap.add_argument("--window", type=int, default=32, help="LingBot local sliding window W (must match precompute)")
    ap.add_argument("--num_scale", type=int, default=8, help="LingBot scale frames (full dense, injected)")
    ap.add_argument("--anchor_margin", type=int, default=None,
                    help="revisit anchor X >= this; default num_scale+window-1 (goal_append window disjoint from scale)")
    # recent (in-view) vs long-term (implicit-memory) revisit balance. long_term forces the matched frame
    # to sit OUTSIDE the current window (recall gap >= min_recall_gap), the memory-testing case.
    ap.add_argument("--long_term_frac", type=float, default=0.7,
                    help="fraction of revisits forced long-term (X outside current window); rest free (natural in-view mix)")
    ap.add_argument("--min_recall_gap", type=int, default=None,
                    help="long-term revisit: min (current - matched) frame gap; default = window")
    ap.add_argument("--goal_tries", type=int, default=40, help="rejection-sampling tries per goal")
    ap.add_argument("--covis_stride", type=int, default=4, help="history frame stride for covisibility")
    ap.add_argument("--covis_tol", type=float, default=0.30, help="depth-consistency tol for covisibility (m)")
    # multi-floor handling (MP3D)
    ap.add_argument("--floor_tol", type=float, default=0.80, help="max |y-floor_y| to count as same floor (m)")
    ap.add_argument("--floor_gap", type=float, default=1.00, help="min height gap between distinct floors (m)")
    ap.add_argument("--clearance_A", type=float, default=0.5, help="min obstacle clearance at A (U-turn room)")
    ap.add_argument("--clearance_B", type=float, default=0.3, help="min obstacle clearance at B (goal)")
    ap.add_argument("--step_m", type=float, default=0.10, help="corridor densify res for B sampling")
    ap.add_argument("--min_seen", type=int, default=4)
    ap.add_argument("--cam_h", type=float, default=0.5, help="camera height above floor navmesh (m)")
    ap.add_argument("--max_turn_deg", type=float, default=4.5, help="max heading change per frame")
    # pure-pursuit controller (measured from InternData-N1)
    ap.add_argument("--v", type=float, default=0.0376, help="speed m/frame")
    ap.add_argument("--lookahead", type=float, default=0.7)
    ap.add_argument("--r_min", type=float, default=0.40, help="min turning radius (m)")
    ap.add_argument("--v_min_frac", type=float, default=0.48,
                    help="speed floor as frac of cruise during sharp turns (N1-measured ~0.48)")
    ap.add_argument("--max_attempts", type=int, default=60); ap.add_argument("--debug", action="store_true")
    # navmesh inflation + ElasticBands clearance smoothing (from PythonRobotics / iPlanner)
    ap.add_argument("--agent_radius", type=float, default=0.30, help="navmesh inflation = robot radius (m)")
    ap.add_argument("--esdf_res", type=float, default=0.05, help="scene ESDF grid resolution (m)")
    ap.add_argument("--eb_iters", type=int, default=60)
    ap.add_argument("--eb_kc", type=float, default=0.5, help="ElasticBands contraction (smoothness)")
    ap.add_argument("--eb_kr", type=float, default=0.8, help="ElasticBands repulsion (clearance)")
    ap.add_argument("--eb_rho0", type=float, default=0.6, help="clearance influence radius (m)")
    ap.add_argument("--eb_step", type=float, default=0.04)
    args = ap.parse_args()
    if args.anchor_margin is None:
        args.anchor_margin = args.num_scale + args.window - 1     # goal_append window disjoint from scale block
    if args.min_recall_gap is None:
        args.min_recall_gap = args.window                        # "outside the current window"
    print(f"[main] window={args.window} num_scale={args.num_scale} -> anchor_margin={args.anchor_margin}; "
          f"long_term_frac={args.long_term_frac} min_recall_gap={args.min_recall_gap}")

    sim = make_sim(args.scene, args.navmesh, agent_radius=args.agent_radius)
    assert sim.pathfinder.is_loaded, "navmesh not loaded"
    lo, hi = sim.pathfinder.get_bounds()
    print(f"[main] navmesh height span {hi[1]-lo[1]:.1f} m (multi-floor if large); "
          f"each episode is confined to its A's floor (per-floor ESDF cached).")
    esdf_cache = {}                                          # floor bucket -> ESDF (built on demand)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    made = 0
    for ep in range(args.n * 6):
        if made >= args.n:
            break
        res = make_episode(sim, rng, args, made, esdf_cache)
        if res is None:
            continue
        rgbs, depths, poses, meta, goal_rgbs = res
        ep_dir = os.path.join(args.out, f"episode_{made:04d}")
        save_traj(ep_dir, rgbs, depths, poses, meta, goal_rgbs)
        gsum = " ".join(f"{g['name']}[{g['kind']}]covis{g['covis']:.2f}"
                        + (f"/head{g['head_off_deg']:.0f}deg" if g.get('head_off_deg') is not None else "")
                        + (f"/gap{g['recall_gap']}" if g.get('recall_gap') is not None else "")
                        for g in meta["goals"])
        print(f"[ep {made}] legs={meta['n_legs']} frames={meta['n_frames']} switches={meta['switches']} "
              f"geo start->A={meta['geo_startA']:.1f} goals: {gsum}")
        made += 1
    print(f"DONE: {made}/{args.n} episodes -> {args.out}")
    sim.close()


if __name__ == "__main__":
    main()
