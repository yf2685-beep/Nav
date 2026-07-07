"""Validate a Habitat renderer against InternData-N1 matterport3d_d435i shipped frames.

Goal: prove we can reproduce the shipped RGB by placing a Habitat camera at each
trajectory's recorded `action` pose. This pins down (a) the data-world -> Habitat-world
axis convention (data is Z-up, Habitat is Y-up) and (b) the camera optical convention.

Strategy: the crux is unknown coordinate conventions, so we SWEEP a small set of
candidate (world-convention, camera-convention) pairs on frame 0 and report which
minimises pixel error vs the shipped 0.jpg. Once the winner is known, rendering the
rest of the trajectory should line up.

Run (in the `habitat` env, headless w/ EGL):
    python validate_render.py \
        --scene /home/asus/Research/datasets/mp3d/.../17DRP5sb8fy/17DRP5sb8fy.glb \
        --traj  /home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i/17DRP5sb8fy/trajectory_10 \
        --frames 0,60,120,180 --out /home/asus/Research/Nav/memnav_viz/render_val
"""
import argparse, os, itertools
import numpy as np
import pandas as pd
from PIL import Image

W, H = 480, 270
FX, FY, CX, CY = 355.81464, 351.687, 240.0, 135.0
HFOV_DEG = float(np.degrees(2 * np.arctan(CX / FX)))  # ~68.0

# Candidate data-world(Z-up) -> Habitat-world(Y-up) rotations (3x3, applied to positions
# and to the pose rotation's world side). Habitat: +Y up, gravity -Y.
WORLD_CANDS = {
    # data (x,y,z) -> habitat (x, z, -y)   [-90 deg about X]
    "zup2yup_a": np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
    # data (x,y,z) -> habitat (x, z, y)
    "zup2yup_b": np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float),
    # data (x,y,z) -> habitat (-x, z, -y)
    "zup2yup_c": np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]], float),
    "identity":  np.eye(3),
}
# Candidate camera-optical conversions (data optical -> habitat optical: -Z forward, +Y up).
CAM_CANDS = {
    "cam_id": np.eye(3),
    "cam_flipYZ": np.diag([1.0, -1.0, -1.0]),   # OpenCV(+Z fwd,+Y down) -> Habitat(-Z fwd,+Y up)
    "cam_flipZ":  np.diag([1.0, 1.0, -1.0]),
}


def load_traj(traj_dir):
    pq = os.path.join(traj_dir, "data/chunk-000/episode_000000.parquet")
    df = pd.read_parquet(pq)
    poses = np.stack([np.array(a.tolist(), float).reshape(4, 4) for a in df["action"]])  # cam->world, Z-up
    ext = np.array(df["observation.camera_extrinsic"].iloc[0].tolist(), float).reshape(4, 4)
    rgb_dir = os.path.join(traj_dir, "videos/chunk-000/observation.images.rgb")
    return poses, ext, rgb_dir


def make_sim(scene_glb):
    import habitat_sim
    bk = habitat_sim.SimulatorConfiguration()
    bk.scene_id = scene_glb
    bk.enable_physics = False
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "color"; rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [H, W]; rgb.hfov = HFOV_DEG; rgb.position = [0, 0, 0]
    dep = habitat_sim.CameraSensorSpec()
    dep.uuid = "depth"; dep.sensor_type = habitat_sim.SensorType.DEPTH
    dep.resolution = [H, W]; dep.hfov = HFOV_DEG; dep.position = [0, 0, 0]
    agent = habitat_sim.agent.AgentConfiguration(); agent.sensor_specifications = [rgb, dep]
    return habitat_sim.Simulator(habitat_sim.Configuration(bk, [agent]))


def pose_to_agent_state(T_cw, Cw, Cc):
    """data cam->world (Z-up) 4x4 -> Habitat agent (pos, quat wxyz) under conventions Cw, Cc."""
    import quaternion  # numpy-quaternion, ships with habitat
    R = T_cw[:3, :3]; t = T_cw[:3, 3]
    R_h = Cw @ R @ Cc
    t_h = Cw @ t
    q = quaternion.from_rotation_matrix(R_h)
    return t_h, q


def render(sim, pos, quat):
    import habitat_sim
    st = habitat_sim.agent.AgentState(); st.position = pos; st.rotation = quat
    sim.get_agent(0).set_state(st)
    obs = sim.get_sensor_observations()
    return obs["color"][..., :3], obs["depth"]


def l1(a, b):
    return float(np.abs(a.astype(float) - b.astype(float)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames", default="0")
    ap.add_argument("--out", default="./render_val")
    ap.add_argument("--sweep", action="store_true", help="sweep conventions on frame 0")
    ap.add_argument("--world", default="zup2yup_a")
    ap.add_argument("--cam", default="cam_flipYZ")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    poses, ext, rgb_dir = load_traj(args.traj)
    frames = [int(x) for x in args.frames.split(",")]
    sim = make_sim(args.scene)

    if args.sweep:
        gt0 = np.array(Image.open(os.path.join(rgb_dir, "0.jpg")))[..., :3]
        results = []
        for wn, cn in itertools.product(WORLD_CANDS, CAM_CANDS):
            pos, q = pose_to_agent_state(poses[0], WORLD_CANDS[wn], CAM_CANDS[cn])
            rgb, _ = render(sim, pos, q)
            e = l1(rgb, gt0)
            results.append((e, wn, cn))
            Image.fromarray(rgb).save(os.path.join(args.out, f"sweep_{wn}_{cn}.png"))
        results.sort()
        print("=== convention sweep on frame 0 (lower L1 = better) ===")
        for e, wn, cn in results:
            print(f"  L1={e:7.2f}  world={wn:12s} cam={cn}")
        Image.fromarray(gt0).save(os.path.join(args.out, "gt_0.png"))
        print("best:", results[0][1:], "-> re-run without --sweep using those")
        return

    Cw, Cc = WORLD_CANDS[args.world], CAM_CANDS[args.cam]
    for i in frames:
        gt = np.array(Image.open(os.path.join(rgb_dir, f"{i}.jpg")))[..., :3]
        pos, q = pose_to_agent_state(poses[i], Cw, Cc)
        rgb, depth = render(sim, pos, q)
        Image.fromarray(np.concatenate([gt, rgb], axis=1)).save(
            os.path.join(args.out, f"cmp_{i}.png"))
        print(f"frame {i}: L1={l1(rgb, gt):.2f}  (left=shipped, right=rendered)")


if __name__ == "__main__":
    main()
