import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="Multi-stop (start -> A -> B) image-goal NavDP benchmark")
parser.add_argument("--scene_dir", type=str, default="./assets/scenes/cluttered_hard")  # internscenes_home image-goal RGB renders black (2-camera bug on referenced USDs); cluttered works
parser.add_argument("--scene_index", type=int, default=0)
parser.add_argument("--scene_scale", type=float, default=1.0)   # 1.0 for cluttered, 0.01 for internscenes
parser.add_argument("--stop_threshold", type=float, default=-3.0)
parser.add_argument("--num_envs", type=int, default=1)           # state machine assumes 1 (matches NavDP eval scripts)
parser.add_argument("--num_episodes", type=int, default=20)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--port", type=int, default=8888)
# multi-stop knobs
parser.add_argument("--b_frac", type=float, default=0.5, help="target fraction of leg-A arc length at which to place B")
parser.add_argument("--b_min_dist", type=float, default=2.0, help="min Euclidean A->B distance so leg B is a real backtrack")
parser.add_argument("--b_min_start", type=float, default=1.0, help="min Euclidean start->B distance")
parser.add_argument("--arrive_thresh", type=float, default=1.0, help="distance to A counting as 'at A'")
parser.add_argument("--vel_thresh", type=float, default=0.5, help="speed below which the robot is 'stopped'")
parser.add_argument("--dwell_steps", type=int, default=3, help="consecutive stopped steps at A required to switch")
parser.add_argument("--max_near_steps", type=int, default=40, help="fallback: force switch after this many steps near A")
parser.add_argument("--success_thresh", type=float, default=1.5, help="scored arrival threshold (parity with NavDP)")
parser.add_argument("--light_type", type=str, default="none", choices=["none", "dome", "distant"], help="add a light: 'distant' (directional, shows shading) or 'dome' (uniform/flat)")
parser.add_argument("--light_intensity", type=float, default=2000.0, help="intensity of the added light")
parser.add_argument("--debug_rgb", action="store_true", help="dump raw main+goal camera RGB to raw_rgb.png each step to verify lighting")
parser.add_argument("--reset_memory", action="store_true", help="CH-reset arm: wipe NavDP memory at the A->B switch")
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import omni
import carb
import numpy as np
import imageio
import os
import csv
import json
import torch
import requests
import io
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from omni.isaac.core.prims import XFormPrimView
import omni.isaac.core.utils.numpy.rotations as rot_utils
import omni.isaac.lab.sim as sim_utils
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text, adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset, imagegoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller

# ----------------------------------------------------------------------------
# Phase-aware termination: leg A must NOT auto-terminate at A (we switch to B
# manually); only leg B terminates on reaching the (now retargeted) /Goal=B.
# env._ms_phase is an int array per env: 0 = LEG_A, 1 = LEG_B.
# ----------------------------------------------------------------------------
def multistop_arrival_check(env, robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot_asset = env.scene[robot_asset_cfg.name]
    robot_pos = robot_asset.data.root_pos_w
    goal_primview = XFormPrimView(prim_paths_expr="/World/envs/env_.*/Goal", name="xform_view")
    goal_pos = goal_primview.get_world_poses()[0]
    robot_vel = robot_asset.data.root_lin_vel_w
    distance = torch.square(robot_pos[:, 0:2] - goal_pos[:, 0:2]).sum(axis=1).sqrt()
    velocity = torch.abs(robot_vel).sum(axis=1)
    arrived = (distance < 1.0) & (velocity < 0.5)
    phase = torch.as_tensor(getattr(env, "_ms_phase", np.zeros(distance.shape[0])),
                            device=arrived.device)
    in_leg_b = phase > 0.5
    return arrived & in_leg_b

planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None

def planning_thread(env, camera_intrinsic):
    global mpc
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            with input_lock:
                if planning_input.current_goal is None or planning_input.current_image is None or planning_input.current_depth is None or planning_input.camera_pos is None or planning_input.camera_rot is None:
                    time.sleep(0.01)
                    continue
                goal = planning_input.current_goal.copy()
                image = planning_input.current_image.copy()
                depth = planning_input.current_depth.copy()
                camera_pos = planning_input.camera_pos.copy()
                camera_rot = planning_input.camera_rot.copy()
            with output_lock:
                planning_output.is_planning = True

            trajectory_points_camera, all_trajectories_camera, all_values_camera = imagegoal_step(goal, image, depth, port=args_cli.port)

            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    point_local = np.array([point[0], point[1], 0.0])
                    point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                    trajectory_points_world.append(point_world[:2])
                trajectory_points_world = np.array(trajectory_points_world)
                batch_optimal_points_world.append(trajectory_points_world)
                mpc = MPC_Controller(trajectory_points_world,
                                     desired_v=args_cli.speed,
                                     v_max=args_cli.speed,
                                     w_max=args_cli.speed)
            batch_optimal_points_world = np.array(batch_optimal_points_world)

            batch_all_points_world = []
            for idx in range(all_trajectories_camera.shape[0]):
                all_trajectories_world = []
                for traj_camera in all_trajectories_camera[idx]:
                    traj_world = []
                    for point in traj_camera:
                        point_local = np.array([point[0], point[1], 0.0])
                        point_world = camera_pos[idx] + camera_rot[idx] @ point_local
                        traj_world.append(point_world[:2])
                    all_trajectories_world.append(np.array(traj_world))
                batch_all_points_world.append(all_trajectories_world)
            batch_all_points_world = np.array(batch_all_points_world)

            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_values_camera = all_values_camera
                planning_output.is_planning = False
                planning_output.planning_error = None

        except Exception as e:
            print(f"Planning error: {e}")
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        time.sleep(0.1)

scene_path = os.path.join(args_cli.scene_dir, os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
usd_path, init_path = find_usd_path(scene_path, task='imagegoal')
scene_config = ImageNavSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.goal_marker = GOAL_CFG
scene_config.goal_camera = DINGO_ImageGoal_CameraCfg
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg
env_config = DingoImageNavCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {"init_point_path": init_path,
                                       'height_offset': 0.1,
                                       'camera_offset': 0.25,
                                       'robot_visible': False,
                                       'light_enabled': False}
# Disable the single-goal arrival terminator: leg A must not end at A. We swap
# in a phase-aware check that only fires once the robot is in leg B (goal = B).
env_config.terminations.arrive_goal.func = multistop_arrival_check
CAMERA_OFFSET = 0.25

env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)

# Optional diagnostic/added light. 'distant' is directional (creates shading -> tells us
# if geometry is actually in the RGB render); 'dome' is uniform (flat, washes out detail).
if args_cli.light_type == "dome":
    _l = sim_utils.DomeLightCfg(intensity=args_cli.light_intensity, color=(0.85, 0.85, 0.85))
    _l.func("/World/MultistopLight", _l)
    print(f"Added DOME light (intensity={args_cli.light_intensity}).")
elif args_cli.light_type == "distant":
    _l = sim_utils.DistantLightCfg(intensity=args_cli.light_intensity, color=(1.0, 1.0, 1.0))
    # tilt ~35 deg off vertical so walls/floors get different shading (quat wxyz about X)
    _ang = 0.6
    _quat = (float(np.cos(_ang / 2)), float(np.sin(_ang / 2)), 0.0, 0.0)
    _l.func("/World/MultistopLight", _l, orientation=_quat)
    print(f"Added DISTANT light (intensity={args_cli.light_intensity}, tilt={_ang}rad).")

# Per-env phase array, read by multistop_arrival_check via env._ms_phase.
ms_phase = np.zeros(args_cli.num_envs, dtype=np.float32)   # 0=LEG_A, 1=LEG_B
env.unwrapped._ms_phase = ms_phase

_, infos = env.reset()
PREHEAT_STEPS = 10
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)

camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

# Prim views for retargeting the goal marker + goal camera from A to B mid-episode.
goal_view = XFormPrimView(prim_paths_expr="/World/envs/env_.*/Goal", name="ms_goal_view")
goalcam_view = XFormPrimView(prim_paths_expr="/World/envs/env_.*/goal_cam", name="ms_goalcam_view")

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller = DifferentialController(name="simple_control",
                                    wheel_radius=DINGO_WHEEL_RADIUS,
                                    wheel_base=DINGO_WHEEL_BASE)
algo = navigator_reset(camera_intrinsic.cpu().numpy(), batch_size=scene_config.num_envs, stop_threshold=args_cli.stop_threshold, port=args_cli.port)

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
cold_pairs = []          # (A_x, A_y, B_x, B_y, B_yaw) for every valid A-success -> cold arm
arm_tag = "chreset" if args_cli.reset_memory else "ch"
save_dir = "./multistop_%s_%s_%s/%s/" % (arm_tag, algo, args_cli.scene_dir.split("/")[-1], scene_path.split("/")[-2])
os.makedirs(save_dir, exist_ok=True)
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4" % i, fps=10) for i in range(scene_config.num_envs)]

N = args_cli.num_envs
# ---- per-env multi-stop state ----
need_init = [True] * N                 # capture A_world / euclid_A at first step of each episode
tau = [[] for _ in range(N)]           # leg-A trajectory: list of (x, y, yaw) in world frame
A_world = np.zeros((N, 2))
B_world = np.zeros((N, 2))
B_yaw = np.zeros((N,))
euclid_A = np.zeros((N,))              # straight-line start->A
euclid_B = np.zeros((N,))              # straight-line A->B
legA_len = np.zeros((N,))
legB_len = np.zeros((N,))
reached_A = np.zeros((N,), dtype=bool)
b_valid = np.zeros((N,), dtype=bool)   # could a B satisfying the distance constraints be placed
b_forced = np.zeros((N,), dtype=bool)  # switch forced by max_near fallback (robot didn't cleanly stop)
dwell_count = np.zeros((N,), dtype=int)
near_count = np.zeros((N,), dtype=int)

def pick_B_from_tau(traj, A_xy):
    """Choose B on the recorded leg-A trajectory near args.b_frac of arc length,
    subject to min A->B and start->B distances. Returns (B_xy, B_yaw, valid, info)."""
    pts = np.asarray(traj, dtype=np.float64)
    if pts.shape[0] < 2:
        return None, None, False, "tau_too_short"
    xy = pts[:, 0:2]
    yaw = pts[:, 2]
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    if total < 1e-3:
        return None, None, False, "no_motion"
    start_xy = xy[0]
    d_A = np.linalg.norm(xy - A_xy[None, :], axis=1)
    d_start = np.linalg.norm(xy - start_xy[None, :], axis=1)
    ok = (d_A >= args_cli.b_min_dist) & (d_start >= args_cli.b_min_start)
    target = args_cli.b_frac * total
    if ok.any():
        cand = np.where(ok)[0]
        idx = cand[np.argmin(np.abs(arc[cand] - target))]
        return xy[idx].copy(), float(yaw[idx]), True, "ok"
    # fallback: farthest-from-A point (best effort), flag invalid
    idx = int(np.argmax(d_A))
    return xy[idx].copy(), float(yaw[idx]), False, "constraints_unmet"

def switch_to_B(i, robot_z, goal_z):
    """Retarget /Goal and goal_cam from A to the chosen B for env i."""
    bx, by = float(B_world[i, 0]), float(B_world[i, 1])
    byaw = float(B_yaw[i])
    dev = env.unwrapped.scene.articulations['robot'].data.root_pos_w.device
    # /Goal marker: same z as A's marker, yaw = B_yaw
    goal_pos = torch.tensor([[bx, by, goal_z]], dtype=torch.float32, device=dev)
    goal_rot = torch.tensor(rot_utils.euler_angles_to_quats(np.array([[0.0, 0.0, byaw]])),
                            dtype=torch.float32, device=dev)
    goal_view.set_world_poses(goal_pos, goal_rot, indices=[i])
    # goal_cam: at B, camera height, oriented to look forward (mirrors imagenav_reset:255)
    cam_pos = torch.tensor([[bx, by, robot_z + CAMERA_OFFSET]], dtype=torch.float32, device=dev)
    cam_rot = torch.tensor(rot_utils.euler_angles_to_quats(np.array([[np.pi / 2, 0.0, byaw - np.pi / 2]])),
                           dtype=torch.float32, device=dev)
    goalcam_view.set_world_poses(cam_pos, cam_rot, indices=[i])

# ---- warmup gate: don't start counting episodes until the planning server has
# returned its first trajectory. Otherwise the model-load / first-inference latency
# leaves the robot motionless and 'stuck' burns the first several episodes. ----
print("Warming up: waiting for first trajectory from planning server...")
_warm = 0
while _warm < 3000:
    with torch.inference_mode():
        wg = infos['observations']['goal_image'].cpu().numpy()[:, :, :, 0:3]
        wi = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
        wd = infos['observations']['depth'].cpu().numpy()[:, :, :]
        wcp = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        wcq = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()[:, [1, 2, 3, 0]]
        wcr = R.from_quat(wcq).as_matrix()
        with input_lock:
            planning_input.current_goal = wg.copy()
            planning_input.current_image = wi.copy()
            planning_input.current_depth = wd.copy()
            planning_input.camera_pos = wcp.copy()
            planning_input.camera_rot = wcr.copy()
        with output_lock:
            ready = planning_output.trajectory_points_world is not None
        if ready:
            break
        obs, rewards, dones, infos = env.step(torch.zeros((args_cli.num_envs, 2), device="cuda:0"))
        _warm += 1
print(f"Planning server ready after {_warm} warmup steps; starting episode 1.")
# NOTE: do NOT call env.reset() here. The warmup stepped under inference_mode, which
# marks sim tensors as inference tensors; reset()'s in-place pose write outside
# inference_mode would raise. The robot is still at episode-0's start (only a few
# stationary steps taken), and per-env state is already fresh, so we start directly.
for i in range(N):
    need_init[i] = True
    tau[i] = []
ms_phase[:] = 0.0
reached_A[:] = False
b_valid[:] = False
b_forced[:] = False
dwell_count[:] = 0
near_count[:] = 0
legA_len[:] = 0.0
legB_len[:] = 0.0

while simulation_app.is_running():
    with torch.inference_mode():
        goal_poses = infos['observations']['goal_pose'].cpu().numpy()[:, 0:2]
        goal_images = infos['observations']['goal_image'].cpu().numpy()[:, :, :, 0:3]
        images = infos['observations']['rgb'].cpu().numpy()[:, :, :, 0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:, :, :]
        if args_cli.debug_rgb:
            _raw = np.concatenate((images[0], goal_images[0]), axis=1).astype(np.uint8)  # [robot cam | goal cam]
            cv2.imwrite("raw_rgb.png", cv2.cvtColor(_raw, cv2.COLOR_RGB2BGR))
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:, [1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()
        cam_yaw = np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0])
        robot_z_all = env.unwrapped.scene.articulations['robot'].data.root_pos_w[:, 2].cpu().numpy()
        robot_speed = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[:, :2].norm(dim=1).cpu().numpy()
        dist_goal = np.linalg.norm(goal_poses, axis=1)   # distance to current /Goal (A or B)

        # ---- per-episode init: capture A and start->A distance ----
        for i in range(N):
            if need_init[i]:
                A_pos = goal_view.get_world_poses()[0][i, 0:2].cpu().numpy()
                A_world[i] = A_pos
                euclid_A[i] = dist_goal[i]
                need_init[i] = False

        # ---- record leg-A trajectory ----
        for i in range(N):
            if ms_phase[i] < 0.5:
                tau[i].append((float(camera_pos[i, 0]), float(camera_pos[i, 1]), float(cam_yaw[i])))

        with input_lock:
            planning_input.current_goal = goal_images.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()

        # ---- A-arrival detection -> switch to B ----
        for i in range(N):
            if ms_phase[i] >= 0.5:
                continue
            near = dist_goal[i] < args_cli.arrive_thresh
            stopped = robot_speed[i] < args_cli.vel_thresh
            near_count[i] = near_count[i] + 1 if near else 0
            dwell_count[i] = dwell_count[i] + 1 if (near and stopped) else 0
            do_switch = (dwell_count[i] >= args_cli.dwell_steps) or (near_count[i] >= args_cli.max_near_steps)
            if not do_switch:
                continue
            b_xy, b_yaw, valid, info = pick_B_from_tau(tau[i], A_world[i])
            if b_xy is None:
                # degenerate leg A (no motion): cannot place B, treat as A-success w/o B
                reached_A[i] = True
                b_valid[i] = False
                ms_phase[i] = 1.0
                print(f"[env {i}] A reached but B unplaceable ({info}); marking b_valid=False")
                continue
            B_world[i] = b_xy
            B_yaw[i] = b_yaw
            euclid_B[i] = float(np.linalg.norm(A_world[i] - b_xy))
            reached_A[i] = True
            b_valid[i] = valid
            b_forced[i] = (near_count[i] >= args_cli.max_near_steps) and (dwell_count[i] < args_cli.dwell_steps)
            goal_z = float(goal_view.get_world_poses()[0][i, 2].cpu().numpy())
            switch_to_B(i, robot_z_all[i], goal_z)
            ms_phase[i] = 1.0
            if args_cli.reset_memory:
                navigator_reset(env_id=i, port=args_cli.port)
            print(f"[env {i}] SWITCH A->B  arc-frac~{args_cli.b_frac}  d(A,B)={euclid_B[i]:.2f}  valid={valid}  forced={b_forced[i]}")

        # based on the current world trajectory
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()
        x0 = np.stack([camera_pos[:, 0], camera_pos[:, 1], np.arctan2(camera_rot[:, 1, 0], camera_rot[:, 0, 0]), [robot_vel], [robot_ang_vel]], axis=-1)

        current_trajectory = None
        current_all_trajectories = None
        current_all_values = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy()
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None

        if current_trajectory is not None:
            action_list = []
            for i in range(args_cli.num_envs):
                vis_image = vis_manager[i].visualize_trajectory(
                    np.concatenate((images[i], goal_images[i]), axis=1), depths[i][:, :, None], camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i]
                )
                if mpc is None:
                    continue
                opt_u_controls, opt_x_states = mpc.solve(x0[i, :3])
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)
                try:
                    leg = "B" if ms_phase[i] >= 0.5 else "A"
                    vis_image = draw_box_with_text(vis_image, 0, 0, 430, 50, "leg %s desired lin.:%.2f ang.:%.2f" % (leg, v, w))
                    vis_image = draw_box_with_text(vis_image, 0, 50, 430, 50, "actual lin.:%.2f ang.:%.2f" % (robot_vel, robot_ang_vel))
                    if current_all_values is not None:
                        vis_image = draw_box_with_text(vis_image, 0, 770, 430, 50, "critic max:%.2f min:%.2f" % (np.max(current_all_values[i]), np.min(current_all_values[i])))
                    vis_image = draw_box_with_text(vis_image, 0, 820, 430, 50, "goal rel:(%.2f, %.2f)" % (goal_poses[i][0], goal_poses[i][1]))
                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except Exception:
                    pass

            action = torch.as_tensor(np.stack(action_list, axis=0), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            step_len = (infos['observations']['policy'][:, 0] * env.unwrapped.step_dt).cpu().numpy()
            for i in range(N):
                if ms_phase[i] >= 0.5:
                    legB_len[i] += step_len[i]
                else:
                    legA_len[i] += step_len[i]
        else:
            action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            print("No trajectory available, using zero action")

        for i in range(N):
            if dones[i] == True:
                episode_num += 1
                in_leg_b = ms_phase[i] >= 0.5
                # Score from the PRE-step goal distance (dist_goal[i]): on `done` the
                # env has already auto-reset, so post-step infos point at the next A.
                final_dist = dist_goal[i]
                rA = bool(reached_A[i])
                rB = bool(in_leg_b and (final_dist < args_cli.success_thresh))
                spl_A = float(np.clip(euclid_A[i] / max(legA_len[i], 1e-6), 0, 1) * rA)
                spl_B = float(np.clip(euclid_B[i] / max(legB_len[i], 1e-6), 0, 1) * rB) if rA else 0.0
                evaluation_metrics.append({
                    'episode': episode_num,
                    'reached_A': float(rA),
                    'reached_B': float(rB) if rA else float('nan'),
                    'b_valid': float(b_valid[i]),
                    'b_forced': float(b_forced[i]),
                    'spl_A': spl_A,
                    'spl_B': spl_B if rA else float('nan'),
                    'euclid_A': float(euclid_A[i]),
                    'euclid_B': float(euclid_B[i]),
                    'legA_len': float(legA_len[i]),
                    'legB_len': float(legB_len[i]),
                    'arm': arm_tag,
                })
                write_metrics(evaluation_metrics, save_dir + "metric.csv")
                if rA and b_valid[i]:
                    cold_pairs.append([float(A_world[i, 0]), float(A_world[i, 1]),
                                       float(B_world[i, 0]), float(B_world[i, 1]), float(B_yaw[i])])
                    np.save(save_dir + "cold_pairs.npy", np.array(cold_pairs, dtype=np.float64))
                fps_writer[i].close()
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4" % episode_num, fps=10)
                # reset per-episode state for the next episode
                need_init[i] = True
                tau[i] = []
                ms_phase[i] = 0.0
                reached_A[i] = False
                b_valid[i] = False
                b_forced[i] = False
                dwell_count[i] = 0
                near_count[i] = 0
                legA_len[i] = 0.0
                legB_len[i] = 0.0
                euclid_B[i] = 0.0

        if episode_num > args_cli.num_episodes:
            break

stop_event.set()
with open(save_dir + "summary.json", "w") as f:
    n = len(evaluation_metrics)
    nA = sum(m['reached_A'] for m in evaluation_metrics)
    cond = [m for m in evaluation_metrics if m['reached_A'] > 0.5 and m['b_valid'] > 0.5]
    nB = sum(m['reached_B'] for m in cond)
    json.dump({
        'episodes': n,
        'SR_A': (nA / n) if n else 0.0,
        'SR_B_given_A': (nB / len(cond)) if cond else 0.0,
        'n_cond_B': len(cond),
        'mean_spl_A': float(np.nanmean([m['spl_A'] for m in evaluation_metrics])) if n else 0.0,
        'mean_spl_B': float(np.nanmean([m['spl_B'] for m in cond])) if cond else 0.0,
    }, f, indent=2)
print("Done. Metrics in", save_dir)
