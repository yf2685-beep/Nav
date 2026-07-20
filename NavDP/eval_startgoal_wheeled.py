import argparse
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="A script to run a car control simulation")
parser.add_argument(
    "--scene_dir", type=str, default="./asset_scenes/cluttered_easy")
parser.add_argument(
    "--scene_index", type=int, default=8)
parser.add_argument(
    "--scene_scale", type=float, default=1.0)
parser.add_argument(
    "--stop_threshold", type=float, default=-3.0)
parser.add_argument(
    "--num_envs", type=int, default=1)
parser.add_argument(
    "--num_episodes", type=int, default=100)
parser.add_argument(
    "--speed", type=float, default=0.5)
parser.add_argument(
    "--port", type=int, default=19999)
parser.add_argument(
    "--per_step_goal", action="store_true",
    help="refresh goal each planning step. Default is frozen (matches paper/HF release). "
         "Enable only for ckpts trained with per-step goal supervision.")
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import omni
import cv2
import carb
import numpy as np
import imageio
import os
import csv
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pxr import Usd, Sdf
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper
from wheeled_robots.controllers.differential_controller import DifferentialController
import torchvision.transforms as F
import time
import threading

from utils_tasks.basic_utils import PlanningInput, PlanningOutput, find_usd_path, write_metrics, draw_box_with_text,adjust_usd_scale
from configs.robots import *
from configs.scenes import *
from configs.tasks import *
from utils_tasks.client_utils import navigator_reset,pointgoal_step
from utils_tasks.visualization_utils import VisualizationManager
from utils_tasks.tracking_utils import MPC_Controller

planning_input = PlanningInput()
planning_output = PlanningOutput()
input_lock = threading.Lock()
output_lock = threading.Lock()
stop_event = threading.Event()
vis_manager = [VisualizationManager(history_size=5) for i in range(args_cli.num_envs)]
mpc = None

# ============================================================
# Professor MVP: optionally replace IsaacSim GT pose with LingBot pose
# Env vars:
#   EVAL_USE_LINGBOT_POSE=1            enable replacement
#   EVAL_LINGBOT_CKPT=<path>           LingBot ckpt (default: /home/nyuair/data-001/lingbot-map-ckpt/lingbot-map.pt)
#   EVAL_LINGBOT_WINDOW=32             sliding window size
# Logs LingBot vs sim pose per step to enable offline drift analysis.
# ============================================================
USE_LINGBOT_POSE = os.environ.get('EVAL_USE_LINGBOT_POSE', '0') == '1'
USE_PGO          = os.environ.get('EVAL_USE_PGO', '0') == '1'
LINGBOT_POSE_ESTIMATOR = None
PGO_CORRECTOR    = None
SIM_TO_LINGBOT_OFFSET = None  # 4x4 rigid transform mapping LingBot frame -> sim frame
SIM_LINGBOT_POSE_LOG = []     # list of (step, sim_pose_4x4, lingbot_pose_4x4_in_sim_frame)
LINGBOT_STEP = 0
if USE_LINGBOT_POSE:
    print('[professor] EVAL_USE_LINGBOT_POSE=1 -- loading LingBot estimator')
    from lingbot_pose_estimator import LingBotPoseEstimator, PGOCorrector
    LINGBOT_POSE_ESTIMATOR = LingBotPoseEstimator(
        ckpt=os.environ.get('EVAL_LINGBOT_CKPT', '/home/nyuair/data-001/lingbot-map-ckpt/lingbot-map.pt'),
        window_size=int(os.environ.get('EVAL_LINGBOT_WINDOW', '32')),
    )
    if USE_PGO:
        print('[professor] EVAL_USE_PGO=1 -- loop closure + snap correction enabled')
        PGO_CORRECTOR = PGOCorrector(
            sim_cosine_threshold=float(os.environ.get('EVAL_PGO_COSINE', '0.992')),
            min_drift_meters=float(os.environ.get('EVAL_PGO_MIN_DRIFT', '0.3')),
            min_frames_gap=int(os.environ.get('EVAL_PGO_MIN_GAP', '25')),
        )

def planning_thread(env, camera_intrinsic):
    global mpc
    """Thread function that continuously plans trajectories"""
    while not stop_event.is_set():
        try:
            # Get latest observations from shared state
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
            
            # Start timing planning
            planning_start = time.time()
            trajectory_points_camera, all_trajectories_camera, all_values_camera,sub_pointgoal_pd = pointgoal_step(goal, image, depth,port=args_cli.port)
            # Transform trajectory from camera frame to world frame
            batch_optimal_points_world = []
            for idx in range(trajectory_points_camera.shape[0]):
                trajectory_points_world = []
                for i, point in enumerate(trajectory_points_camera[idx]):
                    if i < 0:
                        continue
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
                # Transform all trajectories
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

            # Update shared state
            with output_lock:
                planning_output.trajectory_points_world = batch_optimal_points_world
                planning_output.all_trajectories_world = batch_all_points_world
                planning_output.all_values_camera = all_values_camera
                planning_output.sub_pointgoal_pd = sub_pointgoal_pd
                planning_output.is_planning = False
                planning_output.planning_error = None
            
            # Print planning timing
            planning_time = time.time() - planning_start
            print(f"Planning time: {planning_time:.3f}s, Goal shape: {goal.shape}, first goal: {goal[0].tolist()}")
                
        except Exception as e:
            import traceback
            print(f"Planning error: {e}")
            traceback.print_exc()
            with output_lock:
                planning_output.is_planning = False
                planning_output.planning_error = str(e)
        # Small sleep to prevent CPU overload
        time.sleep(0.1)

scene_path = os.path.join(args_cli.scene_dir,os.listdir(args_cli.scene_dir)[args_cli.scene_index]) + "/"
usd_path,init_path = find_usd_path(scene_path,task='pointgoal')
scene_config = PointNavSceneCfg()
scene_config.num_envs = args_cli.num_envs
scene_config.env_spacing = 0.0
scene_config.terrain = BENCH_TERRAIN_CFG
scene_config.terrain.usd_path = usd_path
scene_config.goal = GOAL_CFG
scene_config.robot = DINGO_CFG
scene_config.camera_sensor = DINGO_CameraCfg
scene_config.contact_sensor = DINGO_ContactCfg
env_config = DingoPointNavCfg()
env_config.scene = scene_config
env_config.events.reset_pose.params = {"init_point_path":init_path, 
                                       'height_offset':0.1,
                                       'robot_visible': False,
                                       'light_enabled': False}
env = ManagerBasedRLEnv(env_config)
env = RslRlVecEnvWrapper(env)
adjust_usd_scale(scale=args_cli.scene_scale)
_,infos = env.reset()
# warm-up
PREHEAT_STEPS = 50
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)
    
camera_intrinsic = env.unwrapped.scene.sensors['camera_sensor'].data.intrinsic_matrices[0]

planning_thread_obj = threading.Thread(target=planning_thread, args=(env, camera_intrinsic))
planning_thread_obj.daemon = True
planning_thread_obj.start()

controller = DifferentialController(name="simple_control", 
                                    wheel_radius=DINGO_WHEEL_RADIUS,
                                    wheel_base=DINGO_WHEEL_BASE)
algo = navigator_reset(camera_intrinsic.cpu().numpy(),batch_size=scene_config.num_envs,stop_threshold=args_cli.stop_threshold,port=args_cli.port)

episode_num = args_cli.num_envs - 1
evaluation_metrics = []
save_dir = "./startgoal_%s_%s/%s/"%(algo,args_cli.scene_dir.split("/")[-1],scene_path.split("/")[-2])
os.makedirs(save_dir,exist_ok=True)

euclidean = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))
fps_writer = [imageio.get_writer(save_dir + "fps_%d.mp4"%i, fps=10) for i in range(scene_config.num_envs)]

trajectory_length = np.zeros((scene_config.num_envs))
start_goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
# warm-up
PREHEAT_STEPS = 50
for _ in range(PREHEAT_STEPS):
    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
    obs, rewards, dones, infos = env.step(action)
sub_pointgoal_pd = np.zeros((args_cli.num_envs,3))
while simulation_app.is_running():
    with torch.inference_mode():
        goals = infos['observations']['goal_pose'].cpu().numpy()[:,0:2]
        goals_gt = goals.copy()   # preserve sim-GT goal_robot for honest success_flag
        images = infos['observations']['rgb'].cpu().numpy()[:,:,:,0:3]
        depths = infos['observations']['depth'].cpu().numpy()[:,:,:]
        # get all camera poses
        camera_pos = env.unwrapped.scene.sensors['camera_sensor'].data.pos_w.cpu().numpy()
        camera_rot_quat = env.unwrapped.scene.sensors['camera_sensor'].data.quat_w_world.cpu().numpy()
        camera_rot_quat = camera_rot_quat[:,[1, 2, 3, 0]]
        camera_rot = R.from_quat(camera_rot_quat).as_matrix()

        # === Professor MVP: override sim GT pose with LingBot pose ===
        if USE_LINGBOT_POSE:
            sim_pos_gt   = camera_pos.copy()   # (N,3) GT for logging only
            sim_rot_gt   = camera_rot.copy()   # (N,3,3)
            sim_goal_gt  = goals.copy()        # (N,2) robot-frame goal from sim using GT pose
            # Build sim's 4x4 (env 0 only -- single-env eval)
            sim_pose_4x4 = np.eye(4, dtype=np.float32)
            sim_pose_4x4[:3, :3] = sim_rot_gt[0]
            sim_pose_4x4[:3, 3]  = sim_pos_gt[0]
            # Run LingBot on current RGB (env 0)
            lingbot_pose_4x4 = LINGBOT_POSE_ESTIMATOR.estimate(images[0])  # camera->LingBot world
            # At step 0, lock the rigid alignment LingBot world -> sim world so
            # the FIRST LingBot pose maps to the FIRST sim pose. From then on,
            # LingBot drift is exactly what the policy sees as "pose error".
            if SIM_TO_LINGBOT_OFFSET is None:
                # T_offset such that sim_pose_4x4 == T_offset @ lingbot_pose_4x4
                SIM_TO_LINGBOT_OFFSET = sim_pose_4x4 @ np.linalg.inv(lingbot_pose_4x4)
            aligned_pose = SIM_TO_LINGBOT_OFFSET @ lingbot_pose_4x4  # in sim-world coords
            # === Loop closure + snap correction (Phase 2) ===
            if USE_PGO and PGO_CORRECTOR is not None:
                desc = LINGBOT_POSE_ESTIMATOR.get_descriptor()
                triggered, snap_T = PGO_CORRECTOR.detect_and_correct(
                    LINGBOT_STEP, aligned_pose.astype(np.float32), desc,
                )
                if triggered:
                    # Apply the snap to alignment offset so subsequent frames inherit
                    # the correction.  After this: aligned_pose_new = snap_T @ old aligned.
                    SIM_TO_LINGBOT_OFFSET = snap_T @ SIM_TO_LINGBOT_OFFSET
                    aligned_pose = snap_T @ aligned_pose
                    print(f'[pgo] step={LINGBOT_STEP} CLOSURE triggered, snap dx='
                          f'{snap_T[0,3]:+.3f} dy={snap_T[1,3]:+.3f}')
                # Register the (possibly corrected) frame for future closure matches.
                PGO_CORRECTOR.add(LINGBOT_STEP, aligned_pose.astype(np.float32), desc)
            lb_pos = aligned_pose[:3, 3]
            lb_rot = aligned_pose[:3, :3]
            # Re-derive goal in LingBot's believed robot frame.
            # goal_world (constant per episode) = sim_pose_4x4 @ [sim_goal_gt, 0, 1]
            goal_robot_h = np.array([sim_goal_gt[0, 0], sim_goal_gt[0, 1], 0.0, 1.0], dtype=np.float32)
            goal_world   = sim_pose_4x4 @ goal_robot_h
            goal_lb_h    = np.linalg.inv(aligned_pose) @ goal_world
            goals = goal_lb_h[None, :2].astype(np.float32)
            # Override pose used downstream (planning + MPC world-frame transform).
            camera_pos = lb_pos[None]
            camera_rot = lb_rot[None]
            # Log every 10 steps for drift trace.
            if LINGBOT_STEP % 10 == 0:
                pos_err = float(np.linalg.norm(sim_pos_gt[0] - lb_pos))
                yaw_sim = float(np.arctan2(sim_rot_gt[0,1,0], sim_rot_gt[0,0,0]))
                yaw_lb  = float(np.arctan2(lb_rot[1,0], lb_rot[0,0]))
                yaw_err = float(((yaw_sim - yaw_lb + np.pi) % (2*np.pi)) - np.pi)
                print(f'[lb] step={LINGBOT_STEP:4d} '
                      f'sim=({sim_pos_gt[0,0]:+.2f},{sim_pos_gt[0,1]:+.2f}) '
                      f'lb=({lb_pos[0]:+.2f},{lb_pos[1]:+.2f}) '
                      f'|err|={pos_err:.3f}m yaw_err={yaw_err:+.3f}rad '
                      f'goal_sim=({sim_goal_gt[0,0]:+.2f},{sim_goal_gt[0,1]:+.2f}) '
                      f'goal_lb=({goals[0,0]:+.2f},{goals[0,1]:+.2f})')
            SIM_LINGBOT_POSE_LOG.append({
                'step': LINGBOT_STEP,
                'sim_pos': sim_pos_gt[0].tolist(),
                'lb_pos':  lb_pos.tolist(),
                'sim_goal': sim_goal_gt[0].tolist(),
                'lb_goal':  goals[0].tolist(),
            })
            LINGBOT_STEP += 1

        with input_lock:
            # Default (paper / HF release): freeze the goal at episode start so the
            # planner sees the same start-frame goal vector for the whole episode.
            # With --per_step_goal, refresh each step using the current robot-frame
            # goal_pose (recomputed by oracle_imu_pose_data in wheeled_task.py).
            if args_cli.per_step_goal:
                planning_input.current_goal = goals.copy()
            else:
                if planning_input.current_goal is None:
                    start_goals = goals.copy()
                planning_input.current_goal = start_goals.copy()
            planning_input.current_image = images.copy()
            planning_input.current_depth = depths.copy()
            planning_input.camera_pos = camera_pos.copy()
            planning_input.camera_rot = camera_rot.copy()

        # based on the current world trajectory 
        robot_vel = env.unwrapped.scene.articulations['robot'].data.root_lin_vel_w[0, :2].norm().cpu().numpy()
        robot_ang_vel = env.unwrapped.scene.articulations['robot'].data.root_ang_vel_w[0, 2].cpu().numpy()

        x0 = np.stack([camera_pos[:,0], camera_pos[:,1], np.arctan2(camera_rot[:,1,0], camera_rot[:,0,0]), [robot_vel], [robot_ang_vel]],axis=-1)
        current_trajectory = None
        current_all_trajectories = None
        current_all_values = None
        with output_lock:
            if planning_output.trajectory_points_world is not None:
                current_trajectory = planning_output.trajectory_points_world.copy() if planning_output.trajectory_points_world is not None else None
                current_all_trajectories = planning_output.all_trajectories_world.copy() if planning_output.all_trajectories_world is not None else None
                current_all_values = planning_output.all_values_camera.copy() if planning_output.all_values_camera is not None else None
                sub_pointgoal_pd = planning_output.sub_pointgoal_pd.copy() if planning_output.sub_pointgoal_pd is not None else None
        
        if current_trajectory is not None:
            control_start = time.time()
            action_list = []
            for i in range(args_cli.num_envs):
                vis_image = vis_manager[i].visualize_trajectory(
                    images[i], depths[i][:,:,None], camera_intrinsic.cpu().numpy(),
                    current_trajectory[i],
                    robot_pose=x0[i],
                    all_trajectories_points=current_all_trajectories[i],
                    all_trajectories_values=current_all_values[i]
                )
                if mpc is None:
                    continue
                t0 = time.time()
                opt_u_controls, opt_x_states = mpc.solve(x0[i,:3])
                print(f"solve mpc cost {time.time() - t0}")
                v, w = opt_u_controls[1, 0], opt_u_controls[1, 1]
                action = torch.tensor([v, w], device="cuda:0")
                action_cpu = action.cpu().numpy()
                joint_velocities = controller.forward(action_cpu).joint_velocities
                action_list.append(joint_velocities)
                
                try:
                    vis_image = draw_box_with_text(vis_image,0,770,430,50,"gt point goal:(%.2f, %.2f)"%(goals[i][0],goals[i][1]))
                    if sub_pointgoal_pd is not None:
                        vis_image = draw_box_with_text(vis_image,0,820,430,50,"pd point goal:(%.2f, %.2f)"%(sub_pointgoal_pd[i][0],sub_pointgoal_pd[i][1]))
                    cv2.imwrite(f"frame_test.png", cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
                    fps_writer[i].append_data(vis_image)
                except:
                    pass
                
            action = torch.as_tensor(np.stack(action_list, axis=0),device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            # Get actual joint velocities from Isaac Sim
            actual_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel[0, :2].cpu().numpy()
            desired_joint_velocities = env.unwrapped.scene.articulations['robot'].data.joint_vel_target[0, :2].cpu().numpy()
            trajectory_length += (infos['observations']['policy'][:,0] * env.unwrapped.step_dt).cpu().numpy()
        else:
            action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
            obs, rewards, dones, infos = env.step(action)
            print("No trajectory available, using zero action")
        
        
        for i in range(args_cli.num_envs):
            if dones[i] == True:
                with input_lock:
                    le = np.sqrt(np.square(goals[i][0] - sub_pointgoal_pd[i][0]) + np.square(goals[i][1] - sub_pointgoal_pd[i][1])) if sub_pointgoal_pd is not None else 0.0
                    planning_input.current_goal = None
                    planning_input.current_image = None
                    planning_input.current_depth = None
                    planning_input.camera_pos = None
                    planning_input.camera_rot = None
                with output_lock:
                    planning_output.trajectory_points_world = None
                    planning_output.all_trajectories_world = None
                    planning_output.all_values_camera = None
                    planning_output.sub_pointgoal_pd = None

                episode_num += 1
                navigator_reset(env_id=i,port=args_cli.port)
                # Use TRUE goal-in-robot-frame (sim GT) for success, NOT the
                # LingBot-overridden value -- otherwise drift biases the metric.
                _eval_goal = goals_gt[i] if USE_LINGBOT_POSE else goals[i]
                success_flag = (np.sqrt(np.square(_eval_goal).sum())<1.5).astype(np.float32)
                # Reset LingBot streaming state so next episode starts fresh.
                if USE_LINGBOT_POSE and LINGBOT_POSE_ESTIMATOR is not None:
                    LINGBOT_POSE_ESTIMATOR.reset()
                    if PGO_CORRECTOR is not None:
                        PGO_CORRECTOR.reset()
                    SIM_TO_LINGBOT_OFFSET = None
                    LINGBOT_STEP = 0
                fps_writer[i].close()
                evaluation_metrics.append({'success':success_flag,
                                        'spl': np.clip(euclidean[i] / trajectory_length[i],0,1) * success_flag,
                                        'ne': np.sqrt(np.square(goals[i]).sum()),
                                        'le': le,
                                        'distance':euclidean[i]})
                write_metrics(evaluation_metrics,save_dir+"metric.csv")
                euclidean[i] = np.sqrt(np.square(infos['observations']['goal_pose'].cpu().numpy()[:,0:2]).sum(axis=-1))[i]
                fps_writer[i] = imageio.get_writer(save_dir + "fps_%d.mp4"%episode_num, fps=10)
                trajectory_length[i] = 0.0
        
                # warm-up
                PREHEAT_STEPS = 10
                for _ in range(PREHEAT_STEPS):
                    action = torch.zeros((args_cli.num_envs, 2), device="cuda:0")
                    obs, rewards, dones, infos = env.step(action)
        
        if episode_num > args_cli.num_episodes:
            break
       
                
   

        
