"""Collision critic — geometry-grounded safety scorer for candidate trajectories.

Stage 5 of the RGB-only + LingBot + subgoal plan. Depth was removed from the
navigation *policy* (Stage 1) but is deliberately kept in the dataloader so it
can feed THIS critic: depth → local point cloud → geometric collision labels.

This module is intentionally self-contained (numpy geometry + a small torch net)
so it can be:
  * used at INFERENCE for candidate-trajectory reranking (Stage 6), and
  * trained as a SEPARATE network on auto-generated labels (Stage 9),
  * later folded into the policy loss as a safety penalty (Stage 7).

Frames
------
We work in the robot ground-plane frame: x = forward, y = left (right-handed,
z up). Camera-frame points (x right, y down, z forward) are converted with
``camera_points_to_robot_ground``. Trajectory waypoints (x, y, [θ]) are already
in this frame in LoGoPlanner.
"""

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Geometry: depth -> point cloud, and geometric collision labelling
# ---------------------------------------------------------------------------
def depth_to_pointcloud(depth_hw1, K, max_depth=5.0, min_depth=0.1, stride=4):
    """Unproject a (H, W, 1) metric depth map to (N, 3) CAMERA-frame points.

    Camera convention: x right, y down, z forward (OpenCV). Invalid / out-of-range
    depths are dropped. ``stride`` subsamples pixels for speed.
    """
    depth_hw1 = np.asarray(depth_hw1, np.float32)
    if depth_hw1.ndim == 3:
        d = depth_hw1[..., 0]
    else:
        d = depth_hw1
    H, W = d.shape[:2]
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing='ij')
    dd = d[::stride, ::stride]
    valid = (dd > min_depth) & (dd < max_depth)
    us, vs, dd = us[valid], vs[valid], dd[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (us - cx) / fx * dd
    y = (vs - cy) / fy * dd
    z = dd
    return np.stack([x, y, z], axis=-1).astype(np.float32)  # (N, 3)


def camera_points_to_robot_ground(points_cam):
    """Camera (x right, y down, z fwd) -> robot ground plane (x fwd, y left), 2D.

    Drops the vertical axis (camera y); obstacles are checked in the ground plane.
    """
    points_cam = np.asarray(points_cam, np.float32)
    x_fwd = points_cam[:, 2]
    y_left = -points_cam[:, 0]
    return np.stack([x_fwd, y_left], axis=-1)  # (N, 2)


def label_trajectory_collision(traj_xy, obstacle_xy, footprint_radius=0.3,
                               safety_dist=0.3):
    """Geometric collision label for one candidate trajectory.

    Models the robot footprint as a disc of radius ``footprint_radius`` swept
    along the waypoints. A waypoint collides if the nearest obstacle point is
    inside the footprint; it is "at risk" within ``safety_dist`` beyond it.

    Args:
        traj_xy:      (T, 2) waypoints in robot ground frame (x fwd, y left).
        obstacle_xy:  (N, 2) obstacle points in the same frame.
        footprint_radius: robot body radius (m).
        safety_dist:  buffer beyond the body that still counts as risky (m).

    Returns dict:
        risk:          float in [0, 1]   (1 = collision / penetrating)
        is_collision:  bool              (any waypoint penetrates the body)
        worst_idx:     int               (most dangerous waypoint, -1 if no obs)
        min_clearance: float             (signed body-to-obstacle clearance, m)
        clearance:     (T,) per-waypoint signed clearance
    """
    traj_xy = np.asarray(traj_xy, np.float32)
    obstacle_xy = np.asarray(obstacle_xy, np.float32)
    T = traj_xy.shape[0]
    if obstacle_xy.shape[0] == 0:
        return dict(risk=0.0, is_collision=False, worst_idx=-1,
                    min_clearance=float('inf'),
                    clearance=np.full(T, np.inf, np.float32))
    # nearest obstacle distance per waypoint
    d = np.linalg.norm(traj_xy[:, None, :] - obstacle_xy[None, :, :], axis=-1).min(axis=1)
    clearance = d - footprint_radius                 # signed: <0 = body penetrates
    min_clearance = float(clearance.min())
    worst_idx = int(clearance.argmin())
    is_collision = bool(min_clearance < 0.0)
    # risk ramps 0 (clear beyond safety_dist) -> 1 (touching the body), and is
    # pinned at 1 on actual penetration.
    risk = float(np.clip(1.0 - max(min_clearance, 0.0) / max(safety_dist, 1e-6), 0.0, 1.0))
    if is_collision:
        risk = 1.0
    return dict(risk=risk, is_collision=is_collision, worst_idx=worst_idx,
                min_clearance=min_clearance, clearance=clearance.astype(np.float32))


def label_trajectory_collision_per_waypoint(traj_xy, obstacle_xy, footprint_radius=0.3,
                                            safety_dist=0.3):
    """Per-waypoint risk in [0,1] (for per-waypoint critic supervision)."""
    info = label_trajectory_collision(traj_xy, obstacle_xy, footprint_radius, safety_dist)
    clr = info['clearance']
    wp_risk = np.clip(1.0 - np.clip(clr, 0.0, None) / max(safety_dist, 1e-6), 0.0, 1.0)
    wp_risk[clr < 0.0] = 1.0
    return wp_risk.astype(np.float32), info


# ---------------------------------------------------------------------------
# Stage 6: inference-time reranking of candidate trajectories
# ---------------------------------------------------------------------------
def rerank_trajectories(trajectories, obstacle_xy, goal_xy,
                        footprint_radius=0.3, safety_dist=0.3,
                        collision_threshold=0.5, safety_weight=1.0,
                        learned_values=None):
    """Pick the best of K candidate trajectories: safe AND close to the subgoal.

    The navigation policy proposes K trajectories (the diffusion samples); this
    filters out the geometrically-unsafe ones and, among the survivors, picks the
    one whose endpoint is closest to the subgoal. If ALL candidates are unsafe it
    falls back to the lowest-risk one and flags a stop/replan.

    Args:
        trajectories:  (K, T, >=2) candidate waypoints, robot ground frame (x fwd, y left).
        obstacle_xy:   (N, 2) obstacle points (robot ground frame); empty → no collision.
        goal_xy:       (2,) subgoal position in robot frame.
        footprint_radius, safety_dist: robot body radius + safety buffer (m).
        collision_threshold: candidates with risk >= this are filtered out.
        safety_weight: how strongly risk discounts nav quality in the final score.
        learned_values: optional (K,) learned-critic scores to blend in (added to score).

    Returns dict:
        best_idx, selected (T,>=2), risks (K,), qualities (K,), is_safe (K,),
        all_unsafe (bool), stop (bool).
    """
    trajectories = np.asarray(trajectories, np.float32)
    K = trajectories.shape[0]
    goal_xy = np.asarray(goal_xy, np.float32)[:2]
    risks = np.zeros(K, np.float32)
    quals = np.zeros(K, np.float32)
    for k in range(K):
        info = label_trajectory_collision(trajectories[k, :, :2], obstacle_xy,
                                          footprint_radius, safety_dist)
        risks[k] = info['risk']
        endpoint = trajectories[k, -1, :2]
        quals[k] = -float(np.linalg.norm(endpoint - goal_xy))  # higher = closer to goal

    is_safe = risks < collision_threshold
    score = quals - safety_weight * risks
    if learned_values is not None:
        score = score + np.asarray(learned_values, np.float32)

    all_unsafe = not bool(is_safe.any())
    if all_unsafe:
        best_idx = int(np.argmin(risks))         # least dangerous fallback
        stop = bool(risks[best_idx] >= 1.0)      # truly colliding → stop/replan
    else:
        masked = np.where(is_safe, score, -1e9)
        best_idx = int(np.argmax(masked))
        stop = False

    return dict(best_idx=best_idx, selected=trajectories[best_idx],
                risks=risks, qualities=quals, is_safe=is_safe,
                all_unsafe=all_unsafe, stop=stop)


def obstacles_from_depth(depth_hw1, K, footprint_height=(-0.2, 1.5),
                         max_depth=5.0, min_depth=0.1, stride=4):
    """Depth map → obstacle points in the robot ground frame (x fwd, y left).

    Optionally clips to a vertical band so the floor / ceiling aren't treated as
    obstacles (camera y is down → keep points within [−ceil, −floor]).
    """
    pc_cam = depth_to_pointcloud(depth_hw1, K, max_depth=max_depth,
                                 min_depth=min_depth, stride=stride)
    if pc_cam.shape[0] == 0:
        return np.zeros((0, 2), np.float32)
    if footprint_height is not None:
        # camera y is DOWN, so height above the camera = −camera_y. Keep only points
        # in the robot's vertical band so floor/ceiling aren't treated as obstacles.
        lo, hi = footprint_height
        height = -pc_cam[:, 1]
        pc_cam = pc_cam[(height > lo) & (height < hi)]
    return camera_points_to_robot_ground(pc_cam)


# ---------------------------------------------------------------------------
# Stage 7: differentiable safety penalty for policy training
# ---------------------------------------------------------------------------
def differentiable_safety_loss(pred_traj_xy, obstacle_xy, obstacle_mask=None,
                               footprint_radius=0.3, safety_margin=0.3):
    """Differentiable geometric safety penalty (torch), for the policy loss.

    Penalises each predicted waypoint for coming within (footprint + margin) of an
    obstacle point. Gradients flow back through ``pred_traj_xy`` into the policy
    (the obstacle cloud is treated as a constant). Use a SMALL weight and monitor
    it against the imitation loss so safety never dominates.

    Args:
        pred_traj_xy: (B, T, 2) predicted waypoints (robot frame), differentiable.
        obstacle_xy:  (B, P, 2) obstacle points (robot frame).
        obstacle_mask:(B, P) bool — True = valid point; False (padded/invalid) ignored.
        footprint_radius, safety_margin: robot body radius + buffer (m).

    Returns: scalar tensor — mean hinge penalty over all waypoints.
    """
    margin = footprint_radius + safety_margin
    d = torch.cdist(pred_traj_xy, obstacle_xy.to(pred_traj_xy.dtype))  # (B, T, P)
    if obstacle_mask is not None:
        d = d.masked_fill(~obstacle_mask.unsqueeze(1), 1e6)
    min_d = d.min(dim=-1).values                                       # (B, T)
    return torch.relu(margin - min_d).mean()


# ---------------------------------------------------------------------------
# Learned critic: point cloud + trajectory -> collision risk
# ---------------------------------------------------------------------------
class CollisionCritic(nn.Module):
    """PointNet-style cloud encoder + per-waypoint trajectory head.

    Inputs:
        points: (B, N, 3) local point cloud (robot frame; pad with point_mask).
        traj:   (B, T, traj_dim) candidate trajectory waypoints.
    Outputs:
        traj_risk: (B,)    whole-trajectory collision risk in [0, 1]
        wp_risk:   (B, T)  per-waypoint collision risk in [0, 1]
    """

    def __init__(self, point_feat=128, traj_dim=3, hidden=128):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, point_feat), nn.ReLU(),
        )
        self.wp_mlp = nn.Sequential(
            nn.Linear(traj_dim, 64), nn.ReLU(),
            nn.Linear(64, hidden), nn.ReLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(point_feat + hidden, hidden), nn.ReLU(),
        )
        self.wp_head = nn.Linear(hidden, 1)
        self.traj_head = nn.Linear(hidden, 1)

    def forward(self, points, traj, point_mask=None):
        pf = self.point_mlp(points)                          # (B, N, F)
        if point_mask is not None:
            pf = pf.masked_fill(~point_mask.unsqueeze(-1), -1e9)
        global_pf = pf.max(dim=1).values                     # (B, F) PointNet maxpool
        wf = self.wp_mlp(traj)                               # (B, T, H)
        T = traj.shape[1]
        fused = self.fuse(torch.cat(
            [wf, global_pf.unsqueeze(1).expand(-1, T, -1)], dim=-1))  # (B, T, H)
        wp_risk = torch.sigmoid(self.wp_head(fused)).squeeze(-1)     # (B, T)
        traj_risk = torch.sigmoid(self.traj_head(fused.max(dim=1).values)).squeeze(-1)  # (B,)
        return traj_risk, wp_risk
