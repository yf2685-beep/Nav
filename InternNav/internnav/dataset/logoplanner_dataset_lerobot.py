"""Dataset + collate_fn for LoGoPlanner training.

Extends NavDP_Base_Datset to additionally produce the GT signals needed by the
geometry-grounded losses in LoGoPlanner (Peng et al., arxiv 2512.19629).

Per paper §IV.B, supervision comes from:
  - Per-frame camera poses T_c,i (for ExtrinctHead, eq 4)
  - Per-pixel local points P_cam = D(u,v) · K⁻¹ · [u v 1]ᵀ (eq 2/3)
  - Per-pixel world points P_world = T_c,i · P_cam (eq 6)

All geometry GTs are expressed in the chassis frame of the last context step
(paper §IV.C: "world coordinate system is defined with respect to the chassis
frame of the last time step"). This matches what the model predicts.

Additional batch keys beyond NavDP:
  batch_context_rgb       [B, N, Hc, Wc, 3]   N consecutive RGB frames for Pi3
  batch_context_depth     [B, N, Hc, Wc, 1]   matching depth (for depth prior)
  batch_gt_camera_poses   [B, N, 5]           [x, y, z, sinθ, cosθ] relative to current
  batch_gt_local_points   [B, N, Hc, Wc, 3]   unprojected depth in camera frame
  batch_gt_world_points   [B, N, Hc, Wc, 3]   relative to current chassis
  batch_gt_subgoal        [B, 3]              (x, y, θ) goal in current frame

Context image size defaults to (168, 308) — matches the shape used in
NavDP/baselines/logoplanner/policy_network.py __main__ and is divisible by the
ViT patch size 14.

Caveats / design choices flagged for review:
  - Pose encoding is [x, y, z, sinθ, cosθ] (5 dims, matches ExtrinctHead.fc_pose).
    Paper describes 3-DoF (x, y, θ) but the code outputs 5. Change the encoding
    here + the model head together if the paper encoding is preferred.
  - batch_gt_subgoal reuses `point_goal` (final waypoint in current frame).
    If the paper intends a mid-horizon waypoint, change `_compute_subgoal_gt`.
  - Context window is the last N frames ending at `memory_start_choice` with
    stride 1. If a trajectory is too short, earlier slots are zero-padded and
    their GT is masked to zero (so the MSE contribution vanishes).
"""

import os
import time

import cv2
import numpy as np
import torch
from PIL import Image

from internnav.dataset.navdp_dataset_lerobot import NavDP_Base_Datset


class LoGoPlanner_Dataset(NavDP_Base_Datset):
    def __init__(
        self,
        root_dirs,
        preload_path=False,
        memory_size=8,
        predict_size=24,
        batch_size=64,
        image_size=224,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        debug=False,
        preload=False,
        random_digit=False,
        prior_sample=False,
        context_size=12,
        context_image_height=168,
        context_image_width=308,
        depth_max=5.0,
        depth_min=0.1,
        critic_goal_weight=0.0,
        sequential=False,
        seq_stride=1,
        multistop=False,
        subgoal_dist=1.5,
        subgoal_turn_deg=30.0,
        subgoal_arrival=0.5,
    ):
        super().__init__(
            root_dirs,
            preload_path=preload_path,
            memory_size=memory_size,
            predict_size=predict_size,
            batch_size=batch_size,
            image_size=image_size,
            scene_data_scale=scene_data_scale,
            trajectory_data_scale=trajectory_data_scale,
            debug=debug,
            preload=preload,
            random_digit=random_digit,
            prior_sample=prior_sample,
        )
        self.context_size = context_size
        self.context_image_height = context_image_height
        self.context_image_width = context_image_width
        self.depth_max = depth_max
        self.depth_min = depth_min
        # >0 enables a goal-aware critic GT term (direction 3): the critic value
        # is additionally penalised by how far the trajectory's endpoint lands
        # from the goal, so the critic ranks goal-reaching — not just obstacle
        # avoidance. 0.0 keeps the original obstacle-only critic GT.
        self.critic_goal_weight = critic_goal_weight

        # ---- Stage 2: LogoPlanner-style sequential / streaming sampling ----
        # When sequential=True, each dataset item is a single (episode, timestep)
        # pair read in temporal order (deterministic windows) instead of a
        # randomly-sampled segment. This lets the model build a per-episode KV
        # cache in Stage 3 (the cache stays in the model; the dataloader only
        # guarantees ordering + boundary flags). Random-segment mode
        # (sequential=False) is unchanged and stays the default.
        self.sequential = sequential
        self.seq_stride = max(1, int(seq_stride))
        if self.sequential:
            self._build_frame_index()

        # ---- Stage 4: multi-stop subgoal navigation ----
        # Instead of always conditioning on the FINAL goal image (which the robot
        # often can't see → random walk), split each GT trajectory into several
        # shorter subgoals and condition on the CURRENT (nearby, visible) subgoal.
        # Subgoals are placed every `subgoal_dist` metres, with an extra subgoal
        # whenever heading turns by more than `subgoal_turn_deg` since the last
        # one; the final frame is always the last subgoal. GT-derived only — no
        # learned subgoal selector yet (spec §4: stabilise the base first).
        self.multistop = multistop
        self.subgoal_dist = float(subgoal_dist)
        self.subgoal_turn_deg = float(subgoal_turn_deg)
        self.subgoal_arrival = float(subgoal_arrival)
        self._subgoal_cache = {}

    # ---- Stage 4: subgoal generation ----

    def _compute_subgoals(self, extrinsics):
        """Place subgoal frame indices along an ordered trajectory.

        A subgoal is emitted every `subgoal_dist` metres of travelled distance OR
        whenever heading turns by > `subgoal_turn_deg` since the last subgoal. The
        final frame (L-1) is always the last subgoal.

        Args:
            extrinsics: (L, 4, 4) camera-to-world per frame (temporally ordered).
        Returns:
            np.int64 array of strictly increasing subgoal indices, ending at L-1.
        """
        L = extrinsics.shape[0]
        if L <= 1:
            return np.array([max(0, L - 1)], np.int64)
        pos = extrinsics[:, 0:2, 3]                     # (L, 2) ground-plane xy
        fwd = extrinsics[:, 0:2, 0]                     # (L, 2) forward axis (matches pose enc)
        yaw = np.arctan2(fwd[:, 1], fwd[:, 0])          # (L,)
        turn_rad = np.deg2rad(self.subgoal_turn_deg)

        subgoals = []
        acc_dist = 0.0
        anchor_yaw = yaw[0]
        for i in range(1, L):
            acc_dist += float(np.linalg.norm(pos[i] - pos[i - 1]))
            dyaw = abs(np.arctan2(np.sin(yaw[i] - anchor_yaw), np.cos(yaw[i] - anchor_yaw)))
            if acc_dist >= self.subgoal_dist or dyaw >= turn_rad:
                subgoals.append(i)
                acc_dist = 0.0
                anchor_yaw = yaw[i]
        if not subgoals or subgoals[-1] != L - 1:
            subgoals.append(L - 1)
        return np.array(subgoals, np.int64)

    def _get_subgoals(self, index, extrinsics):
        """Subgoals are a property of the episode → compute once and cache."""
        sg = self._subgoal_cache.get(index)
        if sg is None:
            sg = self._compute_subgoals(extrinsics)
            self._subgoal_cache[index] = sg
        return sg

    @staticmethod
    def _current_subgoal(subgoals, t):
        """First subgoal strictly ahead of the current frame t (else the last)."""
        for s in subgoals:
            if s > t:
                return int(s)
        return int(subgoals[-1])

    @staticmethod
    def _next_subgoal(subgoals, current_idx):
        """The subgoal after `current_idx` (else the final goal)."""
        for s in subgoals:
            if s > current_idx:
                return int(s)
        return int(subgoals[-1])

    def _build_frame_index(self):
        """Flatten every episode into ordered (traj_idx, t, is_first, is_last) frames.

        Builds:
          self.frame_index : list[(traj_idx, t, is_first, is_last)]   (len == __len__)
          self.episodes    : list[list[flat_idx]]  one inner list per episode,
                             frames in temporal order (consumed by the streaming
                             batch sampler so different episodes never mix in a lane).
        """
        self.frame_index = []
        self.episodes = []
        for traj_idx in range(len(self.trajectory_dirs)):
            n_frames = len(self.trajectory_rgb_path[traj_idx])
            if n_frames < 2:
                continue
            ts = list(range(0, n_frames, self.seq_stride))
            ep_flat = []
            for k, t in enumerate(ts):
                is_first = k == 0
                is_last = k == len(ts) - 1
                ep_flat.append(len(self.frame_index))
                self.frame_index.append((traj_idx, t, is_first, is_last))
            self.episodes.append(ep_flat)
        print(
            f'[LoGoPlanner sequential] {len(self.episodes)} episodes, '
            f'{len(self.frame_index)} frames (stride={self.seq_stride})'
        )

    def __len__(self):
        if getattr(self, 'sequential', False):
            return len(self.frame_index)
        return super().__len__()

    # ---- Context image/depth loading (aspect-preserving, not squared) ----

    def _resize_preserve(self, arr, h, w, interp=cv2.INTER_LINEAR):
        return cv2.resize(arr, (w, h), interpolation=interp)

    def process_context_image(self, image_path):
        img = self.load_image(image_path)
        img = self._resize_preserve(img, self.context_image_height, self.context_image_width)
        return np.array(img, np.float32) / 255.0

    def process_context_depth(self, depth_path):
        depth = self.load_depth(depth_path) / 10000.0
        depth = self._resize_preserve(depth, self.context_image_height, self.context_image_width, cv2.INTER_NEAREST)
        depth = np.array(depth, np.float32)
        depth[depth > self.depth_max] = 0.0
        depth[depth < self.depth_min] = 0.0
        return depth[:, :, np.newaxis]

    def process_context_window(self, rgb_paths, depth_paths, end_step):
        """Load N consecutive (RGB, depth) frames ending at end_step (inclusive).

        Returns arrays of shape (N, Hc, Wc, 3) and (N, Hc, Wc, 1) plus a mask
        (N,) bool indicating which slots are valid (False → zero-padded).
        """
        N = self.context_size
        Hc, Wc = self.context_image_height, self.context_image_width
        indices = np.arange(end_step - N + 1, end_step + 1)
        pad = (indices < 0).sum()
        indices_valid = indices[pad:]

        rgb = np.zeros((N, Hc, Wc, 3), np.float32)
        depth = np.zeros((N, Hc, Wc, 1), np.float32)
        mask = np.zeros((N,), np.bool_)

        for slot, t in enumerate(indices_valid, start=pad):
            rgb[slot] = self.process_context_image(rgb_paths[t])
            depth[slot] = self.process_context_depth(depth_paths[t])
            mask[slot] = True
        return rgb, depth, mask, indices

    # ---- Memory loader override ----
    # NavDP_RGBD_Backbone hardcodes (memory_size+1)*264 positional embeddings,
    # i.e. a 12x22 patch grid for 168x308 inputs at ViT patch size 14. The base
    # class would produce square image_size x image_size memory frames; override
    # so memory and context share the aspect-preserving 168x308 resize.

    def process_memory(self, rgb_paths, depth_paths, start_step, memory_digit=1):
        memory_index = np.arange(
            start_step - (self.memory_size - 1) * memory_digit, start_step + 1, memory_digit
        )
        outrange_sum = (memory_index < 0).sum()
        memory_index = memory_index[outrange_sum:]
        Hc, Wc = self.context_image_height, self.context_image_width
        context_image = np.zeros((self.memory_size, Hc, Wc, 3), np.float32)
        context_image[outrange_sum:] = np.array(
            [self.process_context_image(rgb_paths[i]) for i in memory_index]
        )
        context_depth = self.process_context_depth(depth_paths[start_step])
        # valid history positions (False = zero-padded because the window reaches
        # before the episode start). Exposed so the model can mask padded frames.
        memory_mask = np.zeros((self.memory_size,), np.bool_)
        memory_mask[outrange_sum:] = True
        return context_image, context_depth, memory_index, memory_mask

    # ---- Geometry GT computation ----

    def _pose_to_xyzsincos(self, R, T):
        """Encode a ground-plane pose as [x, y, z, sinθ, cosθ].

        θ is the yaw angle between the pose's forward direction and the current
        frame's x-axis. Using a 2D unit vector for yaw (sin, cos) avoids the
        2π wraparound that hurts MSE.
        """
        # Forward axis in world → project to ground plane, normalize, take yaw
        # Convention: R's first column is the "forward" of the pose frame.
        fwd = R[:, 0]
        yaw = np.arctan2(fwd[1], fwd[0])
        return np.array([T[0], T[1], T[2], np.sin(yaw), np.cos(yaw)], np.float32)

    def _intrinsic_for_context(self, K_orig, orig_h, orig_w):
        """Scale a 3x3 intrinsic for the context resolution.

        Raw depth PNGs come at some native resolution. After resizing to
        (context_image_height, context_image_width), the intrinsic's fx, fy,
        cx, cy must be rescaled by the axis ratios.
        """
        sx = self.context_image_width / orig_w
        sy = self.context_image_height / orig_h
        K = K_orig.copy().astype(np.float32)
        K[0, 0] *= sx  # fx
        K[1, 1] *= sy  # fy
        K[0, 2] *= sx  # cx
        K[1, 2] *= sy  # cy
        return K

    def _unproject_depth(self, depth_hw1, K):
        """Unproject a (Hc, Wc, 1) depth image to (Hc, Wc, 3) camera-frame points.

        Invalid depths (==0) produce zero points.
        """
        H, W = depth_hw1.shape[:2]
        vs, us = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        uv1 = np.stack([us, vs, np.ones_like(us)], axis=-1).astype(np.float32)  # (H, W, 3)
        K_inv = np.linalg.inv(K).astype(np.float32)
        rays = uv1 @ K_inv.T  # (H, W, 3)
        d = depth_hw1[..., 0]
        valid = d > 0
        pts = rays * d[..., None]
        pts[~valid] = 0.0
        return pts.astype(np.float32)

    def _compute_geometry_gt(
        self,
        context_extrinsics,
        current_extrinsic,
        base_extrinsic,
        context_depths,
        context_mask,
        K_context,
    ):
        """Compute per-frame pose, local points, and world points relative to current.

        Args:
            context_extrinsics: (N, 4, 4) camera-to-world for each context frame.
                                Slots where mask=False are ignored (identity).
            current_extrinsic:  (4, 4) camera-to-world for the current (memory_start) frame.
            base_extrinsic:     (4, 4) chassis→camera calibration (constant per episode).
            context_depths:     (N, Hc, Wc, 1) resized depth (meters).
            context_mask:       (N,) bool, valid frames.
            K_context:          (3, 3) intrinsic matched to context resolution.

        Returns:
            gt_poses:         (N, 5)
            gt_local_points:  (N, Hc, Wc, 3)
            gt_world_points:  (N, Hc, Wc, 3) — in current-frame world
        """
        N = context_extrinsics.shape[0]
        Hc = self.context_image_height
        Wc = self.context_image_width

        R_cur = current_extrinsic[0:3, 0:3]
        T_cur = current_extrinsic[0:3, 3]

        gt_poses = np.zeros((N, 5), np.float32)
        gt_local = np.zeros((N, Hc, Wc, 3), np.float32)
        gt_world = np.zeros((N, Hc, Wc, 3), np.float32)

        for i in range(N):
            if not context_mask[i]:
                continue
            Ri = context_extrinsics[i, 0:3, 0:3]
            Ti = context_extrinsics[i, 0:3, 3]
            R_rel, T_rel = self.relative_pose(R_cur, T_cur, Ri, Ti, base_extrinsic)
            gt_poses[i] = self._pose_to_xyzsincos(R_rel, T_rel)

            local_pts = self._unproject_depth(context_depths[i], K_context)
            gt_local[i] = local_pts

            # world points (expressed in current chassis frame):
            #   p_world_cur = R_rel · p_cam + T_rel
            valid = local_pts.sum(axis=-1) != 0
            world_pts = local_pts @ R_rel.T + T_rel[None, None, :]
            world_pts[~valid] = 0.0
            gt_world[i] = world_pts.astype(np.float32)

        return gt_poses, gt_local, gt_world

    def _compute_subgoal_gt(self, point_goal):
        """Sub-pointgoal = final target in current frame. Placeholder; see file docstring."""
        return np.array(point_goal, np.float32)

    # ---- Main entry ----

    def __getitem__(self, index):
        start_time = time.time()

        # Stage 2: sequential mode decodes a flat (episode, timestep) index and
        # remaps `index` to the trajectory so every downstream self.*[index]
        # access is correct. Random-segment mode leaves `index` untouched.
        seq_episode_start = False
        seq_done = False
        seq_timestep = -1
        if getattr(self, 'sequential', False):
            traj_idx, t, is_first, is_last = self.frame_index[index]
            index = traj_idx
            seq_episode_start, seq_done, seq_timestep = bool(is_first), bool(is_last), int(t)
            # per-episode seed → augmentation multipliers stay consistent across
            # the frames of one episode (temporal consistency for the same trajectory).
            np.random.seed(traj_idx % (2 ** 31 - 1))

        (
            camera_intrinsic,
            trajectory_base_extrinsic,
            trajectory_extrinsics,
            trajectory_length,
        ) = self.process_data_parquet(index)

        trajectory_path_points, _ = self.process_path_points(index)
        trajectory_obstacle_points, _ = self.process_obstacle_points(index, trajectory_path_points)

        if getattr(self, 'sequential', False):
            # deterministic windows: current frame = t, goal = trajectory end.
            memory_start_choice = int(min(seq_timestep, trajectory_length - 2))
            memory_start_choice = max(0, memory_start_choice)
            target_choice = trajectory_length - 1
            pixel_start_choice = 0
        elif self.prior_sample:
            pixel_start_choice, target_choice = self.rank_steps(
                trajectory_extrinsics, trajectory_obstacle_points
            )
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)
        else:
            pixel_start_choice = np.random.randint(0, trajectory_length // 2)
            target_choice = np.random.randint(pixel_start_choice + 1, trajectory_length - 1)
            memory_start_choice = np.random.randint(pixel_start_choice, target_choice)

        # Stage 4: multi-stop — retarget from the trajectory endpoint to the
        # CURRENT subgoal (first subgoal ahead of memory_start_choice). This makes
        # point_goal / pred_actions / goal_image all subgoal-local downstream.
        current_subgoal_idx = None
        if getattr(self, 'multistop', False):
            subgoals = self._get_subgoals(index, trajectory_extrinsics)
            current_subgoal_idx = self._current_subgoal(subgoals, memory_start_choice)
            target_choice = int(max(current_subgoal_idx, memory_start_choice + 1))
            target_choice = min(target_choice, trajectory_length - 1)

        if self.random_digit:
            memory_digit = np.random.randint(2, 8)
            pred_digit = memory_digit
        else:
            memory_digit = 4
            pred_digit = 4

        memory_images, depth_image, _, memory_mask = self.process_memory(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
            memory_digit=memory_digit,
        )

        context_rgb, context_depth, context_mask, context_indices = self.process_context_window(
            self.trajectory_rgb_path[index],
            self.trajectory_depth_path[index],
            memory_start_choice,
        )

        (
            target_local_points,
            augment_local_points,
            target_world_points,
            augment_world_points,
            action_indexes,
        ) = self.process_actions(
            trajectory_extrinsics,
            trajectory_base_extrinsic,
            memory_start_choice,
            target_choice,
            pred_digit=pred_digit,
        )

        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt_actions = self.xyz_to_xyt(target_local_points, init_vector)
        augment_xyt_actions = self.xyz_to_xyt(augment_local_points, init_vector)
        pred_actions = target_xyt_actions[action_indexes]
        augment_actions = augment_xyt_actions[action_indexes]

        if trajectory_obstacle_points.shape[0] != 0:
            pred_distance = (
                np.abs(target_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            augment_distance = (
                np.abs(augment_world_points[:, np.newaxis, 0:2] - trajectory_obstacle_points[np.newaxis, :, 0:2])
                .sum(axis=-1)
                .min(axis=-1)
            )
            pred_critic = (
                -5.0 * (pred_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (pred_distance[action_indexes][1:] - pred_distance[action_indexes][:-1]).sum()
            )
            augment_critic = (
                -5.0 * (augment_distance[action_indexes[:-1]] < 0.1).mean()
                + 0.5 * (augment_distance[action_indexes][1:] - augment_distance[action_indexes][:-1]).sum()
            )
        else:
            pred_critic = 2.0
            augment_critic = 2.0

        point_goal = target_xyt_actions[-1]

        # direction-3: goal-aware critic GT. Penalise each trajectory's critic
        # value by how far its endpoint lands from the goal, so the critic
        # learns to rank goal-reaching (not only obstacle avoidance).
        if self.critic_goal_weight > 0:
            goal_xy = point_goal[0:2]
            pred_goal_dist = np.linalg.norm(target_xyt_actions[action_indexes][-1][0:2] - goal_xy)
            augment_goal_dist = np.linalg.norm(augment_xyt_actions[action_indexes][-1][0:2] - goal_xy)
            pred_critic = pred_critic - self.critic_goal_weight * float(pred_goal_dist)
            augment_critic = augment_critic - self.critic_goal_weight * float(augment_goal_dist)

        # --- geometry GT (new) ---
        # Use context_indices clipped into valid range; invalid slots already masked.
        clipped_ctx = np.clip(context_indices, 0, trajectory_length - 1)
        ctx_ext = trajectory_extrinsics[clipped_ctx]  # (N, 4, 4)
        cur_ext = trajectory_extrinsics[memory_start_choice]

        # original depth size → we need the native H,W to scale K correctly.
        # Read one depth directly to know its size.
        sample_depth = np.array(Image.open(self.trajectory_depth_path[index][0]), np.uint16)
        orig_h, orig_w = sample_depth.shape[:2]
        K_context = self._intrinsic_for_context(camera_intrinsic, orig_h, orig_w)

        gt_poses, gt_local_points, gt_world_points = self._compute_geometry_gt(
            ctx_ext,
            cur_ext,
            trajectory_base_extrinsic,
            context_depth,
            context_mask,
            K_context,
        )

        gt_subgoal = self._compute_subgoal_gt(point_goal)

        # action deltas (same as NavDP)
        pred_actions = (pred_actions[1:] - pred_actions[:-1]) * 4.0
        augment_actions = (augment_actions[1:] - augment_actions[:-1]) * 4.0

        # timing log (inherited pattern)
        end_time = time.time()
        self.item_cnt += 1
        self.batch_time_sum += end_time - start_time
        if self.item_cnt % self.batch_size == 0:
            avg_time = self.batch_time_sum / self.batch_size
            print(
                f'__getitem__ pid={os.getpid()}, avg_time(last {self.batch_size})={avg_time:.2f}s, cnt={self.item_cnt}'
            )
            self.batch_time_sum = 0.0

        # Phase α: goal_image = RGB at the trajectory endpoint (the "target view").
        # Same shape/preprocessing as context frames so downstream encoders fit.
        # Falls back to the last in-range context frame if endpoint is out of range.
        _last_step = min(int(trajectory_length) - 1, len(self.trajectory_rgb_path[index]) - 1)
        try:
            final_goal_image = self.process_context_image(self.trajectory_rgb_path[index][_last_step])
        except Exception:
            final_goal_image = context_rgb[-1]

        # --- Stage 4: multi-stop subgoal images + metadata ---
        # In multistop mode the model's goal_image is the CURRENT subgoal view
        # (nearby, usually visible) rather than the final destination. We also
        # emit the next subgoal + the final goal so downstream / inference logic
        # can switch subgoals and fall back to the final goal at the end.
        if getattr(self, 'multistop', False) and current_subgoal_idx is not None:
            subgoals = self._get_subgoals(index, trajectory_extrinsics)
            next_subgoal_idx = self._next_subgoal(subgoals, current_subgoal_idx)
            _cs = min(current_subgoal_idx, _last_step)
            _ns = min(next_subgoal_idx, _last_step)
            try:
                subgoal_image = self.process_context_image(self.trajectory_rgb_path[index][_cs])
            except Exception:
                subgoal_image = final_goal_image
            try:
                next_subgoal_image = self.process_context_image(self.trajectory_rgb_path[index][_ns])
            except Exception:
                next_subgoal_image = final_goal_image
            goal_image = subgoal_image  # model conditions on the current subgoal
            # reached: current frame is within arrival distance of the subgoal xy
            _cur_xy = trajectory_extrinsics[memory_start_choice, 0:2, 3]
            _sg_xy = trajectory_extrinsics[_cs, 0:2, 3]
            subgoal_reached = bool(np.linalg.norm(_sg_xy - _cur_xy) < self.subgoal_arrival)
            num_subgoals = int(len(subgoals))
        else:
            goal_image = final_goal_image
            subgoal_image = final_goal_image
            next_subgoal_image = final_goal_image
            current_subgoal_idx = _last_step if current_subgoal_idx is None else current_subgoal_idx
            subgoal_reached = False
            num_subgoals = 1

        # --- Stage 2 metadata: episode bookkeeping + history mask + robot pose ---
        # robot_pose: absolute world pose of the current (memory_start) frame,
        # encoded [x, y, z, sinθ, cosθ] (same convention as gt_camera_poses).
        robot_pose = self._pose_to_xyzsincos(cur_ext[0:3, 0:3], cur_ext[0:3, 3])
        if getattr(self, 'sequential', False):
            episode_id = index
            timestep = seq_timestep
            episode_start = seq_episode_start
            done = seq_done
        else:
            # random-segment mode: derive flags from the sampled current step so the
            # keys always exist (collate/trainer stay uniform across both modes).
            episode_id = index
            timestep = int(memory_start_choice)
            episode_start = bool(memory_start_choice == 0)
            done = bool(memory_start_choice >= trajectory_length - 1)
        reset_cache = episode_start  # reset the model's KV cache at each episode start

        return {
            'point_goal': torch.tensor(point_goal, dtype=torch.float32),
            'goal_image': torch.tensor(goal_image, dtype=torch.float32),
            'memory_rgb': torch.tensor(memory_images, dtype=torch.float32),
            'memory_depth': torch.tensor(depth_image, dtype=torch.float32),
            'context_rgb': torch.tensor(context_rgb, dtype=torch.float32),
            'context_depth': torch.tensor(context_depth, dtype=torch.float32),
            'pred_actions': torch.tensor(pred_actions, dtype=torch.float32),
            'augment_actions': torch.tensor(augment_actions, dtype=torch.float32),
            'pred_critic': torch.tensor(pred_critic, dtype=torch.float32),
            'augment_critic': torch.tensor(augment_critic, dtype=torch.float32),
            'gt_camera_poses': torch.tensor(gt_poses, dtype=torch.float32),
            'gt_local_points': torch.tensor(gt_local_points, dtype=torch.float32),
            'gt_world_points': torch.tensor(gt_world_points, dtype=torch.float32),
            'gt_subgoal': torch.tensor(gt_subgoal, dtype=torch.float32),
            # Stage 2 sequential / KV-cache metadata
            'memory_mask': torch.tensor(memory_mask, dtype=torch.bool),
            'context_mask': torch.tensor(context_mask, dtype=torch.bool),
            'robot_pose': torch.tensor(robot_pose, dtype=torch.float32),
            'episode_id': torch.tensor(episode_id, dtype=torch.long),
            'timestep': torch.tensor(timestep, dtype=torch.long),
            'episode_start': torch.tensor(episode_start, dtype=torch.bool),
            'reset_cache': torch.tensor(reset_cache, dtype=torch.bool),
            'done': torch.tensor(done, dtype=torch.bool),
            # Stage 4 multi-stop subgoal fields
            'subgoal_image': torch.tensor(subgoal_image, dtype=torch.float32),
            'next_subgoal_image': torch.tensor(next_subgoal_image, dtype=torch.float32),
            'final_goal_image': torch.tensor(final_goal_image, dtype=torch.float32),
            'current_subgoal_idx': torch.tensor(int(current_subgoal_idx), dtype=torch.long),
            'subgoal_reached': torch.tensor(subgoal_reached, dtype=torch.bool),
            'num_subgoals': torch.tensor(num_subgoals, dtype=torch.long),
        }


def logoplanner_collate_fn(batch):
    out = {
        'batch_pg':              torch.stack([b['point_goal']       for b in batch]),
        'batch_memory_rgb':      torch.stack([b['memory_rgb']       for b in batch]),
        'batch_memory_depth':    torch.stack([b['memory_depth']     for b in batch]),
        'batch_context_rgb':     torch.stack([b['context_rgb']      for b in batch]),
        'batch_context_depth':   torch.stack([b['context_depth']    for b in batch]),
        'batch_labels':          torch.stack([b['pred_actions']     for b in batch]),
        'batch_augments':        torch.stack([b['augment_actions']  for b in batch]),
        'batch_label_critic':    torch.stack([b['pred_critic']      for b in batch]),
        'batch_augment_critic':  torch.stack([b['augment_critic']   for b in batch]),
        'batch_gt_camera_poses': torch.stack([b['gt_camera_poses']  for b in batch]),
        'batch_gt_local_points': torch.stack([b['gt_local_points']  for b in batch]),
        'batch_gt_world_points': torch.stack([b['gt_world_points']  for b in batch]),
        'batch_gt_subgoal':      torch.stack([b['gt_subgoal']       for b in batch]),
    }
    if 'goal_image' in batch[0]:
        out['batch_goal_image'] = torch.stack([b['goal_image'] for b in batch])
    # Stage 2 sequential / KV-cache metadata (present in both modes).
    for src, dst in [
        ('memory_mask', 'batch_memory_mask'),
        ('context_mask', 'batch_context_mask'),
        ('robot_pose', 'batch_robot_pose'),
        ('episode_id', 'batch_episode_id'),
        ('timestep', 'batch_timestep'),
        ('episode_start', 'batch_episode_start'),
        ('reset_cache', 'batch_reset_cache'),
        ('done', 'batch_done'),
        ('subgoal_image', 'batch_subgoal_image'),
        ('next_subgoal_image', 'batch_next_subgoal_image'),
        ('final_goal_image', 'batch_final_goal_image'),
        ('current_subgoal_idx', 'batch_current_subgoal_idx'),
        ('subgoal_reached', 'batch_subgoal_reached'),
        ('num_subgoals', 'batch_num_subgoals'),
    ]:
        if src in batch[0]:
            out[dst] = torch.stack([b[src] for b in batch])
    return out
