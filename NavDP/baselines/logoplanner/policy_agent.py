import os
import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from policy_network import LoGoPlanner_Policy
from collision_critic import obstacles_from_depth, rerank_trajectories

# Stage 1: at inference the policy must be built with the SAME use_depth as the
# checkpoint. RGB-only checkpoints have no depth_model; building use_depth=True
# would add a randomly-initialised depth encoder and corrupt the trajectory.
# Set USE_DEPTH=0 to load an RGB-only (Stage 1) checkpoint. Default True keeps
# legacy RGB-D checkpoints working.
_USE_DEPTH = os.environ.get('USE_DEPTH', '1') != '0'

# Stage 4 (inference): feed the multi-stop model an in-distribution NEAR subgoal
# (clamp the far goal to SUBGOAL_DIST metres) instead of the far final goal.
# SUBGOAL_INFER=1 to enable; SUBGOAL_DIST sets the subgoal radius (default 1.5 m,
# matching the training subgoal spacing).
_SUBGOAL_INFER = os.environ.get('SUBGOAL_INFER', '0') == '1'
_SUBGOAL_DIST = float(os.environ.get('SUBGOAL_DIST', '1.5'))

class LoGoPlanner_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 context_size=12,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 navi_model = "./100.ckpt",
                 use_critic_rerank=False,
                 footprint_radius=0.3,
                 safety_dist=0.3,
                 collision_threshold=0.5,
                 safety_weight=1.0,
                 device='cuda:0'):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self.context_size = context_size
        # Stage 6: geometric collision reranking of the diffusion candidates.
        # depth (kept after Stage 1) → point cloud → per-candidate collision risk;
        # filter unsafe, pick safest-toward-subgoal. Off by default.
        self.use_critic_rerank = use_critic_rerank
        self.footprint_radius = footprint_radius
        self.safety_dist = safety_dist
        self.collision_threshold = collision_threshold
        self.safety_weight = safety_weight
        self.navi_former = LoGoPlanner_Policy(image_size,memory_size,context_size,predict_size,temporal_depth,heads,token_dim,use_depth=_USE_DEPTH,device=device)
        # Trainer wraps the policy network in LoGoPlanner_Net (a `.policy` attribute),
        # so saved checkpoints have keys prefixed with `policy.`. Strip the prefix
        # before loading; otherwise strict=False silently drops every key and the
        # network keeps its random init.
        raw_sd = torch.load(navi_model, map_location=self.device, weights_only=False)
        if isinstance(raw_sd, dict) and 'state_dict' in raw_sd:
            raw_sd = raw_sd['state_dict']
        if any(k.startswith('policy.') for k in raw_sd.keys()):
            raw_sd = {k[len('policy.'):]: v for k, v in raw_sd.items() if k.startswith('policy.')}
        load_res = self.navi_former.load_state_dict(raw_sd, strict=False)
        print(f"[LoGoPlanner] ckpt load: missing={len(load_res.missing_keys)}, unexpected={len(load_res.unexpected_keys)}")
        if len(load_res.missing_keys) > 0:
            print(f"[LoGoPlanner] WARNING first missing: {load_res.missing_keys[:3]}")
        self.navi_former.to(self.device)
        self.navi_former.eval()
        self.target_H, self.target_W = 168, 308
    
    def reset(self,batch_size,threshold):
        print("================ LogoPlanner Agent Reset ================")
        self.batch_size = batch_size
        self.stop_threshold = threshold
        self.memory_queue = [[] for i in range(batch_size)]
        self.depth_queue = [[] for i in range(batch_size)]
        self.goal_queue = [[] for i in range(batch_size)]
        # Streaming GCT: next step starts a fresh episode (clears KV cache +
        # per-env token buffers inside the policy). Shared KV cache → all envs
        # reset together (lock-step episodes).
        self._stream_first = True

    def reset_env(self,i):
        self.memory_queue[i] = []
        self.depth_queue[i] = []
        self.goal_queue[i] = []
    
    def project_trajectory(self,images,n_trajectories,n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i,:,:,0:2]
            n_value = n_values[i]
            for waypoints,value in zip(n_trajectory,n_value):
                norm_value = np.clip(-value*0.1,0,1)
                colormap = cm.get('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0],3)) - 0.2
                input_points[:,0:2] = waypoints
                input_points[:,1] = -input_points[:,1]
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + self.image_intrinsic[0][2]
                for i in range(camera_x.shape[0]-1):
                    try:
                        if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[i]),int(camera_z[i])),(int(camera_x[i+1]),int(camera_z[i+1])),color.astype(np.uint8).tolist(),5)
                    except:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks,axis=1)

    def process_image(self,images):
        assert len(images.shape) == 4
        H,W,C = images.shape[1],images.shape[2],images.shape[3]
        return_images = []
        for img in images:
            resize_image = cv2.resize(img,(self.target_W, self.target_H))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self,depths):
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0
        H,W,C = depths.shape[1],depths.shape[2],depths.shape[3]
        prop = self.image_size/max(H,W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth,(self.target_W, self.target_H))
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)
    
    def process_pixel(self,pixel_coords,input_images):
        return_pixels = []
        H,W,C = input_images.shape[1],input_images.shape[2],input_images.shape[3]
        prop = self.image_size/max(H,W)
        for pixel_coord,input_image in zip(pixel_coords,input_images):
            panel_image = np.zeros_like(input_image,dtype=np.uint8)
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10
            
            if min_x <= 0:
                panel_image[:,0:10] = 255
            elif min_y <= 0:
                panel_image[0:10,:] = 255
            elif max_x >= panel_image.shape[1]:
                panel_image[:,panel_image.shape[1]-10:] = 255
            elif max_y >= panel_image.shape[0]:
                panel_image[panel_image.shape[0]-10:,:] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                panel_image[min_y:max_y,min_x:max_x] = 255
            
            resize_image = cv2.resize(panel_image,(-1,-1),fx=prop,fy=prop, interpolation=cv2.INTER_NEAREST)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_pixels.append(resize_image)
        return np.array(return_pixels).mean(axis=-1)
    
    def process_pointgoal(self,goals):
        clip_goals = goals.clip(-10,10)
        clip_goals[:,0] = np.clip(clip_goals[:,0],0,10)
        # Stage 4 (inference): the multi-stop model was TRAINED on nearby subgoals
        # (<= subgoal_dist). At eval the env hands us the FAR final goal, which is
        # out-of-distribution. Clamp the goal vector to the subgoal radius along the
        # SAME direction → feed an in-distribution subgoal. As the robot advances,
        # the goal vector shrinks; once within subgoal_dist we feed the true goal.
        # Heading is set to the bearing toward the subgoal. Env-gated (default off).
        if _SUBGOAL_INFER:
            xy = clip_goals[:, 0:2]
            d = np.linalg.norm(xy, axis=-1, keepdims=True)
            far = (d[:, 0] > _SUBGOAL_DIST)
            scale = np.where(d[:, 0:1] > 1e-6, _SUBGOAL_DIST / np.maximum(d, 1e-6), 0.0)
            sub_xy = np.where(far[:, None], xy * scale, xy)
            clip_goals = clip_goals.copy()
            clip_goals[:, 0:2] = sub_xy
            # face the subgoal: theta = atan2(y, x)
            clip_goals[:, 2] = np.where(far, np.arctan2(sub_xy[:, 1], sub_xy[:, 0]), clip_goals[:, 2])
        return clip_goals
    
    def step_nogoal(self,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_images = []
        for i in range(len(self.memory_queue)):
            if len(self.memory_queue[i]) < self.memory_size:
                self.memory_queue[i].append(process_images[i])
                input_image = np.array(self.memory_queue[i])
                input_image = np.pad(input_image,((self.memory_size - input_image.shape[0],0),(0,0),(0,0),(0,0)))
            else:
                del self.memory_queue[i][0]
                self.memory_queue[i].append(process_images[i])    
                input_image = np.array(self.memory_queue[i])
                
            input_images.append(input_image)
        input_image = np.array(input_images)
        input_depth = process_depths
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self.navi_former.predict_nogoal_action(input_image,input_depth)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values) 
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
    
    def get_indices(self, start_choice, current_choice, context_size):
        distance = current_choice - start_choice
        if distance < context_size:
            indices = [start_choice] * (context_size - distance - 1)
            # 添加剩余的点
            indices.extend(range(start_choice, current_choice + 1))
        else:
            # 计算步长，确保能取到8个点（包括起止点）
            step = distance / (context_size - 1)
            # 生成等间隔的索引
            indices = [start_choice + int(round(i * step)) for i in range(context_size)]
            # 确保最后一个点是current_choice
            indices[-1] = current_choice
        return indices
    
    def visualize_rgb_grid(self, rgb_data, prefix):
        """
        可视化RGB数据并保存为图像
        
        Args:
            rgb_data: RGB数据，形状为 (batch_size, num_frames, height, width, 3)
            prefix: 保存图像文件的前缀
        """
        for i in range(rgb_data.shape[0]):  # 遍历每个环境
            rgb = rgb_data[i]  # (num_frames, height, width, 3)
            
            # 创建一个网格图像来显示所有帧
            num_frames, height, width, _ = rgb.shape
            grid_cols = 4  # 网格的列数
            grid_rows = (num_frames + grid_cols - 1) // grid_cols  # 计算需要的行数
            
            # 创建空白网格
            rgb_grid = np.zeros((grid_rows * height, grid_cols * width, 3), dtype=np.uint8)
            
            # 填充网格
            for j in range(num_frames):
                row = j // grid_cols
                col = j % grid_cols
                # 将图像转换为uint8格式
                frame = (rgb[j] * 255).astype(np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # 将帧放入网格
                rgb_grid[row*height:(row+1)*height, col*width:(col+1)*width] = frame
            
            # 保存图像
            cv2.imwrite(f"{prefix}_env_{i}.jpg", rgb_grid)
    
    def step_pointgoal(self,goals,images,depths):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)

        # ---- Streaming GCT path (LOGO_STREAMING=1) --------------------------
        # Feed the single current frame through the backbone's persistent KV
        # cache (anchor 8 + window 64 + trajectory memory) instead of crushing
        # the whole episode into 12 subsampled frames every step.
        if getattr(self.navi_former, '_streaming', False):
            return self._step_pointgoal_stream(goals, images, process_images, process_depths)

        memory_rgbds = []
        context_rgbds = []
        for i in range(len(self.memory_queue)): # envs
            self.memory_queue[i].append(process_images[i])
            self.depth_queue[i].append(process_depths[i])
            self.goal_queue[i].append(goals[i])
            memory_length = len(self.memory_queue[i])
            indices = self.get_indices(0, memory_length - 1, self.context_size)
            input_image = np.array(self.memory_queue[i])[indices]
            input_depth = np.array(self.depth_queue[i])[indices]
            context_rgbds.append(np.concatenate([input_image, input_depth], axis=-1))

            current_length = len(self.memory_queue[i])
            start_idx = max(current_length - self.memory_size, 0)
            indices = list(range(start_idx, current_length))

            zeros_needed = self.memory_size - len(indices)

            if zeros_needed > 0:
                indices = [0] * zeros_needed + indices       

            input_image = np.array([self.memory_queue[i][j] for j in indices])
            input_depth = np.array([self.depth_queue[i][j] for j in indices])
            memory_rgbds.append(np.concatenate([input_image, input_depth], axis=-1))
    
        memory_rgbds = np.array(memory_rgbds) # (1, 8, 224, 224, 4)
        context_rgbds = np.array(context_rgbds) # (1, 8, 224, 224, 4)
        context_rgbds[..., -1][context_rgbds[..., -1] > 5.0] = 0
        context_rgbds[..., -1][context_rgbds[..., -1] < 0.1] = 0
        memory_rgbds[..., -1][memory_rgbds[..., -1] > 5.0] = 0
        memory_rgbds[..., -1][memory_rgbds[..., -1] < 0.1] = 0

        start_goal = goals
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory, sub_pointgoal_pd = self.navi_former.predict_pointgoal_action(start_goal,memory_rgbds,context_rgbds)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())

        print(all_values.max(),all_values.min())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values)

        chosen = good_trajectory[:, 0]
        if self.use_critic_rerank:
            # Stage 6: geometric collision reranking over the diffusion candidates.
            # all_trajectory: (B, K, T, 3); pick per-env the safest-toward-subgoal one.
            chosen = self._rerank_pointgoal(all_trajectory, all_values, goals, process_depths)
        return chosen, all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd

    def _step_pointgoal_stream(self, goals, images, process_images, process_depths):
        """One streaming decision step: push the current frame, assemble the GCT
        summary over the whole episode so far, and run the diffusion policy."""
        imgs = np.asarray(process_images, np.float32)          # (B, H, W, 3) in [0,1]
        deps = np.asarray(process_depths, np.float32)          # (B, H, W, 1) meters
        if deps.ndim == 3:
            deps = deps[..., None]
        deps[..., 0][deps[..., 0] > 5.0] = 0
        deps[..., 0][deps[..., 0] < 0.1] = 0

        image_t = torch.as_tensor(imgs, dtype=torch.float32, device=self.device)
        depth_t = torch.as_tensor(deps, dtype=torch.float32, device=self.device)

        episode_start = bool(getattr(self, '_stream_first', True))
        self._stream_first = False

        # Multi-stop long-route navigation: the model was trained on NEARBY subgoals
        # (~subgoal_dist m), so feeding the far final goal (~6 m, OOD) makes it wander.
        # process_pointgoal (SUBGOAL_INFER=1) clamps the far goal to a subgoal_dist-m
        # waypoint along the SAME bearing; as the robot advances the vector shrinks,
        # so it chains subgoal→subgoal until within subgoal_dist of the true goal.
        start_goal = self.process_pointgoal(np.asarray(goals, dtype=np.float32))
        all_trajectory, all_values, good_trajectory, bad_trajectory, sub_pointgoal_pd = (
            self.navi_former.predict_pointgoal_action_stream(
                start_goal, image_t, depth_t, episode_start=episode_start,
            )
        )
        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = good_trajectory[:, :, :, 0] * 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())

        print(all_values.max(), all_values.min())
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)

        chosen = good_trajectory[:, 0]
        if self.use_critic_rerank:
            chosen = self._rerank_pointgoal(all_trajectory, all_values, goals, process_depths)
        return chosen, all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd

    def _scale_intrinsic(self, K, H, W):
        """Scale a 3x3 intrinsic to a (H, W) image, inferring the source size from
        the principal point (W0 ≈ 2·cx, H0 ≈ 2·cy)."""
        K = np.asarray(K, np.float32).copy()
        W0 = max(2.0 * K[0, 2], 1.0)
        H0 = max(2.0 * K[1, 2], 1.0)
        sx, sy = W / W0, H / H0
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy
        return K

    def _rerank_pointgoal(self, all_trajectory, all_values, goals, process_depths):
        """Per-env collision-aware reranking; returns (B, T, 3) chosen trajectories."""
        B = all_trajectory.shape[0]
        chosen = np.zeros((B, all_trajectory.shape[2], all_trajectory.shape[3]), np.float32)
        for i in range(B):
            depth_i = np.asarray(process_depths[i], np.float32)            # (H, W, 1)
            H, W = depth_i.shape[0], depth_i.shape[1]
            K = self._scale_intrinsic(self.image_intrinsic, H, W)
            obstacle_xy = obstacles_from_depth(depth_i, K)
            res = rerank_trajectories(
                all_trajectory[i], obstacle_xy, np.asarray(goals[i], np.float32)[:2],
                footprint_radius=self.footprint_radius, safety_dist=self.safety_dist,
                collision_threshold=self.collision_threshold, safety_weight=self.safety_weight,
                learned_values=all_values[i],
            )
            sel = res['selected'].copy()
            if res['stop']:
                # all candidates collide → stop in place (rotate-search heading kept)
                sel[:, 0] = 0.0
            chosen[i] = sel
        return chosen

    def step_imagegoal(self, goal_images, images, depths):
        """Phase α: image-goal inference. goal_images shape (B, Hc, Wc, 3)."""
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        process_goal_images = self.process_image(goal_images)
        memory_rgbds = []
        context_rgbds = []
        for i in range(len(self.memory_queue)):
            self.memory_queue[i].append(process_images[i])
            self.depth_queue[i].append(process_depths[i])
            memory_length = len(self.memory_queue[i])
            indices = self.get_indices(0, memory_length - 1, self.context_size)
            input_image = np.array(self.memory_queue[i])[indices]
            input_depth = np.array(self.depth_queue[i])[indices]
            context_rgbds.append(np.concatenate([input_image, input_depth], axis=-1))

            current_length = len(self.memory_queue[i])
            start_idx = max(current_length - self.memory_size, 0)
            indices = list(range(start_idx, current_length))
            zeros_needed = self.memory_size - len(indices)
            if zeros_needed > 0:
                indices = [0] * zeros_needed + indices
            input_image = np.array([self.memory_queue[i][j] for j in indices])
            input_depth = np.array([self.depth_queue[i][j] for j in indices])
            memory_rgbds.append(np.concatenate([input_image, input_depth], axis=-1))

        memory_rgbds = np.array(memory_rgbds)
        context_rgbds = np.array(context_rgbds)
        context_rgbds[..., -1][context_rgbds[..., -1] > 5.0] = 0
        context_rgbds[..., -1][context_rgbds[..., -1] < 0.1] = 0
        memory_rgbds[..., -1][memory_rgbds[..., -1] > 5.0] = 0
        memory_rgbds[..., -1][memory_rgbds[..., -1] < 0.1] = 0

        all_trajectory, all_values, good_trajectory, bad_trajectory, sub_pointgoal_pd = \
            self.navi_former.predict_imagegoal_action(process_goal_images, memory_rgbds, context_rgbds)
        if all_values.max() < self.stop_threshold:
            good_trajectory[:, :, :, 0] = good_trajectory[:, :, :, 0] * 0.0
            good_trajectory[:, :, :, 1] = np.sign(good_trajectory[:, :, :, 1].mean())
        print(all_values.max(), all_values.min())
        trajectory_mask = self.project_trajectory(images, all_trajectory, all_values)
        return good_trajectory[:, 0], all_trajectory, all_values, trajectory_mask, sub_pointgoal_pd
