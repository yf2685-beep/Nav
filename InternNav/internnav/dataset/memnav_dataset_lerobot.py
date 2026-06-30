"""Dataset + collate_fn for `memnav` (implicit-memory image-goal navigation).

Design (v1, "store the KV cache + recompute the window"):

The frozen LingBot-Map stream is precomputed once (offline, lingbot-map env) by
``scripts/dataset_converters/precompute_lingbot_features.py`` and cached next to
every trajectory as ``videos/chunk-000/lingbot_cache.npz``:

  * ``scale_k`` / ``scale_v``  [L, H, 8, P, d]      full K/V of the 8 scale frames
  * ``anchor_k`` / ``anchor_v`` [T-8, L, H, 6, d]    per-frame 6 special-token K/V
  * ``dino_cls`` [T, 1024]                           context-free DINOv2 CLS token

The memory is the LingBot stream anchored at frame 0.  At a sampled current step
``k`` (with sliding window ``W``) the memory partitions into three disjoint,
contiguous regions:

  scale   frames [0 .. 7]       full K/V        (from scale_k/v)
  history frames [8 .. k-W]     6 special only  (from anchor_k/v)
  window  frames [k-W+1 .. k]   recomputed live (raw RGB, in the policy)

The heavy K/V (scale ~1 GB/traj, anchor history ~100 MB) is therefore **not**
loaded here — that would blow up the collated batch.  This dataset is *light*:
it samples ``(k, k_goal)``, builds the NavDP-style action label + critic, reads
only the small ``dino_cls`` retrieval keys, and emits **pointers** so the policy
does the GPU work per sample.

What the policy does with the pointers (v1 architecture — three soft-gated goal
cross-attention readouts + raw goal descriptor):
  * loads the KV cache (``cache_path``, ``cur_step``) → GCT window-forward →
    current "post-GCA" state + compressed trajectory-memory readout.
  * retrieval is **`dino_cls`-only** for v1 (coarse, frame-level): ``goal_cls``
    vs ``mem_cls`` → top-k frames + retrieval/matchability gates.
  * **dense DINO is recomputed on the fly** (no dense storage): the policy loads
    raw RGB at the current window ``[k-7..k]``, the goal frame ``k_goal``, and the
    retrieved frame indices (from ``rgb_dir``) and re-runs the DINO forward for
    the in-FoV and revisit dense cross-attention.
So the dataset emits ``cache_path``, ``rgb_dir``, ``cur_step`` (= k), and
``goal_step`` (= k_goal); everything dense/GCT is computed GPU-side.

Multi-stop seen/unseen structure (within a single trajectory)
-------------------------------------------------------------
  * **seen**   (prob ``seen_ratio``): goal ``k_goal < k`` — already visited.
        Action label = reversed traversed path ``k -> k_goal``.
        ``retrieval_target = k_goal`` (its index in ``mem_cls[0..k]``).
  * **unseen** (prob ``1-seen_ratio``): goal ``k_goal > k`` — never observed.
        Action label = forward path ``k -> k_goal``.  ``retrieval_target = -1``.

We sample ``k >= num_scale + W`` so the scale / history / window regions stay
disjoint (trajectories here are 50-342 frames, so this is essentially always
available).
"""

import os

import numpy as np
import torch

from internnav.dataset.navdp_dataset_lerobot import NavDP_Base_Datset


class MemNav_Dataset(NavDP_Base_Datset):
    def __init__(
        self,
        root_dirs,
        preload_path=False,
        predict_size=24,
        batch_size=64,
        scene_data_scale=1.0,
        trajectory_data_scale=1.0,
        num_scale=8,
        window_size=8,
        seen_ratio=0.5,
        min_gap=4,
        pred_digit=4,
        random_digit=False,
        feature_filename='lingbot_cache.npz',
        rgb_subdir='videos/chunk-000/observation.images.rgb',
        lingbot_repo='/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map',
        image_size=518,
        patch_size=14,
        preprocess_mode='pad',
        repeat=1,
        debug=False,
        **kwargs,
    ):
        # We deliberately do NOT call super().__init__ — the base walk requires
        # depth dirs; memnav only needs parquet (poses), path.ply (critic), the
        # cache npz, and the rgb dir (raw frames for goal/window + lazy matches).
        self.predict_size = predict_size
        self.num_scale = num_scale
        self.window_size = window_size
        self.seen_ratio = seen_ratio
        self.min_gap = min_gap
        self.pred_digit = pred_digit
        self.random_digit = random_digit
        self.feature_filename = feature_filename
        self.rgb_subdir = rgb_subdir
        self.batch_size = batch_size
        self.debug = debug
        self.image_size = image_size
        self.patch_size = patch_size
        self.preprocess_mode = preprocess_mode

        # LingBot's exact image preprocessing (square-pad to image_size), so the
        # goal/window images here match what the GCT window-forward + dense DINO
        # forward expect on the GPU.
        import sys
        if lingbot_repo not in sys.path:
            sys.path.insert(0, lingbot_repo)
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        self._load_and_preprocess = load_and_preprocess_images

        self.trajectory_dirs = []
        self.trajectory_data_dir = []      # parquet (extrinsics / intrinsic)
        self.trajectory_afford_path = []   # path.ply (path + obstacle points)
        self.trajectory_feature_path = []  # lingbot_cache.npz
        self.trajectory_rgb_dir = []       # raw rgb frames dir (window recompute)

        for group_dir in sorted(p for p in os.listdir(root_dirs)):
            group_path = os.path.join(root_dirs, group_dir)
            if not os.path.isdir(group_path):
                continue
            all_scene = np.array(sorted(p for p in os.listdir(group_path)))
            if all_scene.shape[0] == 0:
                continue
            sel_scene = all_scene[np.arange(0, all_scene.shape[0], 1 / scene_data_scale).astype(np.int32)]
            for scene_dir in sel_scene:
                scene_path = os.path.join(group_path, scene_dir)
                if not os.path.isdir(scene_path):
                    continue
                all_traj = np.array(sorted(p for p in os.listdir(scene_path)))
                if all_traj.shape[0] == 0:
                    continue
                sel_traj = all_traj[np.arange(0, all_traj.shape[0], 1 / trajectory_data_scale).astype(np.int32)]
                for traj_dir in sel_traj:
                    entire_task_dir = os.path.join(scene_path, traj_dir)
                    if not os.path.isdir(entire_task_dir):
                        continue
                    data_path = os.path.join(entire_task_dir, 'data/chunk-000/episode_000000.parquet')
                    afford_path = os.path.join(entire_task_dir, 'data/chunk-000/path.ply')
                    feat_path = os.path.join(entire_task_dir, 'videos/chunk-000', self.feature_filename)
                    rgb_dir = os.path.join(entire_task_dir, self.rgb_subdir)
                    if not (os.path.isfile(data_path) and os.path.isfile(feat_path)):
                        continue
                    # need >= num_scale+window+1 frames so a valid current step (k >= lo) +
                    # a goal exist; cheap check via the frame at the minimum index.
                    min_frames = self.num_scale + self.window_size + 1
                    if not os.path.isfile(os.path.join(rgb_dir, f'{min_frames - 1}.jpg')):
                        continue
                    self.trajectory_dirs.append(entire_task_dir)
                    self.trajectory_data_dir.append(data_path)
                    self.trajectory_afford_path.append(afford_path)
                    self.trajectory_feature_path.append(feat_path)
                    self.trajectory_rgb_dir.append(rgb_dir)

        self.repeat = max(1, int(repeat))
        n = len(self.trajectory_dirs)
        print(f"[MemNav_Dataset] {n} trajectories with cached features under {root_dirs} (repeat={self.repeat})")
        if n == 0:
            raise RuntimeError(
                f"No trajectories with '{self.feature_filename}' found under {root_dirs}. "
                "Did you run scripts/dataset_converters/precompute_lingbot_features.py?"
            )

    def __len__(self):
        return len(self.trajectory_dirs) * self.repeat

    # ------------------------------------------------------------------ #
    # Light feature read (CLS only — np.load is lazy, scale_k/v untouched)
    # ------------------------------------------------------------------ #
    def _load_dino_cls(self, index):
        """Return dino_cls [T, 1024] float32 (the symmetric retrieval-key space)."""
        with np.load(self.trajectory_feature_path[index]) as data:
            return data['dino_cls'].astype(np.float32)

    def _load_images(self, rgb_dir, frame_indices):
        """Load + LingBot-preprocess RGB frames -> tensor [N, 3, H, W] in [0, 1]."""
        paths = [os.path.join(rgb_dir, f"{int(i)}.jpg") for i in frame_indices]
        return self._load_and_preprocess(
            paths, mode=self.preprocess_mode, image_size=self.image_size, patch_size=self.patch_size
        )

    # ------------------------------------------------------------------ #
    # seen / unseen step sampling
    # ------------------------------------------------------------------ #
    def _sample_steps(self, T):
        """Choose (mode_seen, k, k_goal). k >= num_scale+window so the
        scale/history/window regions stay disjoint."""
        g = self.min_gap
        k_min = min(self.num_scale + self.window_size, T - 1)  # 16 for 8/8

        def try_seen():
            if T - 1 < k_min:
                return None
            k = np.random.randint(k_min, T)
            g_high = k - g
            if g_high < 0:
                return None
            k_goal = np.random.randint(0, g_high + 1)
            return True, int(k), int(k_goal)

        def try_unseen():
            k_high = T - 1 - g
            if k_high < k_min:
                return None
            k = np.random.randint(k_min, k_high + 1)
            k_goal = np.random.randint(k + g, T)
            return False, int(k), int(k_goal)

        want_seen = np.random.rand() < self.seen_ratio
        order = (try_seen, try_unseen) if want_seen else (try_unseen, try_seen)
        for fn in order:
            res = fn()
            if res is not None:
                return res
        # Degenerate fallback (extremely short trajectory): trivial forward sample.
        k = max(k_min, T - 2)
        return False, int(k), int(T - 1)

    # ------------------------------------------------------------------ #
    # action label + goal-relative pose (no critic — collision is geometric at eval)
    # ------------------------------------------------------------------ #
    def _build_actions(self, extrinsics, base_extrinsic, pred_digit):
        """`extrinsics` = ordered segment current->goal (index 0 = current).
        Returns (pred_actions [predict_size,3] ×4 deltas, goal_rel_pose [3] = goal
        (x,y,θ) relative to current — GT for the revisit aux-pose head)."""
        L = extrinsics.shape[0]
        target_local_points, _, _, _, action_indexes = self.process_actions(
            extrinsics, base_extrinsic, 0, L - 1, pred_digit=pred_digit)
        init_vector = target_local_points[1] - target_local_points[0]
        target_xyt = self.xyz_to_xyt(target_local_points, init_vector)
        pred_xyt = target_xyt[action_indexes]                       # [predict_size+1, 3]
        # aux GT = the TRUE goal (full-path endpoint), not pred_xyt[-1] which is the
        # action-horizon-truncated waypoint (~96 steps) for long revisit segments.
        goal_rel_pose = target_xyt[-1].astype(np.float32).copy()    # goal pose rel to current
        pred_actions = (pred_xyt[1:] - pred_xyt[:-1]) * 4.0         # [predict_size, 3] deltas
        return pred_actions, goal_rel_pose

    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        index = index % len(self.trajectory_dirs)

        (
            _camera_intrinsic,
            base_extrinsic,
            extrinsics,            # [T_pq, 4, 4] camera-to-world per frame
            traj_len_parquet,
        ) = self.process_data_parquet(index)

        dino_cls = self._load_dino_cls(index)              # [T_f, 1024]
        T = int(min(traj_len_parquet, dino_cls.shape[0]))

        pred_digit = np.random.randint(2, 8) if self.random_digit else self.pred_digit

        mode_seen, k, k_goal = self._sample_steps(T)

        # --- retrieval keys: CLS of every observed frame [0..k] ---
        mem_cls = dino_cls[: k + 1].copy()                 # [k+1, 1024]
        goal_cls = dino_cls[k_goal].copy()                 # [1024]

        # --- action segment: ordered current(k) -> goal(k_goal) ---
        if mode_seen:
            seg = extrinsics[k_goal : k + 1][::-1].copy()  # reversed: seg[0] == current
            retrieval_target = int(k_goal)                  # index into mem_cls[0..k]
        else:
            seg = extrinsics[k : k_goal + 1].copy()
            retrieval_target = -1

        pred_actions, goal_rel_pose = self._build_actions(seg, base_extrinsic, pred_digit)

        # --- raw images (LingBot-preprocessed): always needed every sample ---
        rgb_dir = self.trajectory_rgb_dir[index]
        window_idx = list(range(k - self.window_size + 1, k + 1))  # [k-W+1 .. k]
        window_images = self._load_images(rgb_dir, window_idx)     # [W, 3, H, W]
        goal_image = self._load_images(rgb_dir, [k_goal])[0]       # [3, H, W]

        return {
            # light tensors
            'goal_cls': torch.tensor(goal_cls, dtype=torch.float32),
            'mem_cls': torch.tensor(mem_cls, dtype=torch.float32),
            'retrieval_target': torch.tensor(retrieval_target, dtype=torch.long),
            'is_seen': torch.tensor(float(mode_seen), dtype=torch.float32),
            'pred_actions': torch.tensor(pred_actions, dtype=torch.float32),
            'goal_rel_pose': torch.tensor(goal_rel_pose, dtype=torch.float32),   # aux pose GT
            # raw images for the on-the-fly dense DINO + GCT window-forward
            'goal_image': goal_image,                       # [3, H, W]
            'window_images': window_images,                 # [W, 3, H, W]  (current local window [k-W+1..k])
            # pointers for the policy (loaded on GPU, per sample):
            #   - KV cache (scale_k/v, anchor_k/v) for the GCT window-forward
            #   - rgb_dir = path to ALL historical frames: load any retrieved
            #     match's image lazily as rgb_dir/<idx>.jpg
            'cache_path': self.trajectory_feature_path[index],
            'rgb_dir': rgb_dir,
            'cur_step': int(k),
            'goal_step': int(k_goal),
        }


def memnav_collate_fn(batch):
    """Stack light tensors; pad variable-length mem_cls; keep pointers as lists."""
    B = len(batch)
    D = batch[0]['mem_cls'].shape[-1]
    lengths = [b['mem_cls'].shape[0] for b in batch]
    Lmax = max(lengths)

    mem_cls = torch.zeros(B, Lmax, D, dtype=torch.float32)
    mem_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, b in enumerate(batch):
        Li = lengths[i]
        mem_cls[i, :Li] = b['mem_cls']
        mem_mask[i, :Li] = True

    return {
        'batch_goal_cls':        torch.stack([b['goal_cls'] for b in batch]),
        'batch_mem_cls':         mem_cls,
        'batch_mem_mask':        mem_mask,
        'batch_retrieval_target':torch.stack([b['retrieval_target'] for b in batch]),
        'batch_is_seen':         torch.stack([b['is_seen'] for b in batch]),
        'batch_labels':          torch.stack([b['pred_actions'] for b in batch]),
        'batch_goal_rel_pose':   torch.stack([b['goal_rel_pose'] for b in batch]),   # aux pose GT
        # raw images
        'batch_goal_image':      torch.stack([b['goal_image'] for b in batch]),       # [B, 3, H, W]
        'batch_window_images':   torch.stack([b['window_images'] for b in batch]),    # [B, W, 3, H, W]
        # pointers (lists, length B) — the policy loads cache + lazy match frames per sample
        'cache_paths':           [b['cache_path'] for b in batch],
        'rgb_dirs':              [b['rgb_dir'] for b in batch],
        'cur_steps':             [b['cur_step'] for b in batch],
        'goal_steps':            [b['goal_step'] for b in batch],
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dirs", default="/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    ds = MemNav_Dataset(args.root_dirs, predict_size=24)
    print(f"len(dataset) = {len(ds)}")
    n_seen = 0
    for i in range(args.n):
        s = ds[i]
        seen = bool(s['is_seen'].item())
        n_seen += seen
        rt = s['retrieval_target'].item()
        k = s['cur_step']
        T_mem = s['mem_cls'].shape[0]
        if seen:
            assert 0 <= rt < T_mem, f"seen sample {i}: bad retrieval_target {rt} (mem len {T_mem})"
        else:
            assert rt == -1, f"unseen sample {i}: retrieval_target should be -1, got {rt}"
        if i < 6:
            print(
                f"[{i}] seen={seen} k={k} goal={s['goal_step']} ret_t={rt} "
                f"goal_cls={tuple(s['goal_cls'].shape)} mem_cls={tuple(s['mem_cls'].shape)} "
                f"labels={tuple(s['pred_actions'].shape)} "
                f"goal_rel_pose={[round(x,2) for x in s['goal_rel_pose'].tolist()]}"
            )
    print(f"seen fraction over {args.n}: {n_seen / args.n:.2f}")

    batch = memnav_collate_fn([ds[i] for i in range(8)])
    print("collated keys + shapes:")
    for key, v in batch.items():
        if torch.is_tensor(v):
            print(f"  {key}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {key}: list[{len(v)}] e.g. {v[0]}")
