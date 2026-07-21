"""MemNav_Agent — owns the MemNavInference engine + per-env streaming state.

The NavDP/IsaacSim eval sends, each step, the current RGB (+ depth, unused here) and the
goal RGB. This agent keeps a per-env buffer of LingBot-preprocessed frames, appends the
current frame, and asks the inference engine for a local trajectory. During warmup
(fewer than `lo` = num_scale+window-1 frames) it returns a gentle forward nudge so the
robot moves and accumulates observations.

Frame preprocessing reuses LingBot's `load_and_preprocess_images` (via a small temp jpg
per env) so it is identical to the precompute/training preprocessing.
"""
import os
import tempfile

import numpy as np
import torch
from PIL import Image

from memnav_infer import MemNavInference


class MemNav_Agent:
    def __init__(self, intrinsic, checkpoint, lingbot_repo, lingbot_weights,
                 predict_size=24, sample_num=4, device="cuda:0"):
        self.engine = MemNavInference(
            checkpoint=checkpoint, lingbot_repo=lingbot_repo, lingbot_weights=lingbot_weights,
            predict_size=predict_size, device=device)
        self.intrinsic = intrinsic
        self.predict_size = predict_size
        self.sample_num = sample_num
        self.device = device
        self.tmpdir = tempfile.mkdtemp(prefix="memnav_agent_")
        self.batch_size = 1
        self.buffers = []      # per-env list of [3,H,W] preprocessed tensors
        self.goals = []        # per-env preprocessed goal [3,H,W] (or None)

    # ------------------------------------------------------------------ #
    def reset(self, batch_size, threshold=None):
        self.batch_size = int(batch_size)
        self.buffers = [[] for _ in range(self.batch_size)]
        self.goals = [None for _ in range(self.batch_size)]

    def reset_env(self, env_id):
        self.buffers[env_id] = []
        self.goals[env_id] = None

    # ------------------------------------------------------------------ #
    def _preprocess(self, rgb_np, tag):
        """rgb_np [H,W,3] uint8 RGB -> LingBot-preprocessed [3,518,518] (matches training)."""
        path = os.path.join(self.tmpdir, f"{tag}.jpg")
        Image.fromarray(rgb_np.astype(np.uint8)).save(path, quality=95)
        img = self.engine.net.lingbot.load_images([path])   # [1,3,518,518]
        return img[0]

    def _warmup_traj(self):
        """Gentle forward nudge (local frame: +x forward) to gather frames."""
        steps = np.arange(1, self.predict_size + 1, dtype=np.float32)
        xy = np.stack([0.05 * steps, np.zeros_like(steps), np.zeros_like(steps)], axis=1)  # [P,3]
        return xy

    # ------------------------------------------------------------------ #
    def step_imagegoal(self, goal_images, rgb_images):
        """goal_images [B,Hg,Wg,3], rgb_images [B,H,W,3] (RGB uint8).
        Returns execute_traj [B,P,3], all_traj [B,S,P,3], all_values [B,S]."""
        B = self.batch_size
        exec_traj = np.zeros((B, self.predict_size, 3), dtype=np.float32)
        all_traj = np.zeros((B, self.sample_num, self.predict_size, 3), dtype=np.float32)
        all_values = np.zeros((B, self.sample_num), dtype=np.float32)

        for b in range(B):
            if self.goals[b] is None:
                self.goals[b] = self._preprocess(goal_images[b], f"goal_{b}")
            cur = self._preprocess(rgb_images[b], f"cur_{b}")
            self.buffers[b].append(cur)

            try:
                traj, _match = self.engine.predict(self.buffers[b], self.goals[b],
                                                   sample_num=self.sample_num)
            except Exception as e:                  # transient CUDA/inference error -> don't crash the server
                print(f"[MemNav_Agent] predict error (env {b}): {type(e).__name__}: {e}", flush=True)
                traj = None
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            if traj is None:                        # warmup or recovered error
                w = self._warmup_traj()
                exec_traj[b] = w
                all_traj[b] = np.repeat(w[None], self.sample_num, axis=0)
            else:
                exec_traj[b] = traj[0]              # sample 0 executed (no critic to rank)
                all_traj[b] = traj
        torch.cuda.empty_cache()                    # keep GPU footprint low between steps
        return exec_traj, all_traj, all_values
