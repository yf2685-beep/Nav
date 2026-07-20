"""Streaming LingBot-Map pose estimator + simple PGO loop-closure correction.

Phase 1 (Phase A): returns raw LingBot pose per frame.
Phase 2 (Phase B): adds loop-closure detection over a visual-descriptor buffer
                   and applies a snap correction to the alignment offset when a
                   revisit is detected (lightweight PGO that updates only the
                   global alignment, not the full pose graph).

The eval client owns the rigid alignment from LingBot world to sim world.
This module only exposes:
  - estimate(rgb)            -> raw pose_4x4 in LingBot's internal world frame
  - get_descriptor()          -> visual fingerprint of the most recent frame
                                 (used for loop closure detection)
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/nyuair/yuxuan/1 robot navigation/lingbot-map')

from lingbot_map.models.gct_stream_window import GCTStream
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri


class LingBotPoseEstimator:
    """Wraps GCTStream for streaming use inside the eval client.

    Preprocessing matches the official `load_and_preprocess_images(mode='crop')`:
      - keep aspect ratio, set width = image_size, snap height to a patch_size
        multiple, center-crop height to image_size if it overflows.
    """
    def __init__(
        self,
        ckpt: str = '/home/nyuair/data-001/lingbot-map-ckpt/lingbot-map.pt',
        window_size: int = 32,
        image_size: int = 518,
        patch_size: int = 14,
        device: str = 'cuda:0',
        dtype=torch.bfloat16,
    ):
        self.window_size = window_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = device
        self.dtype = dtype

        self.model = GCTStream(
            img_size=image_size,
            patch_size=patch_size,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=window_size,
            kv_cache_scale_frames=8,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,
            camera_num_iterations=4,
        ).to(device)
        self.model.eval()
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        res = self.model.load_state_dict(sd, strict=False)
        print(f'[LingBotPoseEstimator] loaded ckpt: missing={len(res.missing_keys)} '
              f'unexpected={len(res.unexpected_keys)}')
        self.model.aggregator = self.model.aggregator.to(dtype)

        self._frames = []      # list of preprocessed (1, 3, H, W) tensors
        self._last_descriptor = None

    def _preprocess(self, rgb):
        """Match official mode='crop' preprocessing.

        rgb: (H, W, 3) uint8 or float in [0, 255] or [0, 1].
        Output: (1, 3, h, w) in [0, 1], width=image_size, height=patch-multiple.
        """
        if rgb.dtype != np.float32:
            rgb = rgb.astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        # (H, W, 3) -> (1, 3, H, W)
        t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        _, _, H, W = t.shape
        # Match official: new_width = image_size, new_height keeps aspect ratio
        # rounded to patch_size multiple.
        new_width = self.image_size
        new_height = round(H * (new_width / W) / self.patch_size) * self.patch_size
        t = F.interpolate(t, size=(new_height, new_width), mode='bicubic', align_corners=False)
        # Center crop height if overflows.
        if new_height > self.image_size:
            sh = (new_height - self.image_size) // 2
            t = t[:, :, sh:sh+self.image_size, :]
        t = t.clamp(0.0, 1.0)
        return t

    @torch.no_grad()
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """Take a new RGB frame, return latest camera-to-world 4x4 pose (np.float32).

        Pose is in LingBot's internal world frame (first frame in the window =
        identity). The caller must apply rigid alignment to the sim world frame.

        Side effect: updates self._last_descriptor with a visual fingerprint
        derived from the input.
        """
        frame = self._preprocess(rgb).to(self.device)  # (1, 3, h, w)
        self._frames.append(frame)
        if len(self._frames) > self.window_size:
            self._frames.pop(0)

        seq = torch.cat(self._frames, dim=0).unsqueeze(0).to(self.dtype)
        H, W = seq.shape[-2], seq.shape[-1]

        preds = self.model.inference_windowed(
            seq,
            window_size=min(self.window_size, len(self._frames)),
            overlap_size=0,
            overlap_keyframes=None,
        )
        pose_enc = preds['pose_enc']                       # (1, N, 9)
        extr, _ = pose_encoding_to_extri_intri(pose_enc, (H, W))
        latest = extr[0, -1].float().cpu().numpy()         # (3, 4) cam-to-world
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :4] = latest

        # Visual descriptor for loop closure: downsample the input frame and
        # L2-normalize. Simple but cheap; doesn't depend on LingBot internals.
        with torch.no_grad():
            ds = F.adaptive_avg_pool2d(frame, output_size=(8, 8))  # (1, 3, 8, 8)
            self._last_descriptor = (
                F.normalize(ds.flatten(1), dim=1)[0].float().cpu().numpy()
            )
        return pose

    def get_descriptor(self) -> np.ndarray:
        return self._last_descriptor

    def reset(self):
        self._frames.clear()
        self._last_descriptor = None


class PGOCorrector:
    """Loop-closure snap correction (simplified PGO).

    Keeps a buffer of (sim-world-aligned pose, visual descriptor) per frame.
    When a new frame's descriptor matches one in the history (cosine > thresh)
    AND the two LingBot poses are far apart (drift accumulated), it computes
    a rigid snap correction and updates the alignment offset so the *current*
    pose is forced to match the *historical* pose.

    This is a simplified one-shot PGO that only corrects the global alignment
    (not per-edge minimization).  Sufficient as MVP; can upgrade to full
    Gauss-Newton later.
    """
    def __init__(
        self,
        sim_cosine_threshold: float = 0.992,
        min_drift_meters: float = 0.3,
        min_frames_gap: int = 25,
        max_buffer: int = 400,
    ):
        self.sim_cosine_threshold = sim_cosine_threshold
        self.min_drift_meters = min_drift_meters
        self.min_frames_gap = min_frames_gap
        self.max_buffer = max_buffer

        self.history = []   # list of dicts {step, aligned_pose_4x4, descriptor}
        self.last_correction_step = -1_000_000

    def add(self, step: int, aligned_pose_4x4: np.ndarray, descriptor: np.ndarray):
        self.history.append({
            'step': step,
            'pose': aligned_pose_4x4.copy(),
            'desc': descriptor.copy(),
        })
        if len(self.history) > self.max_buffer:
            self.history.pop(0)

    def detect_and_correct(
        self,
        step: int,
        current_aligned_pose: np.ndarray,
        current_descriptor: np.ndarray,
    ):
        """Return (closure_triggered, snap_correction_4x4).

        snap_correction_4x4 is the rigid transform you should pre-multiply with
        the current alignment offset so the corrected pose matches the matched
        historical pose. If no closure, returns (False, identity).
        """
        I4 = np.eye(4, dtype=np.float32)
        if len(self.history) == 0 or current_descriptor is None:
            return False, I4
        # Don't trigger again too soon.
        if step - self.last_correction_step < self.min_frames_gap:
            return False, I4

        # Cosine similarity against all stored descriptors.
        descs = np.stack([h['desc'] for h in self.history], axis=0)   # (M, D)
        cur = current_descriptor
        sims = descs @ cur                                            # (M,)
        # Restrict to entries older than min_frames_gap.
        eligible = np.array([h['step'] <= step - self.min_frames_gap for h in self.history])
        sims_eligible = np.where(eligible, sims, -np.inf)
        best_idx = int(np.argmax(sims_eligible))
        best_sim = float(sims_eligible[best_idx])
        if best_sim < self.sim_cosine_threshold:
            return False, I4

        matched = self.history[best_idx]
        # Require physical drift: the new sim-aligned pose's xy should be far
        # enough from the matched one's xy, otherwise we're just inside the
        # window with no real drift to correct.
        drift_xy = np.linalg.norm(current_aligned_pose[:2, 3] - matched['pose'][:2, 3])
        if drift_xy < self.min_drift_meters:
            return False, I4

        # Snap: we want corrected_pose = matched['pose'].
        # corrected = snap @ current_aligned_pose => snap = matched @ inv(current)
        snap = matched['pose'] @ np.linalg.inv(current_aligned_pose)
        self.last_correction_step = step
        return True, snap.astype(np.float32)

    def reset(self):
        self.history.clear()
        self.last_correction_step = -1_000_000
