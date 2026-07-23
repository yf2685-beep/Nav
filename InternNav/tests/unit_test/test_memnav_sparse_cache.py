import unittest

import torch
import torch.nn as nn

from internnav.model.basemodel.memnav.cache_schema import camera_keyframe_indices
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream


class _FakeCameraHead:
    num_iterations = 1
    trunk_depth = 1

    def __init__(self):
        self.kv_cache = []
        self.frame_idx = 0

    def clean_kv_cache(self):
        self.kv_cache = []
        self.frame_idx = 0


class _FakeAggregator:
    def __init__(self):
        self.kv_cache = {}
        self.total_frames_processed = 0


class _FakeModel:
    def __init__(self):
        self.aggregator = _FakeAggregator()
        self.camera_head = _FakeCameraHead()

    def clean_kv_cache(self):
        self.aggregator.kv_cache = {}
        self.aggregator.total_frames_processed = 0
        self.camera_head.clean_kv_cache()


def _stream():
    stream = LingBotStream.__new__(LingBotStream)
    nn.Module.__init__(stream)
    stream.model = _FakeModel()
    stream.agg = stream.model.aggregator
    stream.depth = 1
    stream.num_scale = 8
    return stream


class MemNavSparseInjectionTest(unittest.TestCase):
    def test_aggregator_uses_sparse_count_not_raw_frame_as_next_time(self):
        stream = _stream()
        scale_k = torch.zeros(1, 1, 8, 1, 1)
        scale_v = torch.zeros_like(scale_k)
        anchor_indices = torch.tensor([8, 13, 18, 23])
        anchor_k = torch.arange(4.0).reshape(1, 1, 4, 1, 1).expand(-1, -1, -1, 6, -1)
        anchor_v = anchor_k.clone()

        stream._inject(
            scale_k,
            scale_v,
            anchor_k,
            anchor_v,
            anchor_frame_indices=anchor_indices,
            raw_start=20,
        )

        self.assertEqual(stream.agg.kv_cache['k_0_special'].shape[2], 3)
        # Official aggregator time advances only for the eight scale frames and
        # the three appended keyframes before raw frame 20.
        self.assertEqual(stream.agg.total_frames_processed, 11)

    def test_camera_uses_sparse_rows_but_preserves_raw_rope_time(self):
        stream = _stream()
        frame_indices = torch.as_tensor(camera_keyframe_indices(30, 8, 5))
        cam_k = torch.zeros(len(frame_indices), 1, 1, 2, 3)
        cam_v = torch.zeros_like(cam_k)

        stream._inject_camera(
            cam_k, cam_v, n=20, cam_frame_indices=frame_indices
        )

        # [0..7, 8, 13, 18] are injected: eleven KV rows.
        self.assertEqual(stream.model.camera_head.kv_cache[0]['k_0'].shape[2], 11)
        # The newly appended goal is still raw temporal frame 20, not sparse row 11.
        self.assertEqual(stream.model.camera_head.frame_idx, 20)

    def test_zero_goal_warm_is_exact_sparse_prefix_without_dense_replay(self):
        stream = _stream()
        scale_k = torch.zeros(1, 1, 8, 1, 1)
        scale_v = torch.zeros_like(scale_k)
        anchor_indices = torch.tensor([8, 13, 18, 23])
        anchor_k = torch.zeros(1, 1, 4, 6, 1)
        anchor_v = torch.zeros_like(anchor_k)
        cache = {
            'scale_k': scale_k,
            'scale_v': scale_v,
            'anchor_k': anchor_k,
            'anchor_v': anchor_v,
            'anchor_frame_indices': anchor_indices,
        }
        stream.load_images = lambda _paths: self.fail(
            'warm=0 must not load or replay raw frames'
        )
        stream._stream_one = lambda _goal, return_agg=False: ('goal', return_agg)

        result = stream.goal_append_warm(
            torch.zeros(3, 4, 4), cache, m=20, rgb_dir='/unused', warm=0,
            return_agg=True,
        )

        self.assertEqual(result, ('goal', True))
        self.assertEqual(stream.agg.kv_cache['k_0_special'].shape[2], 3)
        self.assertEqual(stream.agg.total_frames_processed, 11)


if __name__ == '__main__':
    unittest.main()
