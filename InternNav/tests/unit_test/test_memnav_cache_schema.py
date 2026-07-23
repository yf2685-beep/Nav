import unittest
from pathlib import Path
import tempfile

import numpy as np

from internnav.model.basemodel.memnav.cache_schema import (
    CACHE_SCHEMA_VERSION,
    FLOW_KEYFRAME_INTERVAL_SENTINEL,
    KEYFRAME_POLICY,
    KEYFRAME_POLICY_FLOW,
    auto_keyframe_interval,
    camera_keyframe_indices,
    post_scale_keyframe_indices,
    validate_cache_pair,
    validate_cache_files,
)


def _pair(num_frames=1329, scale=8, interval=5, window=32):
    anchor_indices = post_scale_keyframe_indices(num_frames, scale, interval)
    cam_indices = camera_keyframe_indices(num_frames, scale, interval)
    shared = {
        'cache_schema_version': np.array([CACHE_SCHEMA_VERSION]),
        'keyframe_policy': np.array([KEYFRAME_POLICY]),
        'num_frames': np.array([num_frames]),
        'num_scale_frames': np.array([scale]),
        'keyframe_interval': np.array([interval]),
        'precompute_signature': np.array(['unit-test']),
        'kv_cache_sliding_window': np.array([window]),
    }
    aggregator = {
        **shared,
        'meta': np.array([scale, 6, 1, 1, 1]),
        'dino_cls': np.zeros((num_frames, 2)),
        'anchor_k': np.zeros((len(anchor_indices), 1)),
        'anchor_v': np.zeros((len(anchor_indices), 1)),
        'anchor_frame_indices': anchor_indices,
    }
    camera = {
        **shared,
        'cam_pose_enc': np.zeros((num_frames, 9)),
        'cam_k': np.zeros((len(cam_indices), 1)),
        'cam_v': np.zeros((len(cam_indices), 1)),
        'cam_frame_indices': cam_indices,
    }
    return aggregator, camera


def _flow_pair(
    num_frames=60,
    scale=8,
    indices=(8, 11, 12, 20, 43, 59),
    flow_threshold=15.0,
    max_gap=30,
    window=32,
):
    anchor_indices = np.asarray(indices, dtype=np.int64)
    cam_indices = np.concatenate(
        (np.arange(scale, dtype=np.int64), anchor_indices)
    )
    shared = {
        'cache_schema_version': np.array([CACHE_SCHEMA_VERSION]),
        'keyframe_policy': np.array([KEYFRAME_POLICY_FLOW]),
        'num_frames': np.array([num_frames]),
        'num_scale_frames': np.array([scale]),
        'keyframe_interval': np.array([FLOW_KEYFRAME_INTERVAL_SENTINEL]),
        'flow_threshold': np.array([flow_threshold], dtype=np.float64),
        'max_non_keyframe_gap': np.array([max_gap], dtype=np.int64),
        'precompute_signature': np.array(['unit-test-flow']),
        'kv_cache_sliding_window': np.array([window]),
    }
    aggregator = {
        **shared,
        'meta': np.array([scale, 6, 1, 1, 1]),
        'dino_cls': np.zeros((num_frames, 2)),
        'anchor_k': np.zeros((len(anchor_indices), 1)),
        'anchor_v': np.zeros((len(anchor_indices), 1)),
        'anchor_frame_indices': anchor_indices,
    }
    camera = {
        **shared,
        'cam_pose_enc': np.zeros((num_frames, 9)),
        'cam_k': np.zeros((len(cam_indices), 1)),
        'cam_v': np.zeros((len(cam_indices), 1)),
        'cam_frame_indices': cam_indices,
    }
    return aggregator, camera


class MemNavCacheSchemaTest(unittest.TestCase):
    def test_official_auto_interval_and_indices(self):
        self.assertEqual(auto_keyframe_interval(320), 1)
        self.assertEqual(auto_keyframe_interval(321), 2)
        self.assertEqual(auto_keyframe_interval(1329), 5)
        np.testing.assert_array_equal(
            post_scale_keyframe_indices(20, 8, 5), [8, 13, 18]
        )
        np.testing.assert_array_equal(
            camera_keyframe_indices(20, 8, 5),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 18],
        )

    def test_versioned_sparse_pair_validates(self):
        aggregator, camera = _pair()
        layout = validate_cache_pair(
            aggregator,
            camera,
            expected_num_frames=1329,
            expected_num_scale_frames=8,
            expected_sliding_window=32,
            require_versioned=True,
        )
        self.assertFalse(layout.legacy_dense)
        self.assertEqual(layout.keyframe_interval, 5)
        self.assertEqual(layout.cam_frame_indices[-1], 1328)

    def test_mixed_or_shifted_cache_fails_closed(self):
        aggregator, camera = _pair()
        camera['precompute_signature'] = np.array(['different-run'])
        with self.assertRaisesRegex(ValueError, 'precompute_signature mismatch'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        aggregator, camera = _pair()
        aggregator['anchor_frame_indices'] = aggregator['anchor_frame_indices'] + 1
        with self.assertRaisesRegex(ValueError, 'anchor_frame_indices'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        aggregator, camera = _pair()
        aggregator['anchor_frame_indices'] = aggregator[
            'anchor_frame_indices'
        ].astype(np.float32)
        with self.assertRaisesRegex(ValueError, 'integer dtype'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_invalid_layout_scalars_fail_closed(self):
        for field, value, message in (
            ('num_frames', 0, 'num_frames must be positive'),
            ('num_scale_frames', 2000, 'num_scale_frames must be in'),
            ('keyframe_interval', 0, 'keyframe_interval must be positive'),
            ('precompute_signature', '', 'precompute_signature must be non-empty'),
        ):
            aggregator, camera = _pair()
            aggregator[field] = np.array([value])
            camera[field] = np.array([value])
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_legacy_dense_requires_explicit_opt_in(self):
        num_frames, scale = 12, 8
        aggregator = {
            'meta': np.array([scale, 6, 1, 1, 1]),
            'dino_cls': np.zeros((num_frames, 2)),
            'anchor_k': np.zeros((num_frames - scale, 1)),
            'anchor_v': np.zeros((num_frames - scale, 1)),
        }
        camera = {
            'cam_pose_enc': np.zeros((num_frames, 9)),
            'cam_k': np.zeros((num_frames, 1)),
            'cam_v': np.zeros((num_frames, 1)),
        }
        layout = validate_cache_pair(aggregator, camera)
        self.assertTrue(layout.legacy_dense)
        with self.assertRaisesRegex(ValueError, 'versioned LingBot cache required'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_flow_gate_pair_validates_and_exposes_gate_parameters(self):
        aggregator, camera = _flow_pair()
        layout = validate_cache_pair(
            aggregator,
            camera,
            expected_num_frames=60,
            expected_num_scale_frames=8,
            expected_sliding_window=32,
            require_versioned=True,
        )
        self.assertEqual(layout.keyframe_policy, KEYFRAME_POLICY_FLOW)
        self.assertEqual(layout.keyframe_interval, FLOW_KEYFRAME_INTERVAL_SENTINEL)
        self.assertEqual(layout.flow_threshold, 15.0)
        self.assertEqual(layout.max_non_keyframe_gap, 30)
        np.testing.assert_array_equal(
            layout.anchor_frame_indices, [8, 11, 12, 20, 43, 59]
        )
        self.assertEqual(layout.cam_frame_indices[-1], 59)

    def test_flow_gate_fails_closed(self):
        # not strictly increasing
        aggregator, camera = _flow_pair(indices=(8, 12, 12, 20))
        with self.assertRaisesRegex(ValueError, 'strictly increasing'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        # gap backstop violated
        aggregator, camera = _flow_pair(indices=(8, 11, 45))
        with self.assertRaisesRegex(ValueError, 'max_non_keyframe_gap'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        # first post-scale frame must be a keyframe
        aggregator, camera = _flow_pair(indices=(9, 11, 20))
        with self.assertRaisesRegex(ValueError, 'first[\\s\\S]*post-scale'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        # flow caches must store the interval sentinel
        aggregator, camera = _flow_pair()
        aggregator['keyframe_interval'] = np.array([3])
        camera['keyframe_interval'] = np.array([3])
        with self.assertRaisesRegex(ValueError, 'keyframe_interval='):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        # gate parameters are mandatory
        aggregator, camera = _flow_pair()
        del aggregator['flow_threshold']
        with self.assertRaisesRegex(ValueError, 'lacks.*flow_threshold'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

        # camera indices must be scale frames + the same anchor set
        aggregator, camera = _flow_pair()
        camera['cam_frame_indices'] = camera['cam_frame_indices'][:-1]
        with self.assertRaisesRegex(ValueError, 'cam_frame_indices'):
            validate_cache_pair(aggregator, camera, require_versioned=True)

    def test_flow_gate_file_preflight_roundtrip(self):
        aggregator, camera = _flow_pair()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregator_path = root / 'lingbot_cache.npz'
            camera_path = root / 'lingbot_cam_cache.npz'
            np.savez(aggregator_path, **aggregator)
            np.savez(camera_path, **camera)
            layout = validate_cache_files(
                aggregator_path,
                camera_path,
                expected_num_frames=60,
                expected_num_scale_frames=8,
                expected_sliding_window=32,
            )
            self.assertEqual(layout.keyframe_policy, KEYFRAME_POLICY_FLOW)
            self.assertEqual(layout.flow_threshold, 15.0)

    def test_file_preflight_reads_shapes_and_rejects_truncated_payload(self):
        aggregator, camera = _pair(num_frames=20, interval=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            aggregator_path = root / 'lingbot_cache.npz'
            camera_path = root / 'lingbot_cam_cache.npz'
            np.savez(aggregator_path, **aggregator)
            np.savez(camera_path, **camera)
            layout = validate_cache_files(
                aggregator_path,
                camera_path,
                expected_num_frames=20,
                expected_num_scale_frames=8,
                expected_sliding_window=32,
            )
            self.assertEqual(layout.keyframe_interval, 2)

            camera['cam_v'] = camera['cam_v'][:-1]
            np.savez(camera_path, **camera)
            with self.assertRaisesRegex(ValueError, 'payload/index length mismatch'):
                validate_cache_files(aggregator_path, camera_path)


if __name__ == '__main__':
    unittest.main()
