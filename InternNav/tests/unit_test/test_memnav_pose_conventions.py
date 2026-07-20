import unittest

import numpy as np

from internnav.dataset.memnav_pose_conventions import (
    GENERATED_ZUP_FRAME_CONVENTION,
    HABITAT_TO_DATA_ROTATION,
    generated_camera_extrinsic,
    resolve_memnav_base_extrinsic,
    wrap_radians,
)


def _habitat_yaw(yaw):
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    )


def _navdp_local_translation(action_rotation, world_delta, mount_rotation):
    base_rotation = action_rotation @ np.linalg.inv(mount_rotation)
    raw_local = base_rotation.T @ world_delta
    return np.array([raw_local[1], -raw_local[0], raw_local[2]])


class MemNavPoseConventionTest(unittest.TestCase):
    def test_generated_mount_contains_axis_rotation_and_camera_height(self):
        extrinsic = generated_camera_extrinsic(camera_height_m=0.5)
        np.testing.assert_allclose(extrinsic[:3, :3], HABITAT_TO_DATA_ROTATION)
        np.testing.assert_allclose(extrinsic[:3, 3], [0.0, 0.0, 0.5])

    def test_legacy_identity_mount_is_upgraded_only_for_generated_data(self):
        legacy = np.eye(4)
        corrected = resolve_memnav_base_extrinsic(
            legacy, GENERATED_ZUP_FRAME_CONVENTION + "; yaw_habitat in render frame"
        )
        np.testing.assert_allclose(corrected[:3, :3], HABITAT_TO_DATA_ROTATION)
        np.testing.assert_allclose(
            resolve_memnav_base_extrinsic(legacy, "unrelated dataset"), legacy
        )

    def test_new_generated_mount_passes_through_and_unknown_mount_fails(self):
        fixed = generated_camera_extrinsic(camera_height_m=0.7)
        np.testing.assert_allclose(
            resolve_memnav_base_extrinsic(fixed, GENERATED_ZUP_FRAME_CONVENTION), fixed
        )

        unsupported = np.eye(4)
        unsupported[:3, :3] = np.diag([-1.0, -1.0, 1.0])
        with self.assertRaisesRegex(ValueError, "unsupported camera mount rotation"):
            resolve_memnav_base_extrinsic(
                unsupported, GENERATED_ZUP_FRAME_CONVENTION
            )

    def test_generated_forward_motion_stays_in_navdp_xy_plane(self):
        for yaw in [0.0, np.pi / 2, -np.pi / 2, np.pi]:
            with self.subTest(yaw=yaw):
                rotation_habitat = _habitat_yaw(yaw)
                action_rotation = HABITAT_TO_DATA_ROTATION @ rotation_habitat
                forward_habitat = rotation_habitat @ np.array([0.0, 0.0, -1.0])
                world_delta = HABITAT_TO_DATA_ROTATION @ forward_habitat

                fixed_local = _navdp_local_translation(
                    action_rotation, world_delta, HABITAT_TO_DATA_ROTATION
                )
                legacy_local = _navdp_local_translation(
                    action_rotation, world_delta, np.eye(3)
                )

                np.testing.assert_allclose(fixed_local, [1.0, 0.0, 0.0], atol=1e-7)
                # Identity leaves forward motion in coordinate 2, while
                # xyz_to_xyt consumes only coordinates 0 and 1.
                np.testing.assert_allclose(legacy_local[:2], [0.0, 0.0], atol=1e-7)
                self.assertAlmostEqual(abs(legacy_local[2]), 1.0)

    def test_corrected_coordinates_recover_both_horizontal_axes(self):
        action_rotation = HABITAT_TO_DATA_ROTATION
        delta_habitat = np.array([1.0, 0.2, -2.0])
        world_delta = HABITAT_TO_DATA_ROTATION @ delta_habitat

        legacy_local = _navdp_local_translation(
            action_rotation, world_delta, np.eye(3)
        )
        fixed_local = _navdp_local_translation(
            action_rotation, world_delta, HABITAT_TO_DATA_ROTATION
        )

        np.testing.assert_allclose(legacy_local, [0.2, -1.0, -2.0])
        np.testing.assert_allclose(fixed_local, [2.0, -1.0, 0.2])
        np.testing.assert_allclose(
            fixed_local, [-legacy_local[2], legacy_local[1], legacy_local[0]]
        )
        self.assertAlmostEqual(
            np.linalg.norm(fixed_local[:2]),
            np.linalg.norm(delta_habitat[[0, 2]]),
        )

    def test_angle_wrap_removes_atan2_branch_cut_jump(self):
        angles = np.array([
            (-np.pi + 0.01) - (np.pi - 0.01),
            (np.pi - 0.02) - (-np.pi + 0.02),
            0.25,
        ])
        np.testing.assert_allclose(wrap_radians(angles), [0.02, -0.04, 0.25])


if __name__ == "__main__":
    unittest.main()
