"""Pose conventions shared by generated MemNav datasets and their loader."""

import numpy as np


GENERATED_ZUP_FRAME_CONVENTION = "positions+parquet in data(Zup,M_W)"

# Habitat (Y-up) -> generated data world (Z-up): (x, y, z) -> (x, -z, y).
HABITAT_TO_DATA_ROTATION = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def generated_camera_extrinsic(camera_height_m=0.5):
    """Camera mount expected by NavDP for generated Z-up camera-to-world poses."""
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = HABITAT_TO_DATA_ROTATION
    extrinsic[:3, 3] = HABITAT_TO_DATA_ROTATION @ np.array(
        [0.0, float(camera_height_m), 0.0], dtype=np.float64
    )
    return extrinsic


def wrap_radians(angle):
    """Wrap scalar/array angles to the half-open interval ``[-pi, pi)``."""
    angle = np.asarray(angle)
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def resolve_memnav_base_extrinsic(recorded_extrinsic, frame_convention):
    """Upgrade legacy generated episodes without changing unrelated datasets.

    Legacy ``generate_twoleg.py`` wrote a full camera-to-world rotation to
    ``action`` but recorded an identity camera mount. NavDP removes the mount via
    ``R_action @ inv(R_mount)`` before forming planar labels, so identity leaves
    the optical forward axis in coordinate 2, which ``xyz_to_xyt`` then drops.

    Existing pt1/pt2 episodes carry an explicit generated-frame marker. For only
    those episodes, replace the legacy identity rotation with the known mount.
    Newly generated episodes already contain that mount and pass through as-is.
    """
    extrinsic = np.asarray(recorded_extrinsic, dtype=np.float64)
    if extrinsic.shape != (4, 4):
        raise ValueError(f"camera extrinsic must be 4x4, got {extrinsic.shape}")

    if not str(frame_convention or "").startswith(GENERATED_ZUP_FRAME_CONVENTION):
        return extrinsic.copy()

    rotation = extrinsic[:3, :3]
    if np.allclose(rotation, HABITAT_TO_DATA_ROTATION, atol=1e-6):
        return extrinsic.copy()
    if not np.allclose(rotation, np.eye(3), atol=1e-6):
        raise ValueError(
            "generated Z-up episode has an unsupported camera mount rotation; "
            "expected legacy identity or Habitat-to-data M_W"
        )

    corrected = extrinsic.copy()
    corrected[:3, :3] = HABITAT_TO_DATA_ROTATION
    return corrected
