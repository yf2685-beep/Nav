from pathlib import Path

import pytest

from scripts.dataset_converters.precompute_lingbot_features import (
    select_trajectories,
    validate_frame_capacity,
)


def _trajectory(root: Path, relative: str, frame_count: int):
    episode = root / relative
    rgb_dir = episode / "videos/chunk-000/observation.images.rgb"
    return str(episode), str(rgb_dir), [f"{index}.jpg" for index in range(frame_count)]


def test_trajectory_manifest_preserves_order_and_accepts_diagnostic_suffix(tmp_path):
    root = tmp_path / "data"
    trajectories = [
        _trajectory(root, "group/scene/episode_0001", 10),
        _trajectory(root, "group/scene/episode_0002", 20),
    ]
    manifest = tmp_path / "missing.txt"
    manifest.write_text(
        "group/scene/episode_0002\tlingbot_cache.npz,lingbot_cam_cache.npz\n"
        "group/scene/episode_0001\n",
        encoding="utf-8",
    )

    selected = select_trajectories(trajectories, str(root), str(manifest))

    assert [Path(item[0]).name for item in selected] == ["episode_0002", "episode_0001"]


def test_frame_capacity_fails_before_truncated_temporal_rope(tmp_path):
    trajectories = [
        _trajectory(tmp_path, "group/scene/short", 2048),
        _trajectory(tmp_path, "group/scene/long", 3954),
    ]

    with pytest.raises(ValueError, match=r"max_frame_num=2048.*at least 3954"):
        validate_frame_capacity(trajectories, 2048)

    validate_frame_capacity(trajectories, 4096)
