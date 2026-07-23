"""Versioned LingBot cache layout for dense per-frame outputs and sparse KV memory.

LingBot's keyframe policy still predicts a pose and a DINO descriptor for every
raw video frame.  It only suppresses KV-cache appends for non-keyframes.  A
cache therefore has two different timelines which must never be conflated:

* dense frame outputs: ``dino_cls`` and ``cam_pose_enc`` have ``num_frames`` rows;
* sparse memory: camera/aggregator KVs have scale frames plus selected keyframes.

The explicit raw-frame index arrays below make that distinction auditable and
let runtime injection set camera RoPE time from the raw frame number while
selecting only the available sparse KVs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping
import zipfile

import numpy as np


CACHE_SCHEMA_VERSION = 2
KEYFRAME_POLICY = "post_scale_mod_v1"
# Flow-gated keyframes: selection is causal (reprojection flow vs the last
# committed keyframe, upstream gct_stream_window semantics), so the indices are
# irregular and cannot be re-derived from (num_frames, interval).  The cache
# stores keyframe_interval=0 as an explicit sentinel plus the gate parameters,
# and validation checks structural invariants instead of an arithmetic pattern.
KEYFRAME_POLICY_FLOW = "flow_gate_v1"
FLOW_KEYFRAME_INTERVAL_SENTINEL = 0
DEFAULT_KEYFRAME_BUDGET = 320


def auto_keyframe_interval(num_frames: int, budget: int = DEFAULT_KEYFRAME_BUDGET) -> int:
    """Official LingBot heuristic: keep at most roughly ``budget`` temporal views."""
    if int(num_frames) < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if int(budget) < 1:
        raise ValueError(f"keyframe budget must be positive, got {budget}")
    return max(1, math.ceil(int(num_frames) / int(budget)))


def post_scale_keyframe_indices(
    num_frames: int,
    num_scale_frames: int,
    keyframe_interval: int,
) -> np.ndarray:
    """Raw indices whose post-scale KVs are appended by LingBot streaming."""
    num_frames = int(num_frames)
    num_scale_frames = min(int(num_scale_frames), num_frames)
    keyframe_interval = int(keyframe_interval)
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if num_scale_frames < 1:
        raise ValueError(
            f"num_scale_frames must be positive, got {num_scale_frames}"
        )
    if keyframe_interval < 1:
        raise ValueError(
            f"keyframe_interval must be positive, got {keyframe_interval}"
        )
    return np.arange(
        num_scale_frames, num_frames, keyframe_interval, dtype=np.int64
    )


def camera_keyframe_indices(
    num_frames: int,
    num_scale_frames: int,
    keyframe_interval: int,
) -> np.ndarray:
    """Raw indices represented in the camera-head KV cache."""
    scale = min(int(num_scale_frames), int(num_frames))
    return np.concatenate(
        (
            np.arange(scale, dtype=np.int64),
            post_scale_keyframe_indices(num_frames, scale, keyframe_interval),
        )
    )


def _scalar(mapping: Mapping[str, np.ndarray], name: str, cast):
    value = np.asarray(mapping[name])
    if value.size != 1:
        raise ValueError(f"cache field {name!r} must be scalar, got shape={value.shape}")
    return cast(value.reshape(-1)[0])


def _require_equal(name: str, actual, expected) -> None:
    if actual != expected:
        raise ValueError(f"cache {name} mismatch: got {actual!r}, expected {expected!r}")


def _index_array(name: str, value) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"cache {name} must be one-dimensional, got {raw.shape}")
    if raw.dtype.kind not in "iu":
        raise ValueError(
            f"cache {name} must use an integer dtype, got {raw.dtype}"
        )
    return raw.astype(np.int64, copy=False)


def _strict_indices(
    name: str, value, expected: np.ndarray, policy: str = KEYFRAME_POLICY
) -> np.ndarray:
    indices = _index_array(name, value)
    if not np.array_equal(indices, expected):
        first = next(
            (
                i
                for i, pair in enumerate(zip(indices.tolist(), expected.tolist()))
                if pair[0] != pair[1]
            ),
            min(len(indices), len(expected)),
        )
        raise ValueError(
            f"cache {name} does not match {policy}: "
            f"rows={len(indices)} expected={len(expected)} first_difference={first}"
        )
    return indices


def _flow_gate_indices(
    name: str,
    value,
    *,
    num_scale_frames: int,
    num_frames: int,
    max_non_keyframe_gap: int,
) -> np.ndarray:
    """Structural validation for causally-selected (irregular) keyframe indices.

    The gate guarantees, by construction: the first post-scale frame is always a
    keyframe, indices are strictly increasing raw frame numbers, and no two
    consecutive keyframes are more than ``max_non_keyframe_gap`` frames apart
    (the gap backstop forces a keyframe).  Anything else is a corrupted or
    mislabeled cache.
    """
    indices = _index_array(name, value)
    if num_frames <= num_scale_frames:
        if len(indices):
            raise ValueError(
                f"cache {name} must be empty when every frame is a scale frame"
            )
        return indices
    if len(indices) == 0 or int(indices[0]) != int(num_scale_frames):
        raise ValueError(
            f"cache {name} does not match {KEYFRAME_POLICY_FLOW}: the first "
            f"post-scale frame ({num_scale_frames}) must be a keyframe, got "
            f"{indices[:1].tolist()}"
        )
    if int(indices[-1]) >= int(num_frames):
        raise ValueError(
            f"cache {name} exceeds num_frames={num_frames}: last={int(indices[-1])}"
        )
    gaps = np.diff(indices)
    if len(gaps) and int(gaps.min()) < 1:
        raise ValueError(f"cache {name} must be strictly increasing")
    if len(gaps) and int(gaps.max()) > int(max_non_keyframe_gap):
        raise ValueError(
            f"cache {name} violates max_non_keyframe_gap={max_non_keyframe_gap}: "
            f"largest gap={int(gaps.max())}"
        )
    return indices


@dataclass(frozen=True)
class CacheLayout:
    schema_version: int
    num_frames: int
    num_scale_frames: int
    keyframe_interval: int
    anchor_frame_indices: np.ndarray
    cam_frame_indices: np.ndarray
    precompute_signature: str
    legacy_dense: bool = False
    keyframe_policy: str = KEYFRAME_POLICY
    # flow_gate_v1 only (0 otherwise): gate parameters the indices were selected with.
    flow_threshold: float = 0.0
    max_non_keyframe_gap: int = 0


_SCHEMA_FIELDS = {
    "cache_schema_version",
    "keyframe_policy",
    "num_frames",
    "num_scale_frames",
    "keyframe_interval",
    "precompute_signature",
}


def validate_cache_pair(
    aggregator: Mapping[str, np.ndarray],
    camera: Mapping[str, np.ndarray],
    *,
    expected_num_frames: int | None = None,
    expected_num_scale_frames: int | None = None,
    expected_sliding_window: int | None = None,
    require_versioned: bool = False,
    validate_payload: bool = True,
) -> CacheLayout:
    """Validate paired aggregator/camera caches and return their raw-index layout.

    Old dense caches remain readable when ``require_versioned`` is false.  Sparse
    training must set it true so a partially regenerated or mixed cache fails
    before model construction instead of silently shifting temporal indices.
    """
    agg_keys = set(aggregator.keys())
    cam_keys = set(camera.keys())
    agg_versioned = bool(_SCHEMA_FIELDS & agg_keys)
    cam_versioned = bool(_SCHEMA_FIELDS & cam_keys)
    if agg_versioned != cam_versioned:
        raise ValueError("aggregator and camera caches use different schema generations")

    if not agg_versioned:
        if require_versioned:
            raise ValueError("versioned LingBot cache required but legacy dense cache found")
        if "meta" not in aggregator:
            raise ValueError("legacy aggregator cache lacks meta")
        scale = int(np.asarray(aggregator["meta"]).reshape(-1)[0])
        num_frames = int(np.asarray(camera["cam_pose_enc"]).shape[0])
        if expected_num_frames is not None:
            _require_equal("num_frames", num_frames, int(expected_num_frames))
        if expected_num_scale_frames is not None:
            _require_equal("num_scale_frames", scale, int(expected_num_scale_frames))
        anchor_indices = post_scale_keyframe_indices(num_frames, scale, 1)
        cam_indices = np.arange(num_frames, dtype=np.int64)
        if validate_payload:
            _validate_payload_lengths(
                aggregator, camera, num_frames, anchor_indices, cam_indices
            )
        return CacheLayout(
            schema_version=1,
            num_frames=num_frames,
            num_scale_frames=scale,
            keyframe_interval=1,
            anchor_frame_indices=anchor_indices,
            cam_frame_indices=cam_indices,
            precompute_signature="legacy_dense",
            legacy_dense=True,
        )

    missing_agg = sorted((_SCHEMA_FIELDS | {"anchor_frame_indices"}) - agg_keys)
    missing_cam = sorted((_SCHEMA_FIELDS | {"cam_frame_indices"}) - cam_keys)
    if missing_agg or missing_cam:
        raise ValueError(
            f"incomplete versioned cache metadata: aggregator={missing_agg} camera={missing_cam}"
        )

    values = {}
    for name, cast in (
        ("cache_schema_version", int),
        ("keyframe_policy", str),
        ("num_frames", int),
        ("num_scale_frames", int),
        ("keyframe_interval", int),
        ("precompute_signature", str),
    ):
        agg_value = _scalar(aggregator, name, cast)
        cam_value = _scalar(camera, name, cast)
        _require_equal(name, cam_value, agg_value)
        values[name] = agg_value

    _require_equal(
        "cache_schema_version", values["cache_schema_version"], CACHE_SCHEMA_VERSION
    )
    policy = values["keyframe_policy"]
    if policy not in (KEYFRAME_POLICY, KEYFRAME_POLICY_FLOW):
        raise ValueError(
            f"unknown cache keyframe_policy {policy!r}; expected "
            f"{KEYFRAME_POLICY!r} or {KEYFRAME_POLICY_FLOW!r}"
        )
    num_frames = values["num_frames"]
    scale = values["num_scale_frames"]
    interval = values["keyframe_interval"]
    if num_frames < 1:
        raise ValueError(f"cache num_frames must be positive, got {num_frames}")
    if not 1 <= scale <= num_frames:
        raise ValueError(
            f"cache num_scale_frames must be in [1, {num_frames}], got {scale}"
        )
    flow_threshold = 0.0
    max_non_keyframe_gap = 0
    if policy == KEYFRAME_POLICY:
        if interval < 1:
            raise ValueError(
                f"cache keyframe_interval must be positive, got {interval}"
            )
    else:
        if interval != FLOW_KEYFRAME_INTERVAL_SENTINEL:
            raise ValueError(
                f"{KEYFRAME_POLICY_FLOW} cache must store keyframe_interval="
                f"{FLOW_KEYFRAME_INTERVAL_SENTINEL}, got {interval}"
            )
        for source_name, source in (("aggregator", aggregator), ("camera", camera)):
            missing = sorted(
                name
                for name in ("flow_threshold", "max_non_keyframe_gap")
                if name not in source
            )
            if missing:
                raise ValueError(
                    f"{KEYFRAME_POLICY_FLOW} {source_name} cache lacks {missing}"
                )
        for name, cast in (("flow_threshold", float), ("max_non_keyframe_gap", int)):
            agg_value = _scalar(aggregator, name, cast)
            cam_value = _scalar(camera, name, cast)
            _require_equal(name, cam_value, agg_value)
            values[name] = agg_value
        flow_threshold = values["flow_threshold"]
        max_non_keyframe_gap = values["max_non_keyframe_gap"]
        if not flow_threshold > 0.0:
            raise ValueError(
                f"cache flow_threshold must be positive, got {flow_threshold}"
            )
        if max_non_keyframe_gap < 1:
            raise ValueError(
                f"cache max_non_keyframe_gap must be positive, got {max_non_keyframe_gap}"
            )
    if not values["precompute_signature"]:
        raise ValueError("cache precompute_signature must be non-empty")
    if expected_num_frames is not None:
        _require_equal("num_frames", num_frames, int(expected_num_frames))
    if expected_num_scale_frames is not None:
        _require_equal("num_scale_frames", scale, int(expected_num_scale_frames))
    if expected_sliding_window is not None:
        for source_name, source in (("aggregator", aggregator), ("camera", camera)):
            if "kv_cache_sliding_window" not in source:
                raise ValueError(
                    f"{source_name} cache lacks kv_cache_sliding_window metadata"
                )
            _require_equal(
                f"{source_name} kv_cache_sliding_window",
                _scalar(source, "kv_cache_sliding_window", int),
                int(expected_sliding_window),
            )

    if policy == KEYFRAME_POLICY:
        expected_anchor = post_scale_keyframe_indices(num_frames, scale, interval)
        anchor_indices = _strict_indices(
            "anchor_frame_indices", aggregator["anchor_frame_indices"], expected_anchor
        )
    else:
        anchor_indices = _flow_gate_indices(
            "anchor_frame_indices",
            aggregator["anchor_frame_indices"],
            num_scale_frames=scale,
            num_frames=num_frames,
            max_non_keyframe_gap=max_non_keyframe_gap,
        )
    expected_cam = np.concatenate(
        (np.arange(min(scale, num_frames), dtype=np.int64), anchor_indices)
    )
    cam_indices = _strict_indices(
        "cam_frame_indices", camera["cam_frame_indices"], expected_cam, policy=policy
    )
    if validate_payload:
        _validate_payload_lengths(
            aggregator, camera, num_frames, anchor_indices, cam_indices
        )
    return CacheLayout(
        schema_version=values["cache_schema_version"],
        num_frames=num_frames,
        num_scale_frames=scale,
        keyframe_interval=interval,
        anchor_frame_indices=anchor_indices,
        cam_frame_indices=cam_indices,
        precompute_signature=values["precompute_signature"],
        keyframe_policy=policy,
        flow_threshold=flow_threshold,
        max_non_keyframe_gap=max_non_keyframe_gap,
    )


def _validate_payload_lengths(
    aggregator: Mapping[str, np.ndarray],
    camera: Mapping[str, np.ndarray],
    num_frames: int,
    anchor_indices: np.ndarray,
    cam_indices: np.ndarray,
) -> None:
    lengths = {
        "dino_cls": int(np.asarray(aggregator["dino_cls"]).shape[0]),
        "cam_pose_enc": int(np.asarray(camera["cam_pose_enc"]).shape[0]),
        "anchor_k": int(np.asarray(aggregator["anchor_k"]).shape[0]),
        "anchor_v": int(np.asarray(aggregator["anchor_v"]).shape[0]),
        "cam_k": int(np.asarray(camera["cam_k"]).shape[0]),
        "cam_v": int(np.asarray(camera["cam_v"]).shape[0]),
    }
    expected = {
        "dino_cls": int(num_frames),
        "cam_pose_enc": int(num_frames),
        "anchor_k": len(anchor_indices),
        "anchor_v": len(anchor_indices),
        "cam_k": len(cam_indices),
        "cam_v": len(cam_indices),
    }
    bad = {name: (lengths[name], expected[name]) for name in lengths if lengths[name] != expected[name]}
    if bad:
        raise ValueError(f"LingBot cache payload/index length mismatch: {bad}")


def _npz_shape(path, name: str) -> tuple[int, ...]:
    """Read a .npy member header without materializing a potentially multi-GB KV."""
    member = f"{name}.npy"
    with zipfile.ZipFile(os.fspath(path)) as archive:
        try:
            handle = archive.open(member)
        except KeyError as error:
            raise ValueError(f"cache {path} lacks array {name!r}") from error
        with handle:
            version = np.lib.format.read_magic(handle)
            shape, _fortran_order, _dtype = np.lib.format._read_array_header(
                handle, version
            )
    return tuple(int(value) for value in shape)


def validate_cache_files(
    aggregator_path,
    camera_path,
    *,
    expected_num_frames: int | None = None,
    expected_num_scale_frames: int | None = None,
    expected_sliding_window: int | None = None,
    require_versioned: bool = True,
) -> CacheLayout:
    """Header-only dependency check for a cache pair.

    This is suitable for zero-step preflight over an entire dataset: only small
    metadata/index arrays and ZIP member headers are read, never the KV payloads.
    """
    metadata = _SCHEMA_FIELDS | {
        "kv_cache_sliding_window",
        "flow_threshold",
        "max_non_keyframe_gap",
    }
    with np.load(aggregator_path, allow_pickle=False) as aggregator_file:
        aggregator = {
            name: aggregator_file[name]
            for name in metadata | {"anchor_frame_indices"}
            if name in aggregator_file
        }
        if not require_versioned and "cache_schema_version" not in aggregator_file:
            aggregator["meta"] = aggregator_file["meta"]
    with np.load(camera_path, allow_pickle=False) as camera_file:
        camera = {
            name: camera_file[name]
            for name in metadata | {"cam_frame_indices"}
            if name in camera_file
        }
        if not require_versioned and "cache_schema_version" not in camera_file:
            # Legacy inference requires a row count. This is a small pose array,
            # but use a dummy with the header-derived leading dimension anyway.
            camera["cam_pose_enc"] = np.empty(
                (_npz_shape(camera_path, "cam_pose_enc")[0], 0)
            )

    layout = validate_cache_pair(
        aggregator,
        camera,
        expected_num_frames=expected_num_frames,
        expected_num_scale_frames=expected_num_scale_frames,
        expected_sliding_window=expected_sliding_window,
        require_versioned=require_versioned,
        validate_payload=False,
    )
    actual_lengths = {
        "dino_cls": _npz_shape(aggregator_path, "dino_cls")[0],
        "cam_pose_enc": _npz_shape(camera_path, "cam_pose_enc")[0],
        "anchor_k": _npz_shape(aggregator_path, "anchor_k")[0],
        "anchor_v": _npz_shape(aggregator_path, "anchor_v")[0],
        "cam_k": _npz_shape(camera_path, "cam_k")[0],
        "cam_v": _npz_shape(camera_path, "cam_v")[0],
    }
    expected_lengths = {
        "dino_cls": layout.num_frames,
        "cam_pose_enc": layout.num_frames,
        "anchor_k": len(layout.anchor_frame_indices),
        "anchor_v": len(layout.anchor_frame_indices),
        "cam_k": len(layout.cam_frame_indices),
        "cam_v": len(layout.cam_frame_indices),
    }
    bad = {
        name: (actual_lengths[name], expected_lengths[name])
        for name in actual_lengths
        if actual_lengths[name] != expected_lengths[name]
    }
    if bad:
        raise ValueError(f"LingBot cache payload/index length mismatch: {bad}")
    return layout
