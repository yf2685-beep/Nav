#!/usr/bin/env python
"""Score LingBot pose quality of precomputed caches against generated GT.

Compares keyframe selectors (e.g. flow-gated vs auto fixed-interval) on the
SAME episodes by reading ``cam_pose_enc`` straight out of each arm's
``lingbot_cam_cache.npz`` — i.e. it scores the actual training artifact, not a
separate diagnostic stream.  The metrics (Sim(2)-aligned ATE, RPE at fixed
frame gaps, per-leg drift) replicate scripts from the 2026-07-17 sparse-
keyframe validation so numbers are directly comparable with that report.

Usage (any env with numpy+pandas; no GPU, no torch needed):

    python scripts/eval/diag_flow_gate_ate.py \
        --episodes-root /home/asus/Research/Nav/memnav_viz/validate_gated \
        --arm auto=/path/to/out_root_auto \
        --arm flow20=/path/to/out_root_flow20 \
        --out /tmp/flow_gate_ate.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

CAM_CACHE_NAME = "lingbot_cam_cache.npz"


# --------------------------------------------------------------------------- #
# Scoring primitives (2D Sim(2) alignment on the ground plane)
# --------------------------------------------------------------------------- #
def fit_sim2(source: np.ndarray, target: np.ndarray):
    """Umeyama similarity fit (scale, rotation, translation) in 2D."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError(f"Sim(2) expects matching [N,2] arrays, got {source.shape}, {target.shape}")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(source_centered * source_centered, axis=1))
    if variance < 1e-12:
        raise RuntimeError("degenerate predicted trajectory for Sim(2) alignment")
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def apply_sim2(points: np.ndarray, fit) -> np.ndarray:
    scale, rotation, translation = fit
    return scale * (np.asarray(points) @ rotation.T) + translation


def angular_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first / np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-12)
    dot = np.sum(first * second, axis=-1)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    return np.degrees(np.abs(np.arctan2(cross, dot)))


def load_ground_truth(episode_dir: Path):
    """GT camera-to-world [T,4,4] + gen_meta from a generated episode."""
    import pandas as pd

    dataframe = pd.read_parquet(episode_dir / "data/chunk-000/episode_000000.parquet")
    action = np.asarray(
        [np.stack(item) for item in dataframe["action"]], dtype=np.float64
    ).reshape(-1, 4, 4)
    with (episode_dir / "meta/gen_meta.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    return action, metadata


def summarize_pose(pose: np.ndarray, action: np.ndarray, metadata: dict,
                   rpe_gaps=(16, 64, 128, 256)) -> dict:
    """Sim(2)-aligned planar ATE/RPE + per-leg drift for one episode.

    pose   : [S, 9] LingBot pose encodings (absT xyz in [:3]).
    action : [S, 4, 4] GT camera-to-world.
    Ground planes: LingBot/OpenCV world is x-z (y vertical); the generated
    dataset world is x-y (z vertical).
    """
    length = min(len(pose), len(action))
    if length < 2:
        raise RuntimeError("trajectory is too short to score")
    pose = np.asarray(pose[:length], dtype=np.float64)
    action = np.asarray(action[:length], dtype=np.float64)

    predicted_xy = pose[:, [0, 2]]
    target_xy = action[:, :2, 3]
    fit = fit_sim2(predicted_xy, target_xy)
    aligned = apply_sim2(predicted_xy, fit)
    residual = np.linalg.norm(aligned - target_xy, axis=1)

    result = {
        "n": length,
        "legs": int(metadata.get("n_legs", 0)),
        "sim2_scale": fit[0],
        "ate_rmse_m": float(np.sqrt(np.mean(residual**2))),
        "ate_median_m": float(np.median(residual)),
        "ate_p90_m": float(np.percentile(residual, 90)),
    }

    # RPE uses the one trajectory-level Sim(2), so local nonlinear drift cannot
    # be hidden by independently aligning each interval.
    for gap in rpe_gaps:
        if length <= gap:
            continue
        predicted_delta = aligned[gap:] - aligned[:-gap]
        target_delta = target_xy[gap:] - target_xy[:-gap]
        vector_error = np.linalg.norm(predicted_delta - target_delta, axis=1)
        target_distance = np.linalg.norm(target_delta, axis=1)
        valid = target_distance >= 0.25
        prefix = f"rpe_{gap:03d}"
        result[f"{prefix}_rmse_m"] = float(np.sqrt(np.mean(vector_error**2)))
        result[f"{prefix}_median_m"] = float(np.median(vector_error))
        if np.any(valid):
            result[f"{prefix}_direction_median_deg"] = float(
                np.median(angular_error_deg(predicted_delta[valid], target_delta[valid]))
            )

    switches = [int(value) for value in metadata.get("switches", [])]
    bounds = [0] + [value for value in switches if 0 < value < length] + [length]
    leg_rows = []
    for leg, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        if end - start < 2:
            continue
        predicted_delta = aligned[end - 1] - aligned[start]
        target_delta = target_xy[end - 1] - target_xy[start]
        leg_rows.append({
            "leg": leg,
            "gt_displacement_m": float(np.linalg.norm(target_delta)),
            "direction_error_deg": float(
                angular_error_deg(predicted_delta[None], target_delta[None])[0]
            ),
            "vector_error_m": float(np.linalg.norm(predicted_delta - target_delta)),
        })
    result["legs_detail"] = leg_rows
    return result


def aggregate_rows(rows: list, variant: str) -> dict:
    subset = [row for row in rows if row["variant"] == variant]
    output = {"variant": variant, "episodes": len(subset)}
    for legs in sorted({int(row["legs"]) for row in subset}):
        group = [row for row in subset if int(row["legs"]) == legs]
        ate = np.asarray([row["ate_rmse_m"] for row in group], dtype=np.float64)
        output[f"{legs}leg"] = {
            "episodes": len(group),
            "ate_mean_m": float(np.mean(ate)),
            "ate_median_m": float(np.median(ate)),
            "ate_range_m": [float(np.min(ate)), float(np.max(ate))],
        }
    return output


# --------------------------------------------------------------------------- #
# Cache discovery + comparison driver
# --------------------------------------------------------------------------- #
def discover_cached_episodes(arm_root: Path) -> dict:
    """Map <group>/<scene>/<episode> -> cam-cache path under one arm's out_root."""
    episodes = {}
    for cache_path in sorted(arm_root.rglob(CAM_CACHE_NAME)):
        chunk_dir = cache_path.parent                      # .../videos/chunk-000
        episode_dir = chunk_dir.parent.parent              # .../<episode>
        episodes[os.fspath(episode_dir.relative_to(arm_root))] = cache_path
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-root", required=True,
                        help="Root holding the generated episodes (GT parquet + gen_meta).")
    parser.add_argument("--arm", action="append", required=True, metavar="LABEL=OUT_ROOT",
                        help="Named precompute out_root; repeat per selector arm.")
    parser.add_argument("--rpe-gaps", default="16,64,128,256")
    parser.add_argument("--out", default="", help="Optional JSON dump of all rows.")
    args = parser.parse_args()

    episodes_root = Path(args.episodes_root)
    rpe_gaps = tuple(int(v) for v in args.rpe_gaps.split(",") if v)
    arms = []
    for spec in args.arm:
        label, _, root = spec.partition("=")
        if not root:
            parser.error(f"--arm expects LABEL=OUT_ROOT, got {spec!r}")
        arms.append((label, Path(root)))

    per_arm = {label: discover_cached_episodes(root) for label, root in arms}
    shared = sorted(set.intersection(*(set(v) for v in per_arm.values())))
    if not shared:
        raise SystemExit("no episode is cached under every arm — nothing to compare")
    for label, found in per_arm.items():
        extra = sorted(set(found) - set(shared))
        if extra:
            print(f"[warn] arm {label}: {len(extra)} episode(s) not shared, skipped: {extra[:3]}...")

    rows = []
    key_gap_col = f"rpe_{max(rpe_gaps):03d}_direction_median_deg" if rpe_gaps else None
    header = f"{'episode':44s} {'arm':8s} {'legs':>4s} {'kf':>5s} {'kf%':>6s} " \
             f"{'ate_rmse':>9s} {'ate_p90':>8s} {'dir_med':>8s}"
    print(header)
    print("-" * len(header))
    for episode in shared:
        action, metadata = load_ground_truth(episodes_root / episode)
        for label, _root in arms:
            with np.load(per_arm[label][episode], allow_pickle=False) as cache:
                pose = np.asarray(cache["cam_pose_enc"], dtype=np.float64)
                cam_indices = np.asarray(cache["cam_frame_indices"])
                scale = int(np.asarray(cache["num_scale_frames"]).reshape(-1)[0])
            row = summarize_pose(pose, action, metadata, rpe_gaps=rpe_gaps)
            n_keyframes = int(len(cam_indices) - scale)
            n_streamed = max(1, row["n"] - scale)
            row.update(
                variant=label,
                episode=episode,
                keyframes=n_keyframes,
                keyframe_fraction=n_keyframes / n_streamed,
            )
            rows.append(row)
            direction = row.get(key_gap_col, float("nan")) if key_gap_col else float("nan")
            print(f"{episode:44s} {label:8s} {row['legs']:4d} {n_keyframes:5d} "
                  f"{100.0 * row['keyframe_fraction']:5.1f}% "
                  f"{row['ate_rmse_m']:8.3f}m {row['ate_p90_m']:7.3f}m {direction:7.2f}d")

    print()
    for label, _root in arms:
        print(json.dumps(aggregate_rows(rows, label), indent=2, sort_keys=True))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True, default=float)
        print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
