"""One-shot diagnostic: how often does process_obstacle_points return zero?

For each trajectory in the dataset, replicate the obstacle-extraction logic from
internnav/dataset/navdp_dataset_lerobot.py::process_obstacle_points:
  - load <traj>/data/chunk-000/path.ply
  - extract "path" points by color match to RED  (0.5, 0, 0)
  - extract "obstacle" points by color match to BLUE (0, 0, 0.5)
  - filter obstacle points to ±2 m of the path's XY bounding box
  - count

Report:
  - total trajectories scanned
  - fraction with 0 obstacles (the (2.0, 2.0) fallback case)
  - distribution of obstacle counts

Usage (from InternNav/):
  python scripts/diag_obstacle_coverage.py \
      --root-dir /scratch/lg154/Research/datasets/InternData-N1/vln_n1/_raw \
      --cache /tmp/logoplanner_dataset_lerobot.json \
      --max-trajectories 0
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import open3d as o3d


def find_trajectories(root_dir):
    """Mirror navdp_dataset_lerobot.__init__ walk to enumerate trajectories."""
    out = []
    for group in sorted(os.listdir(root_dir)):
        group_path = os.path.join(root_dir, group)
        if not os.path.isdir(group_path):
            continue
        for scene in sorted(os.listdir(group_path)):
            scene_path = os.path.join(group_path, scene)
            if not os.path.isdir(scene_path):
                continue
            for traj in sorted(os.listdir(scene_path)):
                afford = os.path.join(scene_path, traj, 'data/chunk-000/path.ply')
                if os.path.isfile(afford):
                    out.append((group, scene, traj, afford))
    return out


def load_cache(cache_path):
    """Reuse the existing preload JSON if present."""
    with open(cache_path) as f:
        d = json.load(f)
    out = []
    for afford in d['trajectory_afford_path']:
        # Recover group/scene/traj from path: .../root/group/scene/traj/data/chunk-000/path.ply
        parts = afford.split(os.sep)
        try:
            i = parts.index('data')
            traj = parts[i - 1]
            scene = parts[i - 2]
            group = parts[i - 3]
        except (ValueError, IndexError):
            group = scene = traj = '?'
        out.append((group, scene, traj, afford))
    return out


def obstacle_count_for_traj(afford_path):
    """Replicate process_obstacle_points + process_path_points filtering."""
    pcd = o3d.io.read_point_cloud(afford_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    if colors.size == 0 or points.size == 0:
        return None  # malformed

    # Path: BLACK (0, 0, 0). The dataset code comments say "sometimes the path
    # are saved as black points" — that's what matterport3d_d435i uses.
    cdist_path = np.abs(colors - np.array([0, 0, 0])).sum(axis=-1)
    path_idx = np.where(cdist_path < 0.05)[0]
    if path_idx.size == 0:
        return None
    path_xy = points[path_idx, :2]
    lo = path_xy.min(axis=0)
    hi = path_xy.max(axis=0)

    # Obstacles: blue = (0, 0, 0.5), within ±2 m of path xy-bbox
    cdist_obs = np.abs(colors - np.array([0, 0, 0.5])).sum(axis=-1)
    cond_x = (points[:, 0] >= lo[0] - 2.0) & (points[:, 0] <= hi[0] + 2.0)
    cond_y = (points[:, 1] >= lo[1] - 2.0) & (points[:, 1] <= hi[1] + 2.0)
    sel = (cdist_obs < 0.05) & cond_x & cond_y
    return int(sel.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root-dir', type=str, default=None,
                   help='Dataset root (walked if --cache absent or missing).')
    p.add_argument('--cache', type=str, default='/tmp/logoplanner_dataset_lerobot.json',
                   help='Preload JSON from prior training; reused when present.')
    p.add_argument('--max-trajectories', type=int, default=0,
                   help='If >0, stop after this many trajectories (sampled in order).')
    args = p.parse_args()

    if os.path.isfile(args.cache):
        print(f'[diag] using cache: {args.cache}')
        traj_list = load_cache(args.cache)
    else:
        if not args.root_dir:
            sys.exit(f'[diag] no cache at {args.cache} and --root-dir not given')
        print(f'[diag] walking root: {args.root_dir}')
        traj_list = find_trajectories(args.root_dir)
    print(f'[diag] total trajectories found: {len(traj_list)}')
    if args.max_trajectories and len(traj_list) > args.max_trajectories:
        # uniform stride sample so we cover scene diversity
        step = max(len(traj_list) // args.max_trajectories, 1)
        traj_list = traj_list[::step][:args.max_trajectories]
        print(f'[diag] sampled to {len(traj_list)} (stride {step})')

    counts = []
    malformed = 0
    by_scene = {}
    by_group = {}
    for i, (group, scene, traj, afford) in enumerate(traj_list):
        c = obstacle_count_for_traj(afford)
        if c is None:
            malformed += 1
            continue
        counts.append(c)
        by_scene.setdefault(scene, []).append(c)
        by_group.setdefault(group, []).append(c)
        if (i + 1) % 200 == 0:
            n = len(counts)
            zero = sum(1 for x in counts if x == 0)
            print(f'  scanned {i+1:>5}/{len(traj_list):<5}  '
                  f'fallback={zero}/{n} ({100*zero/max(n,1):.1f}%)', flush=True)

    counts = np.array(counts, np.int64)
    n = len(counts)
    if n == 0:
        sys.exit('[diag] no trajectories parsed; nothing to report')

    zero = int((counts == 0).sum())
    near = int((counts < 10).sum())
    sparse = int((counts < 100).sum())

    print('\n========================================================================')
    print(f'Trajectories scanned: {n}   malformed (skipped): {malformed}')
    print(f'  obstacle count == 0   (FALLBACK):  {zero:>5}  ({100*zero/n:.2f}%)')
    print(f'  obstacle count <  10:              {near:>5}  ({100*near/n:.2f}%)')
    print(f'  obstacle count <  100:             {sparse:>5}  ({100*sparse/n:.2f}%)')
    print(f'  median obstacle count: {int(np.median(counts))}')
    print(f'  mean   obstacle count: {counts.mean():.1f}')
    print(f'  pXX (10/25/50/75/90/99): '
          f'{np.percentile(counts, [10,25,50,75,90,99]).astype(int).tolist()}')

    print('\nBy group:')
    for g, cs in sorted(by_group.items()):
        cs = np.array(cs)
        z = int((cs == 0).sum())
        print(f'  {g:<32} n={len(cs):>5}  fallback={z:>5} ({100*z/len(cs):.1f}%)  '
              f'median={int(np.median(cs))}')

    print('\nWorst 10 scenes by fallback rate (min 5 trajs):')
    rows = []
    for s, cs in by_scene.items():
        if len(cs) >= 5:
            cs = np.array(cs)
            z = int((cs == 0).sum())
            rows.append((100 * z / len(cs), len(cs), z, s))
    rows.sort(reverse=True)
    for rate, n_scene, z, s in rows[:10]:
        print(f'  {rate:5.1f}%   {z:>3}/{n_scene:<3}   {s}')

    print('========================================================================')


if __name__ == '__main__':
    main()
