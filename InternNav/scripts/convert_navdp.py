"""Convert downloaded InternData-N1 vln_n1 tarballs to traj_data_navdp layout.

Each per-scene tarball already follows the LeRobot v2.1 layout that
NavDP_Base_Datset expects (scene/trajectory_NN/{data,videos,meta}/...), so
this is just a managed extract+verify with optional symlinking.

Usage:
    python convert_navdp.py \
        --src /home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data \
        --dst /home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp \
        [--limit 1] [--dry-run] [--verify]

NavDP_Base_Datset.__init__ expects:
    <root_dir>/<group>/<scene>/<traj>/data/chunk-000/episode_000000.parquet
    <root_dir>/<group>/<scene>/<traj>/data/chunk-000/path.ply
    <root_dir>/<group>/<scene>/<traj>/videos/chunk-000/observation.images.rgb/{0,1,...}.jpg
    <root_dir>/<group>/<scene>/<traj>/videos/chunk-000/observation.images.depth/{0,1,...}.png
"""

import argparse
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PARQUET_COLS = ('action', 'observation.camera_intrinsic', 'observation.camera_extrinsic')


def list_tarballs(src_dir: Path) -> list[Path]:
    return sorted(src_dir.rglob('*.tar.gz'))


def extract(tar_path: Path, dst_group_dir: Path, dry_run: bool = False) -> Path:
    """Extract one scene tarball under <dst_group_dir>/. Returns scene dir."""
    dst_group_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, 'r:gz') as tf:
        names = tf.getnames()
    top = {n.split('/', 1)[0] for n in names if n}
    if len(top) != 1:
        raise RuntimeError(f'tarball {tar_path.name} has multiple roots: {top}')
    scene_name = top.pop()
    scene_dir = dst_group_dir / scene_name
    if scene_dir.exists():
        print(f'  [skip] {scene_name} already extracted at {scene_dir}')
        return scene_dir
    if dry_run:
        print(f'  [dry-run] would extract {tar_path.name} -> {dst_group_dir}/{scene_name}')
        return scene_dir
    print(f'  [extract] {tar_path.name} -> {dst_group_dir}/')
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(dst_group_dir)
    return scene_dir


def verify_traj(traj_dir: Path) -> tuple[bool, str]:
    """Sanity-check one trajectory dir against the loader's contract."""
    parquet = traj_dir / 'data' / 'chunk-000' / 'episode_000000.parquet'
    path_ply = traj_dir / 'data' / 'chunk-000' / 'path.ply'
    rgb_dir = traj_dir / 'videos' / 'chunk-000' / 'observation.images.rgb'
    depth_dir = traj_dir / 'videos' / 'chunk-000' / 'observation.images.depth'

    if not parquet.is_file():
        return False, f'missing parquet: {parquet}'
    if not path_ply.is_file():
        return False, f'missing path.ply: {path_ply}'
    if not rgb_dir.is_dir():
        return False, f'missing rgb dir: {rgb_dir}'
    if not depth_dir.is_dir():
        return False, f'missing depth dir: {depth_dir}'

    try:
        df = pd.read_parquet(parquet)
    except Exception as e:
        return False, f'parquet unreadable: {e}'

    for col in REQUIRED_PARQUET_COLS:
        if col not in df.columns:
            return False, f'missing parquet col: {col}'

    n_frames = len(df)
    rgbs = sorted(p.name for p in rgb_dir.iterdir() if p.suffix == '.jpg')
    depths = sorted(p.name for p in depth_dir.iterdir() if p.suffix == '.png')
    if len(rgbs) != len(depths):
        return False, f'rgb/depth count mismatch: {len(rgbs)} vs {len(depths)}'
    if len(rgbs) != n_frames:
        return False, f'image count {len(rgbs)} != parquet rows {n_frames}'

    # Ensure 0..n-1 numbering used by the loader.
    expected_rgb = {f'{i}.jpg' for i in range(n_frames)}
    if set(rgbs) != expected_rgb:
        missing = expected_rgb - set(rgbs)
        return False, f'rgb numbering broken (e.g. missing {sorted(missing)[:3]})'
    expected_depth = {f'{i}.png' for i in range(n_frames)}
    if set(depths) != expected_depth:
        missing = expected_depth - set(depths)
        return False, f'depth numbering broken (e.g. missing {sorted(missing)[:3]})'

    # Spot-check shapes — these reshapes are exactly what the loader does.
    try:
        np.vstack(np.array(df['observation.camera_intrinsic'].iloc[0])).reshape(3, 3)
        np.vstack(np.array(df['observation.camera_extrinsic'].iloc[0])).reshape(4, 4)
        np.array([np.stack(f) for f in df['action']], dtype=np.float64).reshape(-1, 4, 4)
    except Exception as e:
        return False, f'parquet reshape failed: {e}'

    return True, f'ok ({n_frames} frames)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='dir containing <group>/*.tar.gz')
    ap.add_argument('--dst', required=True, help='target traj_data_navdp dir')
    ap.add_argument('--limit', type=int, default=0, help='only process first N tarballs')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--verify', action='store_true', help='verify each traj after extract')
    args = ap.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()
    if not src.is_dir():
        sys.exit(f'source dir not found: {src}')

    tarballs = list_tarballs(src)
    if args.limit:
        tarballs = tarballs[: args.limit]
    if not tarballs:
        sys.exit(f'no .tar.gz under {src}')
    print(f'found {len(tarballs)} tarballs under {src}')

    extracted_scenes: list[Path] = []
    for tp in tarballs:
        # Group name = parent dir (e.g. matterport3d_d435i)
        group = tp.parent.name
        scene_dir = extract(tp, dst / group, dry_run=args.dry_run)
        extracted_scenes.append(scene_dir)

    if args.verify and not args.dry_run:
        print('\n--- verifying ---')
        n_ok = 0
        n_bad = 0
        for scene_dir in extracted_scenes:
            traj_dirs = sorted(p for p in scene_dir.iterdir() if p.is_dir() and p.name.startswith('trajectory_'))
            print(f'{scene_dir.relative_to(dst)}: {len(traj_dirs)} trajectories')
            for traj in traj_dirs:
                ok, msg = verify_traj(traj)
                if ok:
                    n_ok += 1
                else:
                    n_bad += 1
                    print(f'  BAD  {traj.name}: {msg}')
        print(f'\nverified: {n_ok} ok, {n_bad} bad')

    print('done.')


if __name__ == '__main__':
    main()
