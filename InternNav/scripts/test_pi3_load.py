"""Load Pi3 pretrained weights into GeometryModel and report what loaded.

Pi3 (yyfz233/Pi3) shares: encoder, decoder, point_decoder, point_head,
conf_decoder, conf_head, camera_decoder, register_token.
GeometryModel overrides camera_head (Pi3 CameraHead -> ExtrinctHead with extra
fc_pose) and adds: fusion_head, wp_head, world_point_decoder, world_point_head,
depth_model, former_*, state_*, scene_*.

Usage:
    python scripts/test_pi3_load.py [--ckpt <path-to-model.safetensors>]
"""

import argparse
import os
import sys

import torch
from safetensors.torch import load_file

# Make NavDP/baselines/logoplanner importable so GeometryModel resolves.
HERE = os.path.dirname(os.path.abspath(__file__))
INTERNNAV_ROOT = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.dirname(INTERNNAV_ROOT)
LOGO_DIR = os.path.join(ROOT, 'NavDP', 'baselines', 'logoplanner')
sys.path.insert(0, LOGO_DIR)

from geometry_model import GeometryModel  # noqa: E402


DEFAULT_CKPT = '/home/nyuair/data-001/checkpoints/Pi3/model.safetensors'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=DEFAULT_CKPT)
    ap.add_argument('--device', default='cpu', help='cpu or cuda:N (cpu fine for sanity)')
    args = ap.parse_args()

    print(f'instantiating GeometryModel...')
    model = GeometryModel(device=args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  total params: {n_params/1e6:.1f}M')

    print(f'\nloading Pi3 weights from {args.ckpt}')
    if args.ckpt.endswith('.safetensors'):
        pi3_state = load_file(args.ckpt)
    else:
        pi3_state = torch.load(args.ckpt, map_location='cpu')
        if isinstance(pi3_state, dict) and 'state_dict' in pi3_state:
            pi3_state = pi3_state['state_dict']

    print(f'  pi3 ckpt has {len(pi3_state)} tensors, {sum(t.numel() for t in pi3_state.values())/1e6:.1f}M params')

    geom_keys = set(model.state_dict().keys())
    pi3_keys = set(pi3_state.keys())

    shared = geom_keys & pi3_keys
    geom_only = geom_keys - pi3_keys
    pi3_only = pi3_keys - geom_keys

    print(f'\nkey overlap:')
    print(f'  shared (will load):  {len(shared)}')
    print(f'  geom-only (random init): {len(geom_only)}')
    print(f'  pi3-only (unexpected, ignored): {len(pi3_only)}')

    incompat = model.load_state_dict(pi3_state, strict=False)
    print(f'\n--- load_state_dict(strict=False) result ---')
    print(f'  missing keys count: {len(incompat.missing_keys)}')
    print(f'  unexpected keys count: {len(incompat.unexpected_keys)}')

    # Show prefix breakdown for clarity.
    def prefix_count(keys):
        d = {}
        for k in keys:
            p = k.split('.')[0]
            d[p] = d.get(p, 0) + 1
        return sorted(d.items(), key=lambda x: -x[1])

    print(f'\nmissing keys by top-level prefix:')
    for p, c in prefix_count(incompat.missing_keys):
        print(f'  {p}: {c}')

    print(f'\nunexpected keys by top-level prefix:')
    for p, c in prefix_count(incompat.unexpected_keys):
        print(f'  {p}: {c}')

    # Check shapes for ExtrinctHead override.
    print('\ncamera_head detail (Pi3 CameraHead -> ExtrinctHead):')
    for n, p in model.camera_head.named_parameters():
        loaded = ('camera_head.' + n) in shared
        print(f'  camera_head.{n}: shape={tuple(p.shape)}, loaded_from_pi3={loaded}')

    # Quick smoke forward (small dummy input) on cpu — just to confirm no shape errors.
    if args.device == 'cpu':
        print('\nskipping forward smoke (cpu only — Pi3 ops require cuda)')
    else:
        print('\nrunning forward smoke...')
        model = model.to(args.device).eval()
        imgs = torch.randn(1, 12, 168, 308, 3, device=args.device)
        depths = torch.randn(1, 12, 168, 308, 1, device=args.device)
        with torch.no_grad():
            out = model(imgs, depths)
        tokens, geom = out
        print(f'  tokens hidden / state / scene: '
              f'{tokens[0].shape}, {tokens[1].shape}, {tokens[2].shape}')
        print(f'  camera_poses / local_pts / world_pts: '
              f'{geom[0].shape}, {geom[1].shape}, {geom[2].shape}')

    print('\nDONE')


if __name__ == '__main__':
    main()
