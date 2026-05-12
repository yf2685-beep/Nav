"""Build a warm-start LoGoPlannerNet checkpoint with Pi3 + DepthAnythingV2-S
weights pre-injected, so trainer's --ckpt-to-load can pick it up.

Pipeline:
  1. Build LoGoPlannerNet (random init).
  2. Load Pi3 weights into policy.state_encoder (the GeometryModel, which extends Pi3).
  3. Load DepthAnythingV2-S backbone (the .pretrained DINOv2-S) into:
        - policy.rgbd_encoder.rgb_model.*
        - policy.rgbd_encoder.depth_model.*
        - policy.state_encoder.depth_model.*
  4. Save as a HuggingFace-style ckpt dir with pytorch_model.bin + config.json.

Usage:
    python scripts/build_warmstart_ckpt.py \
        --pi3 /home/nyuair/data-001/checkpoints/Pi3/model.safetensors \
        --depth /home/nyuair/data-001/checkpoints/DepthAnythingV2-Small/depth_anything_v2_vits.pth \
        --out  /home/nyuair/data-001/checkpoints/logoplanner_warmstart
"""

import argparse
import os
import sys
from collections import Counter

import torch
from safetensors.torch import load_file


def load_pi3_state(path):
    if path.endswith('.safetensors'):
        return load_file(path)
    sd = torch.load(path, map_location='cpu')
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    return sd


def load_depth_anything_backbone(path):
    """Returns the DINOv2-S backbone state dict (pretrained.*  keys, prefix stripped)."""
    full = torch.load(path, map_location='cpu')
    return {k[len('pretrained.'):]: v for k, v in full.items() if k.startswith('pretrained.')}


def merge_into_lp(lp_state, sub_state, prefix):
    """Merge sub_state into lp_state by prefixing keys; track stats."""
    n_match, n_skip_shape, n_missing = 0, 0, 0
    for k, v in sub_state.items():
        full_k = prefix + k
        if full_k in lp_state:
            if lp_state[full_k].shape == v.shape:
                lp_state[full_k] = v.clone()
                n_match += 1
            else:
                print(f'  shape mismatch: {full_k}: {tuple(lp_state[full_k].shape)} vs ckpt {tuple(v.shape)}')
                n_skip_shape += 1
        else:
            n_missing += 1
    return n_match, n_skip_shape, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pi3', required=True)
    ap.add_argument('--depth', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    # Make repo paths importable.
    HERE = os.path.dirname(os.path.abspath(__file__))
    INTERNNAV_ROOT = os.path.abspath(os.path.join(HERE, '..'))
    ROOT = os.path.dirname(INTERNNAV_ROOT)
    sys.path.insert(0, INTERNNAV_ROOT)
    sys.path.insert(0, os.path.join(INTERNNAV_ROOT, 'src/diffusion-policy'))
    sys.path.insert(0, os.path.join(ROOT, 'NavDP/baselines/logoplanner'))

    from internnav.model.basemodel.logoplanner.logoplanner_policy import (  # noqa: E402
        LoGoPlannerNet, LoGoPlannerModelConfig,
    )
    from scripts.train.configs.logoplanner import logoplanner_exp_cfg  # noqa: E402

    # --- 1) Build random LoGoPlannerNet on CPU
    print('building LoGoPlannerNet (random init, cpu)...')
    cfg_dict = logoplanner_exp_cfg.model_dump()
    cfg_dict['local_rank'] = 0
    cfg = LoGoPlannerModelConfig(model_cfg=cfg_dict)
    model = LoGoPlannerNet(cfg)
    model = model.to('cpu')
    model.policy.device = 'cpu'

    lp_state = model.state_dict()
    n_total = len(lp_state)
    n_params = sum(t.numel() for t in lp_state.values())
    print(f'  state_dict has {n_total} tensors, {n_params/1e6:.1f}M params')

    # --- 2) Inject Pi3 -> policy.state_encoder.*
    print(f'\nloading Pi3: {args.pi3}')
    pi3_state = load_pi3_state(args.pi3)
    print(f'  pi3 ckpt: {len(pi3_state)} tensors')
    nm, ns, nx = merge_into_lp(lp_state, pi3_state, 'policy.state_encoder.')
    print(f'  Pi3 -> state_encoder: matched={nm}, shape-skip={ns}, missing-in-lp={nx}')

    # --- 3) Inject DepthAnythingV2-S backbone -> three places
    print(f'\nloading DepthAnythingV2-S: {args.depth}')
    da_state = load_depth_anything_backbone(args.depth)
    print(f'  da backbone: {len(da_state)} tensors')

    for prefix in [
        'policy.rgbd_encoder.rgb_model.',
        'policy.rgbd_encoder.depth_model.',
        'policy.state_encoder.depth_model.',
    ]:
        nm, ns, nx = merge_into_lp(lp_state, da_state, prefix)
        print(f'  DA -> {prefix:<48s}: matched={nm}, shape-skip={ns}, missing-in-lp={nx}')

    # --- 4) Verify and save
    print('\nverifying load_state_dict round-trip...')
    inc = model.load_state_dict(lp_state, strict=False)
    # Should have NO unexpected; missing should be only the still-uninitialized
    # task-specific heads (action_head, pg_pred_mlp, etc).
    print(f'  missing: {len(inc.missing_keys)}')
    print(f'  unexpected: {len(inc.unexpected_keys)}')
    if inc.missing_keys:
        prefixes = Counter(k.split('.')[0] + '.' + k.split('.')[1] for k in inc.missing_keys if '.' in k)
        print('  missing-key prefix breakdown (top 10):')
        for p, c in prefixes.most_common(10):
            print(f'    {p}: {c}')

    os.makedirs(args.out, exist_ok=True)
    out_bin = os.path.join(args.out, 'pytorch_model.bin')
    torch.save(lp_state, out_bin)
    cfg.save_pretrained(args.out)
    sz_mb = os.path.getsize(out_bin) / 1e6
    print(f'\nsaved: {out_bin} ({sz_mb:.1f} MB)')
    print(f'      + config.json')
    print('\nUse with:')
    print(f'  CKPT_TO_LOAD={args.out} bash scripts/train/runs.sh ...')


if __name__ == '__main__':
    main()
