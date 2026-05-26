"""Forward-pass smoke test for GeometryModel_LingBot.

Builds the new geometry backbone with default config and runs one forward
on random (B=1, T=12, 168x308) RGB-D tensors. Checks that:
  - imports resolve (lingbot_map, depth_anything, Pi3, policy_backbone)
  - construction succeeds (DINOv2 patch_embed + 24 GCA blocks + DA-S + heads)
  - forward runs without shape error / NaN
  - output tuple shape matches the LoGoPlanner Pi3 GeometryModel contract:
        state_token: (B, T, 384)
        scene_token: (B, T, 384)
        camera_poses: (B, T, 5)
        local_points: (B, T, H, W, 3)
        world_points: (B, T, H, W, 3)

Run from this directory:
    cd Nav/NavDP/baselines/logoplanner
    python smoke_lingbot.py
"""

import os
import sys
import time
import torch

# Make sibling modules importable (geometry_model_lingbot, policy_backbone, ...)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geometry_model_lingbot import GeometryModel_LingBot


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    B, T, H, W = 1, 12, 168, 308

    print(f"[smoke] device={device}  shape=(B={B}, T={T}, H={H}, W={W})")
    t0 = time.time()
    model = GeometryModel_LingBot(context_size=T, device=device)
    model = model.to(device)
    model.eval()
    print(f"[smoke] model built in {time.time() - t0:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] params: total={n_params/1e6:.1f}M  trainable={n_train/1e6:.1f}M")

    imgs   = torch.rand(B, T, H, W, 3, dtype=dtype, device=device)
    depths = torch.rand(B, T, H, W, 1, dtype=dtype, device=device) * 5.0   # 0–5 m

    print("[smoke] running forward...")
    t0 = time.time()
    with torch.no_grad():
        (h, state_token, scene_token), (camera_poses, local_points, world_points) = \
            model(imgs, depths)
    print(f"[smoke] forward done in {time.time() - t0:.2f}s")

    # Shape checks (the LoGoPlanner policy contract)
    expected = {
        'state_token':  (B, T, 384),
        'scene_token':  (B, T, 384),
        'camera_poses': (B, T, 5),
        'local_points': (B, T, H, W, 3),
        'world_points': (B, T, H, W, 3),
    }
    got = {
        'state_token':  tuple(state_token.shape),
        'scene_token':  tuple(scene_token.shape),
        'camera_poses': tuple(camera_poses.shape),
        'local_points': tuple(local_points.shape),
        'world_points': tuple(world_points.shape),
    }
    ok = True
    for name, want in expected.items():
        mark = "OK " if got[name] == want else "FAIL"
        if got[name] != want:
            ok = False
        print(f"  [{mark}] {name:13s} want={want}  got={got[name]}")

    # NaN check
    for name, t in [('state_token', state_token), ('scene_token', scene_token),
                    ('camera_poses', camera_poses), ('local_points', local_points),
                    ('world_points', world_points)]:
        if torch.isnan(t).any():
            ok = False
            print(f"  [NaN ] {name} contains NaN")

    print()
    print(f"[smoke] result: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
