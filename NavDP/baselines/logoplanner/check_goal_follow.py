"""Open-loop goal-following diagnostic for the streaming GCT policy.

Question: does the trained policy actually steer toward the goal, or does it
just walk? We feed REAL dataset samples (goal in the training convention =
batch_pg, current-frame chassis coords), run the diffusion prediction, and
measure the angle between the predicted trajectory's heading and the goal
direction. Compare goal-conditioned (mg) vs no-goal (ng), and sanity-check the
GT trajectory's own alignment to the goal.

Interpretation:
  * GT angle small (validates the dataset goal convention) AND
    - mg angle small (≪ ng)  -> model FOLLOWS the goal  -> the IsaacSim eval's
      goal-frame transform is the bug (fixable in eval).
    - mg angle ≈ ng (~random) -> model IGNORES the goal -> conditioning/training
      issue (needs more training or an architecture fix).
  * GT angle large -> the streaming dataset's pg convention itself is wrong.

Run on 131 (data + GPU there). Set LINGBOT/paths via the launcher.
"""
import os
import sys

os.environ.setdefault('LOGO_BACKBONE', 'lingbot_v2')
os.environ['LOGO_STREAMING'] = '1'
os.environ.setdefault('LOGO_N_ANCHOR', '8')
os.environ.setdefault('LOGO_N_TRAJ', '16')
os.environ.setdefault('LOGO_N_WINDOW', '64')

import numpy as np
import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS)  # policy_network, streaming_gct
_INTERNNAV = os.path.normpath(os.path.join(_THIS, '..', '..', '..', 'InternNav'))
sys.path.insert(0, _INTERNNAV)

from policy_network import LoGoPlanner_Policy           # noqa: E402
from streaming_gct import partition_window_tokens       # noqa: E402
from internnav.dataset.logoplanner_dataset_lerobot import (  # noqa: E402
    LoGoPlanner_Dataset, logoplanner_collate_fn,
)

DEV = 'cuda:0'
CKPT = os.environ['DIAG_CKPT']
ROOT = os.environ.get('DIAG_ROOT', '/media/cvpr/yuxuan/logoplanner/data/mini_clean')
CACHE = os.environ.get('DIAG_CACHE', '/tmp/logo_stream_ds_full.json')
N_SAMPLES = int(os.environ.get('DIAG_N', '24'))
N_ANCHOR, N_WINDOW = 8, 64


def angle_to_goal(traj_xy, goal_xy):
    """Angle (deg) between trajectory endpoint heading and goal direction, per sample."""
    th_t = np.arctan2(traj_xy[:, 1], traj_xy[:, 0])
    th_g = np.arctan2(goal_xy[:, 1], goal_xy[:, 0])
    d = np.abs(np.arctan2(np.sin(th_t - th_g), np.cos(th_t - th_g)))
    return np.degrees(d)


def main():
    torch.manual_seed(0)
    policy = LoGoPlanner_Policy(context_size=12, device=DEV).to(DEV).eval()

    sd = torch.load(CKPT, map_location='cpu')
    sd = sd['state_dict'] if isinstance(sd, dict) and 'state_dict' in sd else sd
    if any(k.startswith('policy.') for k in sd):
        sd = {k[len('policy.'):]: v for k, v in sd.items() if k.startswith('policy.')}
    msd = policy.state_dict()
    sd = {k: v for k, v in sd.items()
          if not (k in msd and hasattr(v, 'shape') and tuple(v.shape) != tuple(msd[k].shape))}
    res = policy.load_state_dict(sd, strict=False)
    print(f'[load] missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}')

    ds = LoGoPlanner_Dataset(
        ROOT, preload_path=CACHE, memory_size=8, predict_size=24, batch_size=N_SAMPLES,
        image_size=224, context_size=12, context_image_height=168, context_image_width=308,
        depth_max=5.0, depth_min=0.1, multistop=True, streaming=True,
        n_anchor=N_ANCHOR, n_traj=16, n_window=N_WINDOW,
    )
    print(f'[data] {len(ds)} samples; drawing {N_SAMPLES}')
    samples = [ds[i] for i in range(N_SAMPLES)]
    batch = logoplanner_collate_fn(samples)
    ctx_rgb = batch['batch_context_rgb'].to(DEV, torch.float32)
    ctx_depth = batch['batch_context_depth'].to(DEV, torch.float32)
    pg = batch['batch_pg'].to(DEV, torch.float32)        # (B,3) goal in current chassis frame
    labels = batch['batch_labels'].to(DEV, torch.float32)  # (B,24,3) GT action deltas
    B = pg.shape[0]

    # Process ONE sample at a time: the streaming KV cache is single-stream (B=1).
    traj_mg_list, traj_ng_list = [], []
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for b in range(B):
            (_, state, scene), _ = policy.state_encoder.encode_window_streaming(
                ctx_rgb[b:b + 1], ctx_depth[b:b + 1])
            state, scene = state.float(), scene.float()
            parts = partition_window_tokens(state, scene, N_ANCHOR, N_WINDOW)
            summary = policy.gct_assembler(parts)          # (1,8,D)
            cur = state[:, -1:]
            sg = policy.start_encoder(pg[b:b + 1]).unsqueeze(1)
            state_embed = policy.state_decoder(torch.cat([cur, sg], dim=1))  # (1,1,D)

            def sample(goal_embed):
                na = torch.randn(1, policy.predict_size, 3, device=DEV)
                policy.noise_scheduler.set_timesteps(policy.noise_scheduler.config.num_train_timesteps)
                for k in policy.noise_scheduler.timesteps:
                    npred = policy.predict_noise(na, k.unsqueeze(0), goal_embed, None, None, summary=summary)
                    na = policy.noise_scheduler.step(model_output=npred, timestep=k, sample=na).prev_sample
                return torch.cumsum(na / 4.0, dim=1)        # (1,24,3) waypoints

            traj_mg_list.append(sample(state_embed).float().cpu().numpy())
            traj_ng_list.append(sample(torch.zeros_like(state_embed)).float().cpu().numpy())
    traj_mg = np.concatenate(traj_mg_list, axis=0)          # (B,24,3)
    traj_ng = np.concatenate(traj_ng_list, axis=0)

    # GT trajectory in waypoint space (labels are *4 deltas -> integrate /4)
    gt_traj = torch.cumsum(labels / 4.0, dim=1).float().cpu().numpy()
    pg_xy = pg[:, :2].cpu().numpy()

    a_gt = angle_to_goal(gt_traj[:, -1, :2], pg_xy)
    a_mg = angle_to_goal(traj_mg[:, -1, :2], pg_xy)
    a_ng = angle_to_goal(traj_ng[:, -1, :2], pg_xy)
    len_mg = np.linalg.norm(traj_mg[:, -1, :2], axis=-1)
    len_ng = np.linalg.norm(traj_ng[:, -1, :2], axis=-1)
    # how much does the goal change the trajectory?
    mg_vs_ng = np.linalg.norm(traj_mg[:, -1, :2] - traj_ng[:, -1, :2], axis=-1)
    goal_dist = np.linalg.norm(pg_xy, axis=-1)

    print('\n================ GOAL-FOLLOWING DIAGNOSTIC ================')
    print(f'samples={B}  goal_dist mean={goal_dist.mean():.2f}m')
    print(f'GT  traj->goal angle:  mean={a_gt.mean():5.1f} deg  median={np.median(a_gt):5.1f}  (small => dataset goal convention OK)')
    print(f'mg  traj->goal angle:  mean={a_mg.mean():5.1f} deg  median={np.median(a_mg):5.1f}  (goal-conditioned prediction)')
    print(f'ng  traj->goal angle:  mean={a_ng.mean():5.1f} deg  median={np.median(a_ng):5.1f}  (no-goal baseline; ~90 random)')
    print(f'mg endpoint len mean={len_mg.mean():.2f}m   ng={len_ng.mean():.2f}m')
    print(f'|mg-ng| endpoint shift mean={mg_vs_ng.mean():.2f}m  (large => goal actually changes output)')
    print('\nVERDICT:')
    if a_gt.mean() > 60:
        print('  ** GT angle large -> streaming dataset pg convention is WRONG (fix dataset goal).')
    elif a_mg.mean() < 45 and a_mg.mean() < a_ng.mean() - 15:
        print('  ** Model FOLLOWS the goal (mg≪ng). The IsaacSim eval goal-frame transform is the bug.')
    elif mg_vs_ng.mean() < 0.2:
        print('  ** Goal is IGNORED (mg≈ng). Conditioning/training issue — needs arch fix or more training.')
    else:
        print('  ** Weak/partial goal-following. Likely undertrained or weak goal signal.')
    print('==========================================================')


if __name__ == '__main__':
    main()
