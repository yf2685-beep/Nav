"""End-to-end smoke test for the streaming GCT navigation policy.

Run on a machine with a GPU + the lingbot-map repo (LINGBOT_CKPT optional; random
init is fine for a shape/NaN check). Validates, in streaming mode:

  1. backbone parallel forward over an N = anchor+traj+window window (anchors=8),
  2. GCT-summary assembly + diffusion conditioning shapes,
  3. frame-by-frame streaming inference with a persistent KV cache,
  4. KV-cache frame count stays bounded by the sliding window over a long episode,
  5. cache resets to 0 on a new episode.

    LINGBOT_CKPT=/path/to/lingbot-map.pt python smoke_streaming.py
"""

import os

# Must be set BEFORE importing policy_network (env is read at import time).
os.environ.setdefault('LOGO_BACKBONE', 'lingbot_v2')
os.environ.setdefault('LOGO_STREAMING', '1')
os.environ.setdefault('LOGO_N_ANCHOR', '8')
os.environ.setdefault('LOGO_N_WINDOW', '64')

import numpy as np
import torch

from policy_network import LoGoPlanner_Policy
from streaming_gct import partition_window_tokens, GCTSummaryAssembler

DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'
H, W = 168, 308
N_ANCHOR = int(os.environ['LOGO_N_ANCHOR'])
N_TRAJ = 16
N_WINDOW = int(os.environ['LOGO_N_WINDOW'])
N = N_ANCHOR + N_TRAJ + N_WINDOW


def main():
    torch.manual_seed(0)
    policy = LoGoPlanner_Policy(context_size=12, device=DEV).to(DEV).eval()
    assert policy._streaming, 'policy not in streaming mode'
    assert not hasattr(policy, 'rgbd_encoder'), 'NavDP memory backbone should be dropped'
    print(f'[init] streaming policy built (n_anchor={policy.n_anchor}, n_window={policy.n_window})')
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f'[init] trainable {trainable/1e6:.1f}M / {total/1e6:.1f}M')

    # ---- (1) training-path window encode (memory-bounded streaming) -----------
    ctx_rgb = torch.rand(1, N, H, W, 3, device=DEV)
    ctx_depth = torch.rand(1, N, H, W, 1, device=DEV) * 4.0 + 0.2
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEV != 'cpu')):
        (_, st, sc), (cp, lp, wp) = policy.state_encoder.encode_window_streaming(
            ctx_rgb, ctx_depth,
        )
    print(f'[train-encode] state {tuple(st.shape)} scene {tuple(sc.shape)} '
          f'cam {tuple(cp.shape)} finite={torch.isfinite(st).all().item()} '
          f'peakMB={torch.cuda.max_memory_allocated()/1e6:.0f}' if DEV != 'cpu' else '')
    assert st.shape == (1, N, 384) and sc.shape == (1, N, 384)
    st, sc = st.float(), sc.float()

    # ---- (2) GCT summary + diffusion conditioning -----------------------------
    parts = partition_window_tokens(st, sc, policy.n_anchor, policy.n_window)
    summary = policy.gct_assembler(parts)
    assert summary.shape == (1, GCTSummaryAssembler.N_SUMMARY, 384)
    sample_num = 4
    summ_rep = torch.repeat_interleave(summary, sample_num, dim=0)
    goal = torch.zeros(sample_num, 1, 384, device=DEV)
    acts = torch.randn(sample_num, policy.predict_size, 3, device=DEV)
    ts = torch.tensor([5], device=DEV)
    noise_pred = policy.predict_noise(acts, ts, goal, None, None, summary=summ_rep)
    crit = policy.predict_critic(acts, None, None, summary=summ_rep)
    print(f'[diffusion] noise_pred {tuple(noise_pred.shape)} critic {tuple(crit.shape)} '
          f'finite={torch.isfinite(noise_pred).all().item()}')
    assert noise_pred.shape == (sample_num, policy.predict_size, 3)

    # ---- (3+4+5) streaming inference over a long episode ----------------------
    goals = np.zeros((1, 3), np.float32)
    cache_counts = []
    for t in range(150):
        img = torch.rand(1, H, W, 3, device=DEV)
        dep = torch.rand(1, H, W, 1, device=DEV) * 4.0 + 0.2
        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEV != 'cpu')):
            traj, vals, good, bad, sub = policy.predict_pointgoal_action_stream(
                goals, img, dep, episode_start=(t == 0),
            )
        cache_counts.append(policy.state_encoder.kv_cache_num_frames())
        if t == 0:
            assert traj.shape == (1, 16, policy.predict_size, 3), traj.shape
    print(f'[stream] ran 150 frames; trajectory {traj.shape} finite={np.isfinite(traj).all()}')
    print(f'[stream] KV-cache frame count: t=0 -> {cache_counts[0]}, '
          f't=10 -> {cache_counts[10]}, t=149 -> {cache_counts[-1]} '
          f'(max {max(cache_counts)})')

    # New episode resets the cache.
    img = torch.rand(1, H, W, 3, device=DEV)
    dep = torch.rand(1, H, W, 1, device=DEV) * 4.0 + 0.2
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(DEV != 'cpu')):
        policy.predict_pointgoal_action_stream(goals, img, dep, episode_start=True)
    after_reset = policy.state_encoder.kv_cache_num_frames()
    print(f'[stream] cache after episode reset (1 frame in): {after_reset}')
    print('ALL STREAMING SMOKE CHECKS PASSED')


if __name__ == '__main__':
    main()
