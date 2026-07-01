"""Streaming GCT training config.

Trains the LoGoPlanner diffusion policy on a *streaming* geometric context:
the LingBot GCT backbone (frozen) sees a bounded per-episode window
``[anchor (n_anchor) | trajectory keyframes (n_traj) | recent window (n_window)]``
and the diffusion head conditions on GCT-summary tokens instead of 12
subsampled frames. The NavDP 8-frame memory backbone is dropped.

MUST be paired with these environment variables (the policy reads them at
construction; the dataset reads the matching fields from this config — keep them
in sync):

    export LOGO_BACKBONE=lingbot_v2
    export LOGO_STREAMING=1
    export LOGO_N_ANCHOR=8
    export LOGO_N_WINDOW=64
    export LINGBOT_CKPT=/path/to/lingbot-map.pt   # frozen backbone weights

Launch:
    python InternNav/scripts/train/train.py --model-name logoplanner_streaming \
        --ckpt-to-load <warmstart_or_stage1> --load-from-ckpt True
"""

import copy

from internnav.configs.trainer.il import Loss

from .logoplanner import logoplanner_exp_cfg

logoplanner_streaming_exp_cfg = copy.deepcopy(logoplanner_exp_cfg)
logoplanner_streaming_exp_cfg.name = 'logoplanner_streaming'
logoplanner_streaming_exp_cfg.model_name = 'logoplanner'

# --- Streaming GCT window (must match LOGO_N_ANCHOR / LOGO_N_WINDOW env) ------
logoplanner_streaming_exp_cfg.il.streaming = True
logoplanner_streaming_exp_cfg.il.n_anchor = 8     # anchor context (scale frames)
logoplanner_streaming_exp_cfg.il.n_traj = 16      # trajectory-memory keyframes
logoplanner_streaming_exp_cfg.il.n_window = 64    # local pose-reference window
# Total geometry-backbone window N = 8 + 16 + 64 = 88 frames per sample.

# Stream episodes in temporal order so each window is a real prefix of one
# episode (the geometry GT / window indices stay causal).
logoplanner_streaming_exp_cfg.il.sequential = True
logoplanner_streaming_exp_cfg.il.seq_stride = 1

# Diffusion-stage losses (geometry GT now has N frames and lines up with the
# streaming preds, so pose/local/world CAN be enabled; default to the Stage-2
# diffusion focus with the frozen, pretrained backbone).
logoplanner_streaming_exp_cfg.il.loss = Loss(
    alpha=0.0001,
    dist_scale=1,
    w_diffusion=1.0,
    w_critic=1.0,
    w_pose=0.0,
    w_local=0.0,
    w_world=0.0,
    w_subgoal=0.1,
    stage=2,
    w_safety=0.0,
    safety_footprint=0.3,
    safety_margin=0.3,
    safety_max_points=1024,
)
