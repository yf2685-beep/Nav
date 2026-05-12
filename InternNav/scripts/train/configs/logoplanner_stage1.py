"""Stage 1 training config (paper §V.A): finetune the geometry decoder + task-
specific heads, ViT encoder frozen. Loss = L_pose + L_local + L_world only;
diffusion / critic / subgoal weights are zero.

Inherits from logoplanner_exp_cfg (single-stage all-loss baseline) and overrides
just the bits that differ.
"""

import copy

from internnav.configs.trainer.il import Loss

from .logoplanner import logoplanner_exp_cfg

logoplanner_stage1_exp_cfg = copy.deepcopy(logoplanner_exp_cfg)
logoplanner_stage1_exp_cfg.name = 'logoplanner_stage1'
logoplanner_stage1_exp_cfg.model_name = 'logoplanner'

logoplanner_stage1_exp_cfg.il.loss = Loss(
    alpha=0.0001,
    dist_scale=1,
    w_diffusion=0.0,
    w_critic=0.0,
    w_pose=1.0,
    w_local=0.5,
    w_world=0.5,
    w_subgoal=0.0,
    stage=1,
)
