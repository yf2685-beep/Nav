"""Stage 2 training config (paper §V.A): freeze the geometry-backbone decoder,
train the diffusion head + task-specific heads jointly. Geometry losses are
zeroed; diffusion / critic / subgoal carry the gradient.
"""

import copy

from internnav.configs.trainer.il import Loss

from .logoplanner import logoplanner_exp_cfg

logoplanner_stage2_exp_cfg = copy.deepcopy(logoplanner_exp_cfg)
logoplanner_stage2_exp_cfg.name = 'logoplanner_stage2'
logoplanner_stage2_exp_cfg.model_name = 'logoplanner'

logoplanner_stage2_exp_cfg.il.loss = Loss(
    alpha=0.0001,
    dist_scale=1,
    w_diffusion=1.0,
    w_critic=1.0,
    w_pose=0.0,
    w_local=0.0,
    w_world=0.0,
    w_subgoal=0.1,
    stage=2,
)
