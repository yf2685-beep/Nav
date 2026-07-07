import os

from internnav.configs.model.memnav import memnav_cfg
from internnav.configs.trainer.eval import EvalCfg
from internnav.configs.trainer.exp import ExpCfg
from internnav.configs.trainer.il import IlCfg

# HPC-friendly defaults: env vars override the developer-desktop paths so a
# SLURM job only needs `export MEMNAV_ROOT_DIR / LINGBOT_REPO / LINGBOT_WEIGHTS`.
_ROOT_DIR = os.environ.get(
    'MEMNAV_ROOT_DIR',
    '/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data',
)
_LINGBOT_REPO = os.environ.get(
    'LINGBOT_REPO',
    '/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map',
)
_LINGBOT_WEIGHTS = os.environ.get(
    'LINGBOT_WEIGHTS',
    '/home/asus/Research/Nav/NavDP/baselines/memnav/lingbot-map/weights/lingbot-map-long.pt',
)

memnav_exp_cfg = ExpCfg(
    name='memnav_train',
    model_name='memnav',
    torch_gpu_id=0,
    torch_gpu_ids=[0],
    output_dir='checkpoints/%s/ckpts',
    tensorboard_dir='checkpoints/%s/tensorboard',
    checkpoint_folder='checkpoints/%s/ckpts',
    log_dir='checkpoints/%s/logs',
    local_rank=0,
    seed=0,
    eval=EvalCfg(
        use_ckpt_config=False,
        save_results=True,
        split=['val_seen'],
        ckpt_to_load='',
        max_steps=195,
        sample=False,
        success_distance=3.0,
        start_eval_epoch=-1,
        step_interval=50,
    ),
    il=IlCfg(
        epochs=1000,
        batch_size=8,
        lr=1e-4,
        num_workers=4,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        save_interval_epochs=5,
        save_filter_frozen_weights=True,
        load_from_ckpt=False,
        ckpt_to_load='',
        report_to=os.environ.get('MEMNAV_REPORT_TO', 'wandb'),
        # data + frozen-LingBot paths (override via MEMNAV_ROOT_DIR / LINGBOT_REPO / LINGBOT_WEIGHTS)
        root_dir=_ROOT_DIR,
        lingbot_repo=_LINGBOT_REPO,
        lingbot_weights=_LINGBOT_WEIGHTS,
        image_size=518,
        random_digit=False,
        # policy / diffusion
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        num_diffusion_iters=10,
        # loss weights (consumed by MemNavTrainer)
        w_retrieval=1.0,
        w_aux_pose=0.5,
        ddp_find_unused_parameters=True,
    ),
    model=memnav_cfg,
)
