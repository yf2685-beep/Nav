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
# Frames may live in a read-only squashfs overlay (mp3d pt1.sqf) while caches are
# written to a SEPARATE writable tree — MEMNAV_FEATURE_ROOT points at that tree
# (None = old behavior: cache sits beside the frames). window/num_scale/max_frame_num
# MUST match how the caches were precomputed (mp3d: window=32, num_scale=8, mfn=2048).
_FEATURE_ROOT = os.environ.get('MEMNAV_FEATURE_ROOT') or None
_WINDOW_SIZE = int(os.environ.get('MEMNAV_WINDOW', '32'))
_NUM_SCALE = int(os.environ.get('MEMNAV_NUM_SCALE', '8'))
_MAX_FRAME_NUM = int(os.environ.get('MEMNAV_MAX_FRAME_NUM', '2048'))
# Restrict to episodes with n_legs <= MEMNAV_MAX_LEGS (unset or 0 = keep all; 2 = two-leg only).
_MAX_LEGS = int(os.environ.get('MEMNAV_MAX_LEGS') or 0) or None
# Step-based checkpointing: save every N optimizer steps so a wall-clock timeout banks
# recent progress (epoch-based saves never fired — runs die mid-epoch-0). Non-None here
# switches train.py to save_strategy='steps' for memnav (other models stay on 'epoch').
_SAVE_STEPS = int(os.environ.get('MEMNAV_SAVE_STEPS', '100'))
# RevisitMerge.aux_pose_head calibration: 'empirical' (frozen at the fitted axis+scale
# constant) or 'trainable' (same init, but its own weight/bias adapt via w_aux_pose*aux_loss
# -- see RevisitMerge's docstring for why gradient reaches this head even though the
# upstream camera poses are frozen). Both remain a per-video-scale-ambiguity-limited
# diagnostic, not a precision signal -- see the ground-anchored-scale TODO there.
_AUX_POSE_CALIBRATION = os.environ.get('MEMNAV_AUX_POSE_CALIBRATION', 'empirical')

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
        save_interval_steps=_SAVE_STEPS,
        save_filter_frozen_weights=True,
        load_from_ckpt=False,
        ckpt_to_load='',
        report_to=os.environ.get('MEMNAV_REPORT_TO', 'wandb'),
        # data + frozen-LingBot paths (override via MEMNAV_ROOT_DIR / LINGBOT_REPO / LINGBOT_WEIGHTS)
        root_dir=_ROOT_DIR,
        feature_root=_FEATURE_ROOT,
        lingbot_repo=_LINGBOT_REPO,
        lingbot_weights=_LINGBOT_WEIGHTS,
        image_size=518,
        random_digit=False,
        # memory-partition geometry — MUST match the precompute (mp3d: 32/8/2048).
        # Read by MemNav_Dataset (window_size/num_scale) and LingBotStream (window/
        # num_scale/max_frame_num) so training reproduces the cached streaming exactly.
        window_size=_WINDOW_SIZE,
        num_scale=_NUM_SCALE,
        max_frame_num=_MAX_FRAME_NUM,
        # episode leg filter (None = all legs; 2 = two-leg episodes only)
        max_legs=_MAX_LEGS,
        # goal_append_warm's live-recompute depth before streaming the goal (deeper than
        # window_size on purpose): window_size's cold start at the window boundary starves
        # the goal's pose estimate (no real predecessors); goal_warm=64 empirically matches
        # a true continuous-stream oracle (scripts/diag_lingbot_pose_accuracy.py), while the
        # nominal window leaves ~30% avoidable error on the table. Eviction stays at
        # window_size — the model's own KV eviction trims back to that during the warm
        # recompute, which measured the same accuracy as never evicting at all.
        goal_warm=64,
        aux_pose_calibration=_AUX_POSE_CALIBRATION,
        # policy / diffusion
        predict_size=24,
        temporal_depth=8,
        heads=8,
        token_dim=384,
        num_diffusion_iters=10,
        # loss weights (consumed by MemNavTrainer)
        w_retrieval=1.0,   # ranking InfoNCE (which candidate frame matches)
        w_gate=1.0,        # revisit/novel gate BCE (is there a match at all)
        w_aux_pose=0.5,
        ddp_find_unused_parameters=True,
    ),
    model=memnav_cfg,
)
