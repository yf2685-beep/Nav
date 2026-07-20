#!/bin/bash
#SBATCH --job-name=logo_rgb_multistop
#SBATCH --partition=spark            # same partition as the prior lingbot_v2 runs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00              # 1699 steps (~1 epoch on the mini set) fits well under 4h
#SBATCH --output=logo_rgb_multistop_%j.log
set -x

# ============================================================================
# Stage 1-7 LoGoPlanner run: RGB-only policy + LingBot backbone + multi-stop
# subgoal data. (collision-critic Stage 7 stays OFF here: w_safety=0 by default.)
#
# Config defaults already baked in (scripts/train/configs/logoplanner.py):
#   il.use_depth=False   il.multistop=True   il.sequential=False
# So this script only needs the env, the LingBot ckpt, and the data root.
# ============================================================================

# --- FILL IN: activate the training env (copy these lines from your existing
#     /media/cvpr/yuxuan/logoplanner_setup/lingbot_v2_stage2_stageA.sh) ---
# e.g.  source /media/cvpr/yuxuan/envs/enerverse_arm/bin/activate
#  or   source activate enerverse_arm
source activate enerverse_arm        # <-- adjust to your enerverse_arm activation

# --- repo on 131 (per README): /media/cvpr/yuxuan/logoplanner/Nav ---
cd /media/cvpr/yuxuan/logoplanner/Nav

# --- LingBot backbone + ckpt (env vars are read at import time) ---
export LOGO_BACKBONE=lingbot_v2
export LINGBOT_CKPT=/media/cvpr/yuxuan/logoplanner/lingbot-map.pt   # <-- 131 path to lingbot-map.pt

# --- data root: dir that CONTAINS the group dir (matterport3d_d435i) ---
ROOT_DIR=/media/cvpr/yuxuan/logoplanner/data/InternData-N1/vln_n1/traj_data_navdp   # <-- 131 traj_data_navdp
DATASET_JSON=/media/cvpr/yuxuan/logoplanner/logoplanner_dataset_rgb_multistop.json  # written on first run, reused after

# --- (optional) warm-start from an existing stage-1 ckpt. The old ckpts were
#     trained WITH depth; the depth_model keys load as 'unexpected' (strict=False
#     drops them) — harmless. Omit --ckpt-to-load to train from scratch. ---
# CKPT=/media/cvpr/yuxuan/logoplanner/Nav/InternNav/checkpoints/logo_s1long/ckpts/checkpoint-XXXXlogoplanner.ckpt

python InternNav/scripts/train/train.py \
  --model-name logoplanner_stage2 \
  --epochs 1 \
  --batch-size 2 \
  --root-dir "$ROOT_DIR" \
  --dataset-navdp "$DATASET_JSON" \
  --report-to tensorboard
  # --ckpt-to-load "$CKPT" --load-from-ckpt True   # uncomment to warm-start

# Output: checkpoints/logoplanner_stage2/ckpts/checkpoint-<step>/...logoplanner.ckpt
# On the mini set (3398 traj, batch 2) 1 epoch ≈ 1699 steps → checkpoint-1699.
