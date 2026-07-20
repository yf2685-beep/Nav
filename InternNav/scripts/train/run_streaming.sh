#!/bin/bash
# Streaming GCT LoGoPlanner training (single GPU). Run from InternNav/:
#   bash scripts/train/run_streaming.sh
#
# Streams each episode through the frozen LingBot GCT backbone (anchor 8 +
# window 64 + trajectory memory + KV cache); the diffusion head conditions on
# GCT-summary tokens. Drops the NavDP 8-frame memory backbone.
#
# Override any var on the command line, e.g.:
#   BATCH_SIZE=2 EPOCHS=3 ROOT_DIR=/path bash scripts/train/run_streaming.sh
set -eo pipefail

NAME=${NAME:-logoplanner_streaming}
MODEL=logoplanner_streaming                 # registered in train.py -> streaming config
BATCH_SIZE=${BATCH_SIZE:-1}                  # streaming window N=88 frames/sample; start small
EPOCHS=${EPOCHS:-1}
# Streaming loads N frames/sample (~2N CIFS reads) — use workers to prefetch in
# parallel, else the dataloader (not the GPU) is the bottleneck.
NUM_WORKERS=${NUM_WORKERS:-8}
# MAX_STEPS>0 caps optimizer steps (overrides EPOCHS) — use for a quick sanity.
MAX_STEPS=${MAX_STEPS:-0}
MAXSTEP_ARG=""; [ "${MAX_STEPS:-0}" -gt 0 ] 2>/dev/null && MAXSTEP_ARG="--max-steps $MAX_STEPS"
# LAMBDA_CRITIC overrides the critic loss weight (default in config = 1.0). Prior
# experiments show a smaller critic weight (e.g. 0.3) is more stable / better SR.
LAMBDA_CRITIC=${LAMBDA_CRITIC:-}
LAMBDACRIT_ARG=""; [ -n "$LAMBDA_CRITIC" ] && LAMBDACRIT_ARG="--lambda-critic $LAMBDA_CRITIC"

# --- paths (131); override if yours differ -----------------------------------
# ROOT_DIR is the PARENT of the dataset group: loader globs root/GROUP/SCENE/TRAJ
# (here GROUP=matterport3d_d435i). Do NOT point it at matterport3d_d435i itself.
ROOT_DIR=${ROOT_DIR:-/media/cvpr/yuxuan/logoplanner/data/mini_clean}
DATASET_CACHE=${DATASET_CACHE:-/tmp/logoplanner_stream_dataset.json}
# Warm start: the LingBot aggregator is always loaded from LINGBOT_CKPT; this
# ckpt supplies the DA-S depth prior + fusion + diffusion head. MUST be a .ckpt
# FILE (not a dir) — LoGoPlannerNet.from_pretrained uses strict=False only for a
# file, so the streaming-specific gct_assembler inits fresh and the resized
# cond_pos_embed (len 12) is dropped cleanly. A DIRECTORY path takes the
# strict=True branch and will CRASH on the new/changed keys. lingbot_v2_distill
# is the lingbot_v2 lineage so its DA-S/fusion/heads keys match. Set
# LOAD_FROM_CKPT=False to train diffusion+assembler from scratch (backbone still
# from LINGBOT_CKPT, but DA-S depth prior would then be random — not recommended).
CKPT_TO_LOAD=${CKPT_TO_LOAD:-/media/cvpr/yuxuan/logoplanner/checkpoints/lingbot_v2_distill_step3000_stripped.ckpt}
LOAD_FROM_CKPT=${LOAD_FROM_CKPT:-True}

# --- streaming GCT switches (MUST match configs/logoplanner_streaming.py) -----
export LOGO_BACKBONE=lingbot_v2
export LOGO_STREAMING=1
export LOGO_N_ANCHOR=${LOGO_N_ANCHOR:-8}
export LOGO_N_WINDOW=${LOGO_N_WINDOW:-64}
export LINGBOT_CKPT=${LINGBOT_CKPT:-/media/cvpr/yuxuan/logoplanner/lingbot-map-ckpt/lingbot-map.pt}
# Stage-1 geometry supervision over the window (needs a big GPU): set this to 1.
# Default (unset) = Stage-2 streaming, frozen backbone under no_grad, bounded mem.
# export LOGO_STREAM_BACKBONE_GRAD=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1   # flush loss/getitem prints to the slurm log immediately
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_SHOW_CPP_STACKTRACES=1
# vendored diffusion_policy submodule (same as runs.sh)
export PYTHONPATH="$PWD/src/diffusion-policy:${PYTHONPATH:-}"

export LOGO_N_TRAJ=${LOGO_N_TRAJ:-16}
echo "[run_streaming] MODEL=$MODEL N=$((LOGO_N_ANCHOR + LOGO_N_TRAJ + LOGO_N_WINDOW)) frames/window  workers=$NUM_WORKERS  max_steps=$MAX_STEPS"
echo "[run_streaming] ROOT_DIR=$ROOT_DIR"
echo "[run_streaming] LINGBOT_CKPT=$LINGBOT_CKPT"

python scripts/train/train.py \
    --name "$NAME" \
    --model-name "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --root-dir "$ROOT_DIR" \
    --dataset-navdp "$DATASET_CACHE" \
    --ckpt-to-load "$CKPT_TO_LOAD" \
    --load-from-ckpt "$LOAD_FROM_CKPT" \
    $MAXSTEP_ARG $LAMBDACRIT_ARG
