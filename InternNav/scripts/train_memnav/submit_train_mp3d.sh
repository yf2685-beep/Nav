#!/bin/bash
# Submit MemNav mp3d training with the BATCHED encode_memory path.
#
# The batched path (MEMNAV_STREAM_GROUP>1) runs G trajectory streams on the aggregator's
# batch dim instead of a per-sample Python loop — ~Gx fewer frozen-LingBot forwards/step.
# It is numerically equivalent to the old path (validated by scripts/diag_batched_window.py).
#
# Usage:
#   bash scripts/train_memnav/submit_train_mp3d.sh
#   MEMNAV_STREAM_GROUP=8 BATCH_SIZE=16 EPOCHS=10 bash scripts/train_memnav/submit_train_mp3d.sh
#   PARTITION=a100_tandon NODELIST=ga001 bash scripts/train_memnav/submit_train_mp3d.sh
#
# Env overrides (all optional; defaults set inside the sbatch):
#   MEMNAV_STREAM_GROUP  batched-stream group size (default 4; 1 = old per-sample path)
#   NAME BATCH_SIZE EPOCHS NUM_WORKERS MEMNAV_REPORT_TO   training knobs
#   PARTITION NODELIST ACCOUNT TIME                        SLURM targeting overrides
set -euo pipefail

REPO=/scratch/lg154/Research/Nav/InternNav
SBATCH_FILE="${REPO}/scripts/train_memnav/train_memnav_mp3d.sbatch"

# export training/model knobs so --export=ALL carries them into the job
export MEMNAV_STREAM_GROUP="${MEMNAV_STREAM_GROUP:-4}"
export NAME="${NAME:-memnav_mp3d_g${MEMNAV_STREAM_GROUP}}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export EPOCHS="${EPOCHS:-10}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MEMNAV_REPORT_TO="${MEMNAV_REPORT_TO:-wandb}"

# optional SLURM targeting overrides (empty = use the #SBATCH directives in the file)
EXTRA=()
[[ -n "${PARTITION:-}" ]] && EXTRA+=(--partition="${PARTITION}")
[[ -n "${NODELIST:-}"  ]] && EXTRA+=(--nodelist="${NODELIST}")
[[ -n "${ACCOUNT:-}"   ]] && EXTRA+=(--account="${ACCOUNT}")
[[ -n "${TIME:-}"      ]] && EXTRA+=(--time="${TIME}")

echo "submitting: NAME=${NAME} STREAM_GROUP=${MEMNAV_STREAM_GROUP} BATCH_SIZE=${BATCH_SIZE} EPOCHS=${EPOCHS} report=${MEMNAV_REPORT_TO}"
[[ ${#EXTRA[@]} -gt 0 ]] && echo "slurm overrides: ${EXTRA[*]}"

OUT="$(sbatch --export=ALL "${EXTRA[@]}" "${SBATCH_FILE}")"
echo "${OUT}"
JID="$(grep -oE '[0-9]+' <<<"${OUT}" | tail -1)"
echo
echo "monitor:"
echo "  squeue -j ${JID}"
echo "  tail -f ${REPO}/logs/train_memnav/mp3d-${JID}.out    # step loss / mem"
echo "  tail -f ${REPO}/logs/train_memnav/mp3d-${JID}.err"
