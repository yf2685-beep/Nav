#!/bin/bash
#SBATCH --job-name=memnav_train
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# MemNav single-GPU training / smoke on the InternData-N1 MP3D shard.
# Mirrors the `sbatch_lg` pattern in GR00T_cam: standalone .sbatch with
# #SBATCH directives + inline body.
#
# Consumes `--skip_scale` lingbot_cache.npz files under MEMNAV_ROOT_DIR
# (scale_k/v recomputed on the fly via LingBotStream.get_scale_kv).
#
# Overridable env vars (defaults shown):
#   NAME=memnav_smoke                    wandb run name + checkpoints/<NAME>/
#   BATCH_SIZE=2
#   EPOCHS=1
#   NUM_WORKERS=0
#   MEMNAV_ROOT_DIR=/scratch/lg154/Research/datasets/InternData-N1/vln_n1/_raw
#   LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
#   LINGBOT_WEIGHTS=$LINGBOT_REPO/weights/lingbot-map-long.pt
#   WANDB_PROJECT=memnav
#   MEMNAV_REPORT_TO=wandb   (wandb | tensorboard | none)
#
# Submit examples:
#   sbatch InternNav/scripts/train_memnav/train_memnav.sbatch
#   NAME=memnav_smoke_skip_scale BATCH_SIZE=2 EPOCHS=1 \
#       sbatch --export=ALL InternNav/scripts/train_memnav/train_memnav.sbatch
#
# Interactive rerun inside an existing salloc:
#   NAME=memnav_smoke_skip_scale \
#     srun --overlap --jobid=<JOBID> --mem=0 \
#          bash InternNav/scripts/train_memnav/train_memnav.sbatch

set -euo pipefail

mkdir -p /scratch/lg154/Research/Nav/InternNav/logs/train_memnav

REPO_ROOT=${REPO_ROOT:-/scratch/lg154/Research/Nav/InternNav}
cd "${REPO_ROOT}"

# --- python env ---
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/memnav

# --- paths + PYTHONPATH ---
export LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
export LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
export MEMNAV_ROOT_DIR=${MEMNAV_ROOT_DIR:-/scratch/lg154/Research/datasets/InternData-N1/vln_n1/_raw}
export PYTHONPATH="${REPO_ROOT}/src/diffusion-policy:${PYTHONPATH:-}"

# --- per-job scratch (isolated TMP / HOME / XDG so concurrent jobs don't collide) ---
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/memnav_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${TMPDIR}" TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

# --- runtime knobs ---
# glibc: cap per-thread malloc arenas — big SLURM nodes have 96+ cores and
# Python threading libs (OpenMP/BLAS) trigger a per-thread arena each, which
# balloons virtual + resident heap on model init. THE common "why does my
# training job OOM on a big node" fix.
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
THREADS_PER_PROC=$(( ${SLURM_CPUS_PER_TASK:-4} ))
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(( THREADS_PER_PROC > 4 ? 4 : THREADS_PER_PROC ))}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}

# --- training config ---
NAME=${NAME:-memnav_smoke}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

# --- wandb (per-job dirs so concurrent runs don't collide) ---
export MEMNAV_REPORT_TO=${MEMNAV_REPORT_TO:-wandb}
if [[ "${MEMNAV_REPORT_TO}" == "wandb" ]]; then
  export WANDB_PROJECT=${WANDB_PROJECT:-memnav}
  export WANDB_NAME=${WANDB_NAME:-${NAME}}
  export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
  export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
  export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
  export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
  mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
fi

# --- pre-flight ---
[[ -f "${LINGBOT_WEIGHTS}" ]] \
  || { echo "Missing LingBot weights: ${LINGBOT_WEIGHTS}" >&2; exit 1; }
[[ -d "${MEMNAV_ROOT_DIR}" ]] \
  || { echo "Missing MemNav root: ${MEMNAV_ROOT_DIR}" >&2; exit 1; }

echo "===== TRAIN ENV CHECK ====="
echo "host:        $(hostname)"
which python; python -V
echo "torch:       $(python -c 'import torch; print(torch.__version__, "cuda", torch.cuda.is_available())')"
echo "exp:         ${NAME}"
echo "root:        ${MEMNAV_ROOT_DIR}"
echo "weights:     ${LINGBOT_WEIGHTS}"
echo "bs/ep/wk:    ${BATCH_SIZE} / ${EPOCHS} / ${NUM_WORKERS}"
echo "report_to:   ${MEMNAV_REPORT_TO} (project=${WANDB_PROJECT:-<n/a>} name=${WANDB_NAME:-<n/a>})"
nvidia-smi -L || true

exec python scripts/train/train.py \
    --name "${NAME}" \
    --model-name memnav \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --epochs "${EPOCHS}" \
    --root-dir "${MEMNAV_ROOT_DIR}"
