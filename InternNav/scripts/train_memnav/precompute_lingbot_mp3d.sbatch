#!/bin/bash
#SBATCH --job-name=memnav_precompute
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/precompute_lingbot/slurm-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/precompute_lingbot/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Precompute LingBot --skip_scale caches for half of the local MP3D shard
# on a single A100. NUM_SHARDS=2, SHARD=0 = interleaved half (~1699/3398 trajs).
# At ~15 s/traj this takes ~7 h; --time=10:00:00 gives 3 h of headroom.
#
# The precompute script skips any traj whose lingbot_cache.npz already
# exists, so:
#   * The 200-traj smoke shard is skipped for free.
#   * To fetch the other half later, re-submit with SHARD=1.
#
# Overridable env vars (defaults shown):
#   NUM_SHARDS=2   How the full traj list is split (interleaved).
#   SHARD=0        Which half to run this time (0 or 1).
#   MP3D_ROOT=/scratch/lg154/.../vln_n1/_raw
#   LINGBOT_REPO=/scratch/lg154/.../lingbot-map
#   LINGBOT_WEIGHTS=$LINGBOT_REPO/weights/lingbot-map-long.pt
#
# Submit examples:
#   sbatch InternNav/scripts/train_memnav/precompute_lingbot_mp3d.sbatch
#   SHARD=1 sbatch InternNav/scripts/train_memnav/precompute_lingbot_mp3d.sbatch

set -euo pipefail

mkdir -p /scratch/lg154/Research/Nav/InternNav/logs/precompute_lingbot

REPO_ROOT=${REPO_ROOT:-/scratch/lg154/Research/Nav/InternNav}
cd "${REPO_ROOT}"

# --- python env ---
source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/lg154/conda-envs/memnav

# --- paths + PYTHONPATH ---
export LINGBOT_REPO=${LINGBOT_REPO:-/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map}
export LINGBOT_WEIGHTS=${LINGBOT_WEIGHTS:-${LINGBOT_REPO}/weights/lingbot-map-long.pt}
export MP3D_ROOT=${MP3D_ROOT:-/scratch/lg154/Research/datasets/InternData-N1/vln_n1/_raw}
export PYTHONPATH="${REPO_ROOT}/src/diffusion-policy:${LINGBOT_REPO}:${PYTHONPATH:-}"

# --- per-job scratch ---
JOB_TMP_ROOT="${SLURM_TMPDIR:-/tmp/$USER}/precompute_${SLURM_JOB_ID:-$$}"
export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${TMPDIR}" TEMP="${TMPDIR}"
export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}"

# --- runtime knobs ---
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-${OMP_NUM_THREADS}}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# --- sharding ---
NUM_SHARDS=${NUM_SHARDS:-2}
SHARD=${SHARD:-0}

# --- pre-flight ---
[[ -f "${LINGBOT_WEIGHTS}" ]] \
  || { echo "Missing LingBot weights: ${LINGBOT_WEIGHTS}" >&2; exit 1; }
[[ -d "${MP3D_ROOT}" ]] \
  || { echo "Missing shard root: ${MP3D_ROOT}" >&2; exit 1; }

echo "===== PRECOMPUTE ENV CHECK ====="
echo "host:        $(hostname)"
echo "shard:       ${SHARD} / ${NUM_SHARDS}"
echo "root:        ${MP3D_ROOT}"
echo "weights:     ${LINGBOT_WEIGHTS}"
which python; python -V
python -c 'import torch; print("torch:", torch.__version__, "cuda", torch.cuda.is_available())'
nvidia-smi -L || true

exec python scripts/dataset_converters/precompute_lingbot_features.py \
    --root_dirs "${MP3D_ROOT}" \
    --lingbot_repo "${LINGBOT_REPO}" \
    --weights "${LINGBOT_WEIGHTS}" \
    --image_size 518 \
    --num_scale_frames 8 \
    --use_sdpa \
    --skip_scale \
    --num_shards "${NUM_SHARDS}" \
    --shard "${SHARD}"
