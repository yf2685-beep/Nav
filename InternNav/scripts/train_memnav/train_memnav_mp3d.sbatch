#!/bin/bash
#SBATCH --job-name=memnav_mp3d
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/mp3d-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/mp3d-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
# REQUIRED: complete scripts/train_memnav/README.md before every sbatch submission.
#
# MemNav training on the PACKED mp3d revisit data (scenes 0-53).
#
# Frames live in a read-only squashfs (pt1.sqf); the LingBot KV caches live in a
# SEPARATE writable tree (precompute_mp3d_pt1.sbatch wrote them there via --out_root).
# So we run training INSIDE the same apptainer overlay used for precompute:
#   apptainer exec --overlay pt1.sqf:ro <base.sif>
# Inside the container the frames appear at /mp3d_revisit_v0/vln_n1/traj_data, which is
# MEMNAV_ROOT_DIR; caches are read from MEMNAV_FEATURE_ROOT on host /scratch. The
# dataset only enumerates trajectories that HAVE a cache (memnav_dataset_lerobot.py
# skips uncached ones), so this is safe to run while shards 4-7 are still precomputing —
# it trains on whatever is ready (currently ~1788 of 1944 trajectories).
#
# CONSISTENCY (critical): the memory partition geometry MUST match the precompute.
# The caches were built with --kv_cache_sliding_window 32 and --max_frame_num 2048, so:
#   MEMNAV_WINDOW=32   -> MemNav_Dataset.window_size AND LingBotStream.window (== KV
#                          manager sliding window). Using 8 here would mis-partition the
#                          scale/history/window regions vs. how anchor_k/v was captured.
#   MEMNAV_NUM_SCALE=8 -> scale frames [0..7].
#   MEMNAV_MAX_FRAME_NUM=2048 -> sizes the 3D-RoPE table for long (>1024-frame) 3leg
#                          episodes; LingBotStream now extends the camera-head rope too.
# These three flow config(memnav.py) -> dataset + LingBotStream via env vars.
#
# Overridable env vars (defaults shown):
#   NAME=memnav_mp3d
#   BATCH_SIZE=4
#   EPOCHS=1
#   NUM_WORKERS=4
#   MEMNAV_REPORT_TO=wandb   (wandb | tensorboard | none)
#
# Submit:
#   sbatch InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch
#   NAME=memnav_mp3d_smoke BATCH_SIZE=2 EPOCHS=1 NUM_WORKERS=0 MEMNAV_REPORT_TO=none \
#       sbatch --export=ALL InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch
set -euo pipefail

mkdir -p /scratch/lg154/Research/Nav/InternNav/logs/train_memnav

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REPO_ROOT=/scratch/lg154/Research/Nav/InternNav

# in-container overlay path (frames) + host feature tree (caches)
export MEMNAV_ROOT_DIR=/mp3d_revisit_v0/vln_n1/traj_data
export MEMNAV_FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt

# memory-partition geometry — MUST match the precompute
export MEMNAV_WINDOW=32
export MEMNAV_NUM_SCALE=8
export MEMNAV_MAX_FRAME_NUM=2048

# training knobs
export NAME=${NAME:-memnav_mp3d}
export BATCH_SIZE=${BATCH_SIZE:-4}
export EPOCHS=${EPOCHS:-1}
export NUM_WORKERS=${NUM_WORKERS:-4}
export MEMNAV_REPORT_TO=${MEMNAV_REPORT_TO:-wandb}
export WANDB_PROJECT=${WANDB_PROJECT:-memnav}
export WANDB_NAME=${WANDB_NAME:-${NAME}}

echo "host: $(hostname); start: $(date)"
nvidia-smi -L || true

apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        ls "${MEMNAV_ROOT_DIR}" >/dev/null || { echo "ABORT: overlay missing ${MEMNAV_ROOT_DIR}"; exit 1; }
        [[ -d "${MEMNAV_FEATURE_ROOT}" ]] || { echo "ABORT: feature tree missing ${MEMNAV_FEATURE_ROOT}"; exit 1; }
        [[ -f "${LINGBOT_WEIGHTS}" ]]     || { echo "ABORT: weights missing ${LINGBOT_WEIGHTS}"; exit 1; }

        source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
        conda activate /scratch/lg154/conda-envs/memnav
        cd '"${REPO_ROOT}"'

        export PYTHONPATH="'"${REPO_ROOT}"'/src/diffusion-policy:${PYTHONPATH:-}"

        # per-job scratch (isolated TMP/XDG/wandb so concurrent jobs do not collide)
        JOB_TMP_ROOT="/tmp/${USER}/memnav_${SLURM_JOB_ID:-$$}"
        export TMPDIR="${JOB_TMP_ROOT}/tmp" TMP="${TMPDIR}" TEMP="${TMPDIR}"
        export XDG_CACHE_HOME="${JOB_TMP_ROOT}/xdg_cache"
        export XDG_CONFIG_HOME="${JOB_TMP_ROOT}/xdg_config"
        export XDG_STATE_HOME="${JOB_TMP_ROOT}/xdg_state"
        mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${XDG_STATE_HOME}"

        # big-node malloc / thread hygiene (see train_memnav.sbatch rationale)
        export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
        export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
        export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
        export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
        export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

        if [[ "${MEMNAV_REPORT_TO}" == "wandb" ]]; then
          export WANDB_DIR="${JOB_TMP_ROOT}/wandb_root"
          export WANDB_CACHE_DIR="${JOB_TMP_ROOT}/wandb_cache"
          export WANDB_CONFIG_DIR="${JOB_TMP_ROOT}/wandb_config"
          export WANDB_DATA_DIR="${JOB_TMP_ROOT}/wandb_data"
          mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}" "${WANDB_CONFIG_DIR}" "${WANDB_DATA_DIR}"
        fi

        echo "===== TRAIN ENV CHECK ====="
        which python; python -V
        python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
        echo "name=${NAME} bs=${BATCH_SIZE} ep=${EPOCHS} wk=${NUM_WORKERS}"
        echo "root=${MEMNAV_ROOT_DIR}"
        echo "feat=${MEMNAV_FEATURE_ROOT}"
        echo "window=${MEMNAV_WINDOW} num_scale=${MEMNAV_NUM_SCALE} max_frame_num=${MEMNAV_MAX_FRAME_NUM}"

        exec python scripts/train/train.py \
            --name "${NAME}" \
            --model-name memnav \
            --batch-size "${BATCH_SIZE}" \
            --num-workers "${NUM_WORKERS}" \
            --epochs "${EPOCHS}" \
            --root-dir "${MEMNAV_ROOT_DIR}"
    '

echo "=== done: $(date) ==="
