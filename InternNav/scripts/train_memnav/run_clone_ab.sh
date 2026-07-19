#!/bin/bash
# Bare runner for the encode_memory profiler — launch via
#   srun --overlap --jobid=<interactive> bash scripts/train_memnav/run_profile_encode.sh
# onto an already-held GPU allocation. Mirrors run_batched_window_check.sh.
set -euo pipefail

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
REPO_ROOT=/scratch/lg154/Research/Nav/InternNav

export MEMNAV_ROOT_DIR=/mp3d_revisit_v0/vln_n1/traj_data
export MEMNAV_FEATURE_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data
export LINGBOT_REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export LINGBOT_WEIGHTS=${LINGBOT_REPO}/weights/lingbot-map-long.pt
export MEMNAV_WINDOW=32
export MEMNAV_NUM_SCALE=8
export MEMNAV_MAX_FRAME_NUM=2048
export MEMNAV_REPORT_TO=none
export PROFILE_GROUP=${PROFILE_GROUP:-4}
export BATCH_SIZE=${BATCH_SIZE:-16}
export PROFILE_REPS=${PROFILE_REPS:-3}
export PROFILE_SEED=${PROFILE_SEED:-0}

echo "host: $(hostname); start: $(date)"
nvidia-smi -L || true

apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
        conda activate /scratch/lg154/conda-envs/memnav
        cd '"${REPO_ROOT}"'
        export PYTHONPATH="'"${REPO_ROOT}"'/src/diffusion-policy:${PYTHONPATH:-}"
        export TMPDIR="/tmp/${USER}/memnav_prof_${SLURM_JOB_ID:-$$}"; mkdir -p "${TMPDIR}"
        export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
        export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
        which python; python -V
        exec python scripts/diag_clone_ab.py
    '
echo "=== done: $(date) ==="
