#!/bin/bash
#SBATCH --job-name=memnav_batchchk
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/batchchk-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/batchchk-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=0:40:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Numerical-equivalence check for the batched window_forward path
# (MEMNAV_STREAM_GROUP). Runs scripts/diag_batched_window.py inside the SAME
# apptainer overlay as training so it sees the packed frames + host caches.
set -euo pipefail

mkdir -p /scratch/lg154/Research/Nav/InternNav/logs/train_memnav

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
export CHECK_GROUP=${CHECK_GROUP:-4}

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
        JOB_TMP_ROOT="/tmp/${USER}/memnav_chk_${SLURM_JOB_ID:-$$}"
        export TMPDIR="${JOB_TMP_ROOT}/tmp"; mkdir -p "${TMPDIR}"
        export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
        which python; python -V
        exec python scripts/diag_batched_window.py
    '

echo "=== done: $(date) ==="
