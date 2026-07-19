#!/bin/bash
# Bare runner for the batched-window equivalence check — no SBATCH headers, meant
# to be launched via `srun --overlap --jobid=<interactive>` onto an already-held
# GPU allocation (e.g. an L40S interactive session on gl026). Mirrors the apptainer
# setup in run_batched_window_check.sbatch.
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
        export TMPDIR="/tmp/${USER}/memnav_chk_${SLURM_JOB_ID:-$$}"; mkdir -p "${TMPDIR}"
        export MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
        which python; python -V
        exec python scripts/diag_batched_window.py
    '
echo "=== done: $(date) ==="
