#!/bin/bash
#SBATCH --job-name=trajviz
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/trajviz-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/trajviz-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:40:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Visualize LingBot predicted camera trajectory (native full-seq + our streaming)
# vs GT for ONE episode (scripts/diag_traj_viz.py). Tests the "weak at sharp turns"
# hypothesis: does the predicted track diverge from GT at the leg-switch align_turn?
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
export TRAJ_IDX=${TRAJ_IDX:--1}
export MAX_FRAMES=${MAX_FRAMES:-512}
export START_FRAME=${START_FRAME:-0}
export NATIVE_WINDOW=${NATIVE_WINDOW:-64}
export STREAM_STRIDE=${STREAM_STRIDE:-4}
export IMG_MODE=${IMG_MODE:-pad}
export OUT_TAG=${OUT_TAG:-native_stream}

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
        export TOKENIZERS_PARALLELISM=false
        exec python scripts/diag_traj_viz.py \
            --traj_idx "${TRAJ_IDX}" \
            --max_frames "${MAX_FRAMES}" \
            --start_frame "${START_FRAME}" \
            --native_window "${NATIVE_WINDOW}" \
            --stream_stride "${STREAM_STRIDE}" \
            --img_mode "${IMG_MODE}" \
            --out /scratch/lg154/Research/Nav/InternNav/logs/train_memnav/traj_viz_'"${OUT_TAG}"'
    '
echo "=== done: $(date) ==="
