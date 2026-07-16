#!/bin/bash
#SBATCH --job-name=gate_probe
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/gateprobe-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/gateprobe-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=a100_tandon,h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Offline CLS-separation probe (scripts/diag_gate_features.py) inside the SAME
# apptainer overlay training uses — frames + goal_{j}.jpg live in the squashfs,
# caches on host. GPU only to run the frozen DINO on the goal images (few hundred).
#
# If GPU partitions are saturated, run it inside an existing interactive
# allocation instead (it auto-falls back to CPU if no GPU is visible):
#   srun --jobid=<ALLOC> --overlap bash InternNav/scripts/train_memnav/run_cls_probe.sbatch
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
export MAX_PER_GROUP=${MAX_PER_GROUP:-300}

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
        exec python scripts/diag_gate_features.py \
            --max_per_group "${MAX_PER_GROUP}"
    '
echo "=== done: $(date) ==="
