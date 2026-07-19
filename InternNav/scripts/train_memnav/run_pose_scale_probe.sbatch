#!/bin/bash
#SBATCH --job-name=pose_scale
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/posescale-%j.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/posescale-%j.err
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
# Pose-scale diagnostic (scripts/diag_pose_scale.py) inside the SAME apptainer
# overlay training uses. Decides whether the stuck aux-pose loss is a per-traj
# scale-ambiguity problem (=> scale-invariant target) or an arch/loss problem
# (=> keep metric, fix head). Frozen LingBot on GPU; ~40 trajectories + 256 samples.
#
# If GPU partitions are saturated, run inside an existing interactive alloc:
#   srun --jobid=<ALLOC> --overlap bash InternNav/scripts/train_memnav/run_pose_scale_probe.sbatch
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
export N_TRAJ=${N_TRAJ:-40}
export N_SAMPLES=${N_SAMPLES:-256}
export N_DECOMP=${N_DECOMP:-100}
# ONLY_C=1 -> skip B/A (already have those numbers), run only the drift-vs-OOD decomposition
export ONLY_C=${ONLY_C:-1}

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
        ONLY_C_FLAG=""
        if [ "${ONLY_C}" = "1" ]; then ONLY_C_FLAG="--only_C --out /scratch/lg154/Research/Nav/InternNav/logs/train_memnav/pose_decomp"; fi
        exec python scripts/diag_pose_scale.py \
            --n_traj "${N_TRAJ}" \
            --n_samples "${N_SAMPLES}" \
            --n_decomp "${N_DECOMP}" \
            ${ONLY_C_FLAG}
    '
echo "=== done: $(date) ==="
