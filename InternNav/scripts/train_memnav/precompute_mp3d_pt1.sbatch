#!/bin/bash
#SBATCH --job-name=lingbot_precompute_pt1
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/precomp-%A_%a.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/precomp-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
# REQUIRED: complete scripts/train_memnav/README.md before every sbatch submission.
#
# Precompute frozen LingBot-Map KV caches for the PACKED mp3d 0-53 scenes.
# Frames live in a read-only squashfs (pt1.sqf). We mount it the NYU-supported way:
#   apptainer exec --overlay pt1.sqf:ro <base.sif>
# Inside the container the tree appears at /mp3d_revisit_v0/vln_n1/traj_data (that is how the
# image was packed with -keep-as-directory). We READ frames from there and WRITE caches to a
# SEPARATE writable tree on host /scratch (--out_root), which is bind-mounted rw (the overlay
# is :ro, so writes land on the real filesystem, not the image).
# --skip_scale drops the ~1.08 GB/traj scale KV (policy recomputes it on GPU from the first 8
# frames); the cache holds only dino_cls + anchor_k/v + cam_k/v.
#
# NOTE for training later: run training inside the SAME overlay so rgb paths resolve, i.e.
#   root_dirs=/mp3d_revisit_v0/vln_n1/traj_data  (in-container overlay path)
#   feature_root=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data (host)
#
# Submit (from InternNav/, or with the full path shown):
#   smoke (2 trajs, 1 shard):  sbatch --array=0 --export=ALL,NUM_SHARDS=1,LIMIT=2 scripts/train_memnav/precompute_mp3d_pt1.sbatch
#   full  (8-way shard):       sbatch --array=0-7%8 --export=ALL,NUM_SHARDS=8 scripts/train_memnav/precompute_mp3d_pt1.sbatch
# Idempotent (skips a traj whose out npz already exists), so re-submit fills gaps.
set -euo pipefail

export NUM_SHARDS="${NUM_SHARDS:-1}"
export LIMIT="${LIMIT:-0}"
export SHARD="${SLURM_ARRAY_TASK_ID:-0}"

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt1.sqf
BASE_SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
export ROOT_DIRS=/mp3d_revisit_v0/vln_n1/traj_data                      # in-container overlay path
export OUT_ROOT=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data
export REPO=/scratch/lg154/Research/Nav/NavDP/baselines/memnav/lingbot-map
export WEIGHTS=${REPO}/weights/lingbot-map-long.pt

echo "host: $(hostname); shard ${SHARD}/${NUM_SHARDS}; limit=${LIMIT}; start: $(date)"
mkdir -p "${OUT_ROOT}"

apptainer exec --nv \
    --overlay "${IMG}:ro" \
    -B /scratch/lg154 \
    "${BASE_SIF}" \
    bash -c '
        set -euo pipefail
        ls "${ROOT_DIRS}" >/dev/null || { echo "ABORT: overlay missing ${ROOT_DIRS}"; exit 1; }
        source /scratch/lg154/miniconda3/etc/profile.d/conda.sh
        conda activate /scratch/lg154/conda-envs/memnav
        cd /scratch/lg154/Research/Nav/InternNav
        python scripts/dataset_converters/precompute_lingbot_features.py \
            --root_dirs "${ROOT_DIRS}" \
            --out_root  "${OUT_ROOT}" \
            --lingbot_repo "${REPO}" \
            --weights "${WEIGHTS}" \
            --skip_scale --use_sdpa \
            --kv_cache_sliding_window 32 \
            --max_frame_num 2048 \
            --num_shards "${NUM_SHARDS}" --shard "${SHARD}" \
            --limit "${LIMIT}"
    '

echo "=== shard ${SHARD} done: $(date) ==="
