#!/bin/bash
#SBATCH --job-name=lingbot_precompute_pt2
#SBATCH --output=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/precomp2-%A_%a.out
#SBATCH --error=/scratch/lg154/Research/Nav/InternNav/logs/mp3d_gen/precomp2-%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --account=torch_pr_769_tandon_advanced
#SBATCH --partition=h100_tandon,h200_tandon
#SBATCH --gres=gpu:1
#SBATCH --export=ALL
#
# Precompute frozen LingBot-Map KV caches for the PACKED mp3d 54-89 scenes.
# Frames live in a read-only squashfs (pt2.sqf). We mount it the NYU-supported way:
#   apptainer exec --overlay pt2.sqf:ro <base.sif>
# Inside the container the tree appears at /mp3d_revisit_v0/vln_n1/traj_data (that is how the
# image was packed with -keep-as-directory). We READ frames from there and WRITE caches to a
# SEPARATE writable tree on host /scratch (--out_root), which is bind-mounted rw (the overlay
# is :ro, so writes land on the real filesystem, not the image).
# --cam_only --overwrite rewrites ONLY lingbot_cam_cache.npz (cam_k/v + cam_pose_enc), now
# including the whole-episode ground_h_est used by scale_mode='ground' (added 5e46781, absent
# from the Jul-12 caches). The expensive full cache (lingbot_cache.npz: dino_cls + anchor_k/v)
# is left untouched. --overwrite is required because the stale cam caches already exist and must
# be replaced; drop it for later gap-fills once every traj has a ground_h_est cam cache.
#
# NOTE for training later: run training inside the SAME overlay so rgb paths resolve, i.e.
#   root_dirs=/mp3d_revisit_v0/vln_n1/traj_data  (in-container overlay path)
#   feature_root=/scratch/lg154/Research/datasets/mp3d_revisit_v0_feat/vln_n1/traj_data (host)
#
# Submit (from InternNav/, or with the full path shown):
#   smoke (2 trajs, 1 shard):  sbatch --array=0 --export=ALL,NUM_SHARDS=1,LIMIT=2 scripts/train_memnav/precompute_mp3d_pt2.sbatch
#   full  (8-way shard):       sbatch --array=0-7%8 --export=ALL,NUM_SHARDS=8 scripts/train_memnav/precompute_mp3d_pt2.sbatch
# With --overwrite this pass rewrites every cam cache (the point: back-fill ground_h_est). Once
# complete, remove --overwrite in the python call below to restore skip-if-exists gap-filling.
set -euo pipefail

export NUM_SHARDS="${NUM_SHARDS:-1}"
export LIMIT="${LIMIT:-0}"
export SHARD="${SLURM_ARRAY_TASK_ID:-0}"

IMG=/scratch/lg154/Research/datasets/_overlays/mp3d_revisit_v0_pt2.sqf
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
            --cam_only --skip_scale --overwrite --use_sdpa \
            --kv_cache_sliding_window 32 --num_scale_frames 8 \
            --max_frame_num 2048 \
            --num_shards "${NUM_SHARDS}" --shard "${SHARD}" \
            --limit "${LIMIT}"
    '

echo "=== shard ${SHARD} done: $(date) ==="
