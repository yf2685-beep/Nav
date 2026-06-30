#!/bin/bash
# Smoke-test run for NavDP training on the 2-scene InternData-N1 (v0.1-mini) shard.
#
# Usage (from anywhere):
#   bash /home/asus/Research/Nav/InternNav/scripts/train/runs_navdp.sh
#
# One-time setup first:
#   conda activate navdp
#   # python deps the navdp env was missing:
#   pip install "transformers==4.51.0" "tyro>=0.9.26,<0.10" "tensorboard==2.20.0" \
#               "accelerate==1.7.0" "setuptools<81" ftfy regex pyarrow
#   # Long-CLIP source (imported transitively by train.py via cma/rdp policies):
#   git clone --depth 1 https://github.com/beichenzbc/Long-CLIP.git \
#       internnav/model/basemodel/Long-CLIP        # (symlinked as LongCLIP/)
#   # DepthAnythingV2 ViT-S backbone weights (hardcoded path checkpoints/depth_anything_v2_vits.pth):
#   hf download depth-anything/Depth-Anything-V2-Small depth_anything_v2_vits.pth --local-dir /tmp/da_v2_dl
#   mkdir -p checkpoints && cp /tmp/da_v2_dl/depth_anything_v2_vits.pth checkpoints/
#
# What this does:
#   - Single GPU (cuda:0), batch_size=2, num_workers=0 to keep tracebacks readable.
#   - Plain `python` (NOT torchrun) -> world_size=1, no DDP.
#   - Points root_dir at the dir that CONTAINS the `matterport3d_d435i` group dir.
#   - Moves the leftover *.tar.gz out of the scan path (the loader does os.listdir()
#     on every entry and crashes on a file).
#   - Runs 1 epoch. Override via env vars, e.g.:  BATCH_SIZE=4 EPOCHS=2 bash runs_navdp.sh

set -eo pipefail

# --- paths ---
INTERNNAV=/home/asus/Research/Nav/InternNav
ROOT_DIR=${ROOT_DIR:-/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data}
GROUP_DIR="$ROOT_DIR/matterport3d_d435i"
DATASET_CACHE=${DATASET_CACHE:-/tmp/navdp_smoke_dataset.json}

# --- run knobs ---
NAME=${NAME:-navdp_smoke}
BATCH_SIZE=${BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-1}
NUM_WORKERS=${NUM_WORKERS:-0}

# --- env ---
source /home/asus/miniconda3/etc/profile.d/conda.sh
conda activate navdp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_SHOW_CPP_STACKTRACES=1

cd "$INTERNNAV"
# Make the vendored diffusion_policy submodule importable as `diffusion_policy`.
# (Run `git submodule update --init` first if src/diffusion-policy is empty.)
export PYTHONPATH="$PWD/src/diffusion-policy:${PYTHONPATH:-}"

# --- guard: move tarballs out of the scan path ---
mkdir -p "$ROOT_DIR/../_tarballs"
shopt -s nullglob
for f in "$GROUP_DIR"/*.tar.gz; do
    mv "$f" "$ROOT_DIR/../_tarballs/" && echo "[guard] moved aside: $(basename "$f")"
done
shopt -u nullglob

echo "[info] scenes to train on:"
ls "$GROUP_DIR"

# Force a fresh dataset scan/index each run.
rm -f "$DATASET_CACHE"

python scripts/train/train.py \
    --name "$NAME" \
    --model-name navdp \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --root-dir "$ROOT_DIR" \
    --dataset-navdp "$DATASET_CACHE"
