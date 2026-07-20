#!/bin/bash
#SBATCH --job-name=logo_streaming
#SBATCH --partition=spark
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=infinite
#SBATCH --output=/media/cvpr/yuxuan/logoplanner/logo_streaming_%j.log
#
# Submit streaming GCT LoGoPlanner training on the spark partition:
#   sbatch InternNav/scripts/train/submit_streaming.sbatch
# Tail the log:
#   tail -f /media/cvpr/yuxuan/logoplanner/logo_streaming_<jobid>.log

set -eo pipefail

# spark = ARM (aarch64) GB10 nodes. The conda envs on the dgx-login node
# (/home/cvpr/miniconda3) are NOT visible on compute nodes. The enerverse env is
# shipped as a conda-pack tarball on the shared CIFS mount and must be unpacked
# to NODE-LOCAL /home/cvpr (CIFS can't host it — pip/chmod fails). Unpack once
# per node (cached on subsequent jobs), then put it on PATH.
ENV_DIR=/home/cvpr/enerverse_arm
ENV_TARBALL=/media/cvpr/yuxuan/enerverse_arm.tar.gz
if [ ! -x "$ENV_DIR/bin/python" ]; then
    echo "[env] unpacking $ENV_TARBALL -> /home/cvpr (first run on $(hostname))..."
    tar xzf "$ENV_TARBALL" -C /home/cvpr
fi
export PATH="$ENV_DIR/bin:$PATH"
echo "[env] python=$(which python)  arch=$(uname -m)  host=$(hostname)"

cd /media/cvpr/yuxuan/logoplanner/Nav/InternNav

# enerverse_arm is an inference pack and each compute node has its OWN local copy
# in a different state, so a fixed dep list is unreliable. Run the iterative
# resolver: it imports the train.py entrypoint and pip-installs whatever
# ModuleNotFoundError names, looping until imports resolve (no-op once complete).
# (lerobot is NOT needed — the *_lerobot dataset just reads parquet via pandas.)
if ! python scripts/train/train.py --help >/dev/null 2>&1; then
    echo "[env] resolving InternNav training import chain...";
    bash /media/cvpr/yuxuan/logoplanner/setup_stream_env.sh 2>&1 | grep -E "missing|IMPORTS_OK|Successfully" || true;
fi
# HF Trainer's report_to='tensorboard' needs the tensorboard package at Trainer
# init (a runtime callback requirement the import resolver above can't catch).
python -c "import tensorboard" 2>/dev/null || pip install -q tensorboard

# All streaming env vars + paths live in run_streaming.sh; override here if needed.
# e.g. BATCH_SIZE=2 EPOCHS=5 below.
BATCH_SIZE=${BATCH_SIZE:-1} EPOCHS=${EPOCHS:-1} bash scripts/train/run_streaming.sh
