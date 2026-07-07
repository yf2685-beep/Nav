#!/usr/bin/env bash
# Revisit-definition sweep: end-to-end runner.
# Measures over what (relative pose, covisibility) envelope LingBot correctly
# relocalizes a goal image B inserted after its match frame X on leg A —
# the envelope defines "revisit" for the memnav gate.
#
# Stages run in DIFFERENT conda envs; run them one after another:
#   1  (habitat)     revisit_sweep_gen.py      leg-A + grid goals + GT covisibility
#   1.5(lingbot-map) precompute_lingbot_features.py   KV caches (~1.1 GB/traj!)
#   2  (memnav)      revisit_sweep_eval.py     insertion -> relocalization errors
#   3  (memnav)      revisit_sweep_analyze.py  envelope heatmaps + summary.md
set -e

MP3D=/home/asus/Research/datasets/mp3d
HERE=$(cd "$(dirname "$0")" && pwd)

# QUICK=1 -> ~20-minute end-to-end run: 1 short trajectory per scene, 3 anchors,
# thinned grid (fwd {0,1,2,4} x lat 0 x dyaw {0,±45,±90,180} = 24 goals/anchor,
# ~144 insertions total). Own output dirs so it never mixes with the full run.
if [ "${QUICK:-0}" = "1" ]; then
  OUT=${OUT:-/home/asus/Research/datasets/memnav_sweep_quick}
  VIZ=${VIZ:-/home/asus/Research/Nav/memnav_viz/revisit_sweep_quick}
  N=${N:-1}
  GEN_ARGS="--dA_min 4 --dA_max 8 --anchors 3 --fwd 0 1 2 4 --lat 0 \
            --dyaw 0 45 -45 90 -90 180 --n_neg 6 --covis_stride 4"
else
  OUT=${OUT:-/home/asus/Research/datasets/memnav_sweep}
  VIZ=${VIZ:-/home/asus/Research/Nav/memnav_viz/revisit_sweep}
  N=${N:-4}          # trajectories per scene
  GEN_ARGS=""
fi

stage1() {
  for SC in 17DRP5sb8fy 1LXtFkjw3qL; do
    conda run --no-capture-output -n habitat python "$HERE/revisit_sweep_gen.py" \
      --scene "$MP3D/$SC.glb" --navmesh "$MP3D/$SC.navmesh" \
      --out "$OUT" --n "$N" --seed 0 $GEN_ARGS
  done
}

stage15() {
  # kv_cache_sliding_window=64 = LingBot's intended setting (sharp degradation <32);
  # must match revisit_sweep_eval.py --window. NOTE: memnav training caches use 8.
  conda run --no-capture-output -n lingbot-map python \
    /home/asus/Research/Nav/InternNav/scripts/dataset_converters/precompute_lingbot_features.py \
    --root_dirs "$OUT" \
    --lingbot_repo /home/asus/Research/lingbot-map \
    --weights /home/asus/Research/lingbot-map/weights/lingbot-map-long.pt \
    --kv_cache_sliding_window 64 \
    --use_sdpa ${PRECOMPUTE_ARGS:-}
}

stage2() {
  # expandable_segments: the growing 64-frame KV cache fragments the allocator
  # (~9 GB reserved-unallocated otherwise) — required to fit next to other GPU jobs.
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  conda run --no-capture-output -n memnav python "$HERE/revisit_sweep_eval.py" \
    --sweep_root "$OUT" --out "$VIZ/results.parquet"
}

stage3() {
  conda run --no-capture-output -n memnav python "$HERE/revisit_sweep_analyze.py" \
    --results "$VIZ/results.parquet" --out_dir "$VIZ"
}

case "${1:-all}" in
  1) stage1 ;;
  1.5) stage15 ;;
  2) stage2 ;;
  3) stage3 ;;
  all) stage1; stage15; stage2; stage3 ;;
  *) echo "usage: $0 [1|1.5|2|3|all]"; exit 1 ;;
esac
