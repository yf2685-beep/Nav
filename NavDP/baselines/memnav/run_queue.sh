#!/bin/bash
# Queued, self-healing MemNav IsaacSim closed-loop runner (machine 127).
# Waits until the GPU has enough free memory, then runs the memnav server + imagegoal eval,
# auto-restarting the whole pipeline if either process dies (e.g. gets killed). Detached from
# any terminal — launch it and walk away. Writes a DONE marker when an eval run completes.
#
#   nohup setsid bash run_queue.sh > run_queue.log 2>&1 & disown
#
# Watch:  tail -f run_queue.log       Stop:  touch STOP_QUEUE   (or kill the process group)
set -u
cd "$(dirname "$0")"
HERE="$(pwd)"
source /home/nyuair/miniconda3/etc/profile.d/conda.sh

# ---- config ----
NEED_MIB=${NEED_MIB:-24000}          # free GPU memory required before we launch (server ~9G + IsaacSim ~15G)
CONSEC=${CONSEC:-3}                   # consecutive free readings required (avoid transient dips)
POLL=${POLL:-30}                     # seconds between GPU polls
SETTLE=${SETTLE:-25}                 # ROOT-CAUSE FIX: seconds to let the server's reset/GPU-init settle before
                                     # launching IsaacSim. Firing IsaacSim's CUDA/Vulkan init at the same instant
                                     # the server is building its navigator on the same GPU races the graphics
                                     # stack and silently SIGKILLs the server (no traceback). ~18s gap is enough.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-30}     # give up after this many pipeline attempts
ATTEMPT_TIMEOUT=${ATTEMPT_TIMEOUT:-2400}   # max seconds for one eval attempt (40 min)
PORT=${PORT:-8899}                   # default off 8888 to dodge a stale server squatting on the common port
SCENE_DIR=${SCENE_DIR:-/home/nyuair/junyi/NavDP-old/asset_scenes/cluttered_hard}
SCENE_INDEX=${SCENE_INDEX:-0}
CKPT="$HERE/checkpoints/memnav_pilot.ckpt"
LB_REPO=/home/nyuair/yuxuan/lingbot-map
LB_W=/home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt
NAVDP=/home/nyuair/yuxuan/1\ robot\ navigation/NavDP
SLOG="$HERE/queue_server.log"
ELOG="$HERE/queue_eval.log"

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

free_mib(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '; }

wait_for_gpu(){
  local ok=0
  while :; do
    [ -f "$HERE/STOP_QUEUE" ] && { log "STOP_QUEUE found — exiting."; exit 0; }
    local f; f=$(free_mib)
    if [ "${f:-0}" -ge "$NEED_MIB" ]; then ok=$((ok+1)); log "GPU free ${f} MiB (need $NEED_MIB) [$ok/$CONSEC]";
    else ok=0; log "GPU free ${f} MiB (need $NEED_MIB) — waiting"; fi
    [ "$ok" -ge "$CONSEC" ] && return 0
    sleep "$POLL"
  done
}

cleanup(){ [ -n "${SRV_PID:-}" ] && kill "$SRV_PID" 2>/dev/null; [ -n "${EV_PID:-}" ] && kill "$EV_PID" 2>/dev/null;
           pkill -f "eval_imagegoal_wheeled.py" 2>/dev/null; sleep 3; }
trap cleanup EXIT

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  [ -f "$HERE/STOP_QUEUE" ] && { log "STOP_QUEUE — done."; exit 0; }
  log "===== attempt $attempt/$MAX_ATTEMPTS ====="
  wait_for_gpu

  # a stale server squatting on $PORT makes bind() fail -> the fresh server exits at startup, which
  # looks like a mysterious "server died" and retries forever. Fail fast with a clear message instead.
  if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    log "PORT $PORT already in use (stale server?). Kill it or run with PORT=<free port>. Exiting."
    exit 1
  fi

  # --- server (enerverse) ---
  log "launching memnav server..."
  ( conda activate enerverse
    CUDA_VISIBLE_DEVICES=0 exec python "$HERE/memnav_server.py" --port "$PORT" \
      --checkpoint "$CKPT" --lingbot_repo "$LB_REPO" --lingbot_weights "$LB_W" \
      --sample_num 4 --device cuda:0 ) > "$SLOG" 2>&1 &
  SRV_PID=$!
  for i in $(seq 1 60); do grep -q "Running on http://127" "$SLOG" 2>/dev/null && break; kill -0 "$SRV_PID" 2>/dev/null || break; sleep 2; done
  if ! kill -0 "$SRV_PID" 2>/dev/null; then log "server died during startup — retrying"; sleep 20; continue; fi
  log "server up (pid $SRV_PID). pre-warming..."
  curl -s -m 120 -X POST "http://127.0.0.1:$PORT/navigator_reset" -H "Content-Type: application/json" \
    -d '{"intrinsic":[[300,0,320],[0,300,240],[0,0,1]],"stop_threshold":-0.5,"batch_size":1}' >/dev/null 2>&1
  if ! kill -0 "$SRV_PID" 2>/dev/null; then log "server died during pre-warm — retrying"; sleep 20; continue; fi
  log "server pre-warmed OK."

  # ROOT-CAUSE FIX: let the server's GPU allocation settle before IsaacSim inits its own CUDA/Vulkan
  # context on the same card — launching them back-to-back races the graphics stack and kills the server.
  log "letting server settle ${SETTLE}s before IsaacSim (avoids GPU-init race)..."
  sleep "$SETTLE"
  if ! kill -0 "$SRV_PID" 2>/dev/null; then log "server died during settle — retrying"; sleep 20; continue; fi

  # --- eval (isaaclabjunyi) ---
  log "launching IsaacSim eval (scene_index $SCENE_INDEX)..."
  ( conda activate isaaclabjunyi
    cd "$NAVDP"
    CUDA_VISIBLE_DEVICES=0 exec python eval_imagegoal_wheeled.py --port "$PORT" \
      --scene_dir "$SCENE_DIR" --scene_index "$SCENE_INDEX" --scene_scale 1.0 ) > "$ELOG" 2>&1 &
  EV_PID=$!

  # --- supervise ---
  start=$(date +%s); progressed=0
  while :; do
    [ -f "$HERE/STOP_QUEUE" ] && { log "STOP_QUEUE — stopping."; cleanup; exit 0; }
    if ! kill -0 "$EV_PID" 2>/dev/null; then log "eval exited."; break; fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then log "SERVER DIED mid-run — restarting pipeline."; break; fi
    grep -qE "Planning time|success_flag|episode" "$ELOG" 2>/dev/null && progressed=1
    now=$(date +%s); [ $((now-start)) -ge "$ATTEMPT_TIMEOUT" ] && { log "attempt timeout."; break; }
    sleep 10
  done

  # grep -c already prints "0" on no-match (and exits 1); the old `|| echo 0` appended a
  # second line -> "0\n0" -> "[: integer expression expected". Use `|| true` and keep one number.
  n_plan=$(grep -c "Planning time" "$ELOG" 2>/dev/null || true); n_plan=${n_plan:-0}
  # success = the robot actually navigated for a while (enough planning steps flowed)
  if [ "${n_plan:-0}" -ge 10 ]; then
    log "RUN SUCCEEDED — $n_plan closed-loop planning steps. Marking DONE."
    echo "done attempt=$attempt planning_steps=$n_plan $(date)" > "$HERE/QUEUE_DONE"
    log "vis images: $NAVDP/imagegoal_memnav_$(basename "$SCENE_DIR")/"
    cleanup
    exit 0
  fi
  cleanup
  log "attempt $attempt did not complete (only $n_plan planning steps). backing off 30s."
  sleep 30
done
log "gave up after $MAX_ATTEMPTS attempts."
