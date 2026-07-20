"""MemNav Flask server — same HTTP contract as the other NavDP baselines.

Endpoints:
  POST /navigator_reset      json{intrinsic, stop_threshold, batch_size} -> build/reset agent
  POST /navigator_reset_env  json{env_id}                                -> reset one env buffer
  POST /imagegoal_step       files{image, goal, depth}                   -> {trajectory,...}

Run in the `enerverse` env:
  python memnav_server.py --port 8888 \
      --checkpoint <memnav.ckpt> \
      --lingbot_repo /home/nyuair/yuxuan/lingbot-map \
      --lingbot_weights /home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt
"""
import argparse
import sys

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, "/home/nyuair/yuxuan/lingbot-map")
from policy_agent import MemNav_Agent

app = Flask(__name__)
memnav = None
args = None


@app.route("/navigator_reset", methods=["POST"])
def memnav_reset():
    # NOTE: the agent (incl. the frozen LingBot) is built ONCE in the main thread at
    # startup — NOT here. Building/running CUDA + flash-attn/xformers inside a Flask
    # worker thread crashes the process silently (no traceback). Here we only reset state.
    intrinsic = np.array(request.get_json().get("intrinsic"))
    threshold = np.array(request.get_json().get("stop_threshold"))
    batchsize = int(np.array(request.get_json().get("batch_size")))
    memnav.intrinsic = intrinsic
    memnav.reset(batchsize, threshold)
    return jsonify({"algo": "memnav"})


@app.route("/navigator_reset_env", methods=["POST"])
def memnav_reset_env():
    memnav.reset_env(int(request.get_json().get("env_id")))
    return jsonify({"algo": "memnav"})


@app.route("/imagegoal_step", methods=["POST"])
def memnav_imagegoal_step():
    B = memnav.batch_size
    # current RGB (concat over envs -> [B,H,W,3])
    image = np.asarray(Image.open(request.files["image"].stream).convert("RGB"))
    image = image.reshape((B, -1, image.shape[1], 3))
    # goal RGB
    goal = np.asarray(Image.open(request.files["goal"].stream).convert("RGB"))
    goal = goal.reshape((B, -1, goal.shape[1], 3))

    exec_traj, all_traj, all_values = memnav.step_imagegoal(goal, image)
    return jsonify({
        "trajectory": exec_traj.tolist(),
        "all_trajectory": all_traj.tolist(),
        "all_values": all_values.tolist(),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8888)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--lingbot_repo", default="/home/nyuair/yuxuan/lingbot-map")
    ap.add_argument("--lingbot_weights",
                    default="/home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt")
    ap.add_argument("--sample_num", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    # Build the agent (frozen LingBot + trained heads) ONCE, in the MAIN thread, before
    # serving. Building/running it inside a Flask worker thread crashed the process silently.
    memnav = MemNav_Agent(
        np.eye(3), checkpoint=args.checkpoint,
        lingbot_repo=args.lingbot_repo, lingbot_weights=args.lingbot_weights,
        predict_size=24, sample_num=args.sample_num, device=args.device)
    memnav.reset(1)
    print("[memnav_server] agent ready — serving.", flush=True)
    # Match the working navdp/logoplanner servers: default threading, localhost.
    app.run(host="127.0.0.1", port=args.port)
