"""MemNav Flask server — the model half of the two-process closed-loop eval.

Runs in the `memnav` conda env. The Habitat client (habitat env) streams RGB
frames over HTTP; this server maintains the live LingBot memory and plans
trajectories toward a goal image on request.

Endpoints (NavDP wire-contract style):
  POST /navigator_reset      JSON {camera_height?, stop_threshold?, batch_size?}
                             -> {"algo": "memnav"}   (starts a fresh episode)
  POST /memory_step          files: image (jpg)      -> {"frame_idx": i}
                             stream a frame into memory WITHOUT planning (leg replay)
  POST /imagegoal_step       files: image (jpg), goal (jpg)
                             -> {"trajectory": [24,3] metres (x fwd, y left, theta),
                                 "all_trajectory": [N,24,3], "all_values": [N],
                                 "gate": float, "match_idx": int, "frame_idx": int}
                             streams the frame, then plans toward the goal.

Usage:
  conda activate memnav
  python memnav_server.py --port 18888 \
    --checkpoint /home/asus/Research/Nav/InternNav/checkpoints/memnav_2leg_axisfix/checkpoint-1500/memnav.ckpt
"""

import argparse
import os

# must precede any torch import (policy_agent): reduces fragmentation OOMs from
# the large KV-cache alloc/free cycle each plan() runs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from flask import Flask, jsonify, request

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=18888)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--internnav_root", type=str,
                    default="/home/asus/Research/Nav/InternNav")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--num_samples", type=int, default=16)
parser.add_argument("--exclude_recent", type=int, default=83,
                    help="retrieval candidate gap (dataset default)")
parser.add_argument("--retrieval", choices=["head", "raw"], default="raw",
                    help="match selector: trained projection vs raw dino-cls cosine")
parser.add_argument("--gate_skip_below", type=float, default=0.0,
                    help="skip the goal-insert tower when trained gate < this (0 = never skip)")
parser.add_argument("--anchor_switch_margin", type=float, default=0.01,
                    help="sticky-anchor ratchet: switch match only on a clear score win")
parser.add_argument("--buffer_root", type=str, default="/tmp/memnav_server_buffer")
args = parser.parse_args()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
from policy_agent import MemNavAgent  # noqa: E402  (after chdir so lingbot paths resolve)

agent = MemNavAgent(
    checkpoint=args.checkpoint,
    internnav_root=args.internnav_root,
    device=args.device,
    exclude_recent=args.exclude_recent,
    num_samples=args.num_samples,
    buffer_root=args.buffer_root,
    gate_skip_below=args.gate_skip_below,
    retrieval_mode=args.retrieval,
    anchor_switch_margin=args.anchor_switch_margin,
)

app = Flask(__name__)


@app.route("/navigator_reset", methods=["POST"])
def navigator_reset():
    payload = request.get_json(silent=True) or {}
    cam_h = float(payload.get("camera_height", 0.5))
    agent.reset(camera_height=cam_h)
    return jsonify({"algo": "memnav"})


@app.route("/navigator_reset_env", methods=["POST"])
def navigator_reset_env():
    # single-env server: same as a full reset (used by the cold/reset-memory arm)
    agent.reset(camera_height=agent.camera_height)
    return jsonify({"algo": "memnav"})


@app.route("/memory_step", methods=["POST"])
def memory_step():
    idx = agent.add_frame(request.files["image"].read())
    return jsonify({"frame_idx": idx})


@app.route("/imagegoal_step", methods=["POST"])
def imagegoal_step():
    agent.add_frame(request.files["image"].read())
    out = agent.plan(request.files["goal"].read())
    return jsonify(out)


if __name__ == "__main__":
    print(f"[memnav_server] ready on :{args.port} "
          f"(W={agent.W}, S={agent.S}, amargin={agent.amargin}, "
          f"exclude_recent={agent.exclude_recent}, samples={agent.num_samples})")
    app.run(host="0.0.0.0", port=args.port, threaded=False)
