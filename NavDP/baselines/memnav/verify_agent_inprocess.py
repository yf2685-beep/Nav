"""Agent-level smoke: feed raw-numpy RGB frames (simulating IsaacSim) to MemNav_Agent,
cross the warmup boundary, check warmup->real trajectory transition + batch assembly."""
import os, sys, time
sys.path.insert(0, "/home/nyuair/yuxuan/lingbot-map")
sys.path.insert(0, "/home/nyuair/yuxuan/1 robot navigation/Nav/NavDP/baselines/memnav")
import numpy as np
from PIL import Image
from policy_agent import MemNav_Agent

RGB = ("/home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp/"
       "matterport3d_d435i/17DRP5sb8fy/trajectory_10/videos/chunk-000/observation.images.rgb")
CKPT = "/tmp/claude-1000/-home-nyuair-yuxuan/728eec84-0d5f-445c-b895-a46be90f5482/scratchpad/memnav_ckpt/memnav.ckpt"

def load_rgb(i):
    return np.asarray(Image.open(os.path.join(RGB, f"{i}.jpg")).convert("RGB"))  # [H,W,3] uint8

agent = MemNav_Agent(
    intrinsic=np.eye(3), checkpoint=CKPT,
    lingbot_repo="/home/nyuair/yuxuan/lingbot-map",
    lingbot_weights="/home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt",
    predict_size=24, sample_num=4, device="cuda:0")
agent.reset(batch_size=1)

goal = load_rgb(5)[None]     # [1,H,W,3]  (a 'seen' location)
print(f"warmup boundary lo={agent.engine.lo}; feeding 18 frames")
for step in range(18):
    rgb = load_rgb(step)[None]           # [1,H,W,3]
    t0 = time.time()
    exe, allt, allv = agent.step_imagegoal(goal, rgb)
    kind = "warmup" if step < agent.engine.lo else "REAL"
    print(f"  step {step:2d} [{kind:6s}] {time.time()-t0:4.1f}s exec={exe.shape} all={allt.shape} "
          f"x_end={exe[0,-1,0]:+.2f} finite={np.isfinite(exe).all()}")
print("\nRESULT: AGENT-SMOKE PASS")
