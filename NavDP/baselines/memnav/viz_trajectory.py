"""Visualize what the trained MemNav policy 'wants to do': the current camera view, the
goal image, and the predicted trajectory (bird's-eye, robot-local). Saves traj_viz.png."""
import os, sys
sys.path.insert(0, "/home/nyuair/yuxuan/lingbot-map")
sys.path.insert(0, "/home/nyuair/yuxuan/1 robot navigation/Nav/NavDP/baselines/memnav")
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from memnav_infer import MemNavInference

RGB = ("/home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp/"
       "matterport3d_d435i/17DRP5sb8fy/trajectory_10/videos/chunk-000/observation.images.rgb")
CKPT = "/home/nyuair/yuxuan/1 robot navigation/Nav/NavDP/baselines/memnav/checkpoints/memnav_pilot.ckpt"
K = 22            # current step
GOAL = 6          # a 'seen' goal (visited earlier)
OUT = "/tmp/claude-1000/-home-nyuair-yuxuan/728eec84-0d5f-445c-b895-a46be90f5482/scratchpad/traj_viz.png"

eng = MemNavInference(checkpoint=CKPT, lingbot_repo="/home/nyuair/yuxuan/lingbot-map",
                      lingbot_weights="/home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt",
                      device="cuda:0")

buffer = [eng.net.lingbot.load_images([os.path.join(RGB, f"{i}.jpg")])[0] for i in range(K + 1)]
goal_pp = eng.net.lingbot.load_images([os.path.join(RGB, f"{GOAL}.jpg")])[0]
traj, info = eng.predict(buffer, goal_pp, sample_num=6)     # [6,24,3]
print(f"predicted traj {traj.shape}  gate={info['gate']:.2f}  match_idx={info['match_idx']}")

cur_img = np.asarray(Image.open(os.path.join(RGB, f"{K}.jpg")).convert("RGB"))
goal_img = np.asarray(Image.open(os.path.join(RGB, f"{GOAL}.jpg")).convert("RGB"))

fig = plt.figure(figsize=(14, 5))
plt.subplots_adjust(wspace=0.25, left=0.04, right=0.97, top=0.86, bottom=0.1)
ax1 = fig.add_subplot(1, 3, 1); ax1.imshow(cur_img); ax1.set_title(f"current view (step {K})\nwhat the robot sees", fontsize=11); ax1.axis("off")
ax2 = fig.add_subplot(1, 3, 2); ax2.imshow(goal_img); ax2.set_title(f"goal image (frame {GOAL})\nwhere it must go", fontsize=11); ax2.axis("off")
ax3 = fig.add_subplot(1, 3, 3)
# bird's-eye: x forward (up), y left (left) — robot at origin looking +x
for s in range(traj.shape[0]):
    ax3.plot(-traj[s, :, 1], traj[s, :, 0], color="#39d6c6", alpha=0.3, lw=1.2)
mean = traj.mean(0)
ax3.plot(-mean[:, 1], mean[:, 0], color="#f0b429", lw=3, label="executed path")
ax3.scatter([0], [0], c="k", marker="^", s=120, zorder=5, label="robot")
ax3.scatter([-mean[-1, 1]], [mean[-1, 0]], c="#f0b429", s=80, zorder=5)
ax3.set_title(f"predicted trajectory (24 waypoints)\ngate={info['gate']:.2f} · 6 diffusion samples", fontsize=11)
ax3.set_xlabel("← left    (metres)    right →"); ax3.set_ylabel("forward (metres) →")
ax3.axhline(0, color="#ccc", lw=.6); ax3.axvline(0, color="#ccc", lw=.6)
ax3.set_aspect("equal"); ax3.grid(alpha=.25); ax3.legend(loc="upper left", fontsize=9)
fig.suptitle("MemNav — trained policy: sees current frame + goal → plans a local trajectory", fontsize=13, y=0.98)
fig.savefig(OUT, dpi=120, facecolor="white")
print(f"saved {OUT}")
