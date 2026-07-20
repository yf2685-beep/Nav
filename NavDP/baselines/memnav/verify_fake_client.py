"""Fake IsaacSim client — proves the MemNav server generates trajectories.
Mirrors utils_tasks/client_utils.imagegoal_step exactly: encode current RGB + goal as jpg,
POST /imagegoal_step, parse the returned trajectory. Feeds real frames, prints each result."""
import io, json, os, time
import cv2, numpy as np, requests
from PIL import Image

RGB = ("/home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp/"
       "matterport3d_d435i/17DRP5sb8fy/trajectory_10/videos/chunk-000/observation.images.rgb")
PORT = 8888
def load(i): return np.asarray(Image.open(os.path.join(RGB, f"{i}.jpg")).convert("RGB"))  # HxWx3 RGB

# reset
r = requests.post(f"http://localhost:{PORT}/navigator_reset",
                  json={"intrinsic": [[300,0,320],[0,300,240],[0,0,1]], "stop_threshold": -0.5, "batch_size": 1},
                  timeout=120)
print("navigator_reset ->", r.json())

goal = load(5)                                   # a 'seen' location (frame 5)
_, gjpg = cv2.imencode('.jpg', goal)

for k in range(20):
    img = load(k)
    _, ijpg = cv2.imencode('.jpg', img)
    depth = np.zeros((img.shape[0], img.shape[1]), np.uint16)
    _, djpg = cv2.imencode('.png', depth)
    files = {'image': ('image.jpg', ijpg.tobytes(), 'image/jpeg'),
             'goal':  ('goal.jpg',  gjpg.tobytes(), 'image/jpeg'),
             'depth': ('depth.png', djpg.tobytes(), 'image/png')}
    t0 = time.time()
    try:
        resp = requests.post(f"http://localhost:{PORT}/imagegoal_step", files=files,
                             data={'rgb_time': time.time(), 'depth_time': time.time()}, timeout=300)
        out = resp.json()
        traj = np.array(out['trajectory'])            # [B, 24, 3] executed
        allt = np.array(out['all_trajectory'])        # [B, 4, 24, 3]
        b0 = traj[0]
        kind = "warmup" if k < 15 else "REAL "
        print(f"step {k:2d} [{kind}] {time.time()-t0:5.1f}s | HTTP {resp.status_code} | "
              f"traj{traj.shape} all{allt.shape} | endpoint(x,y,θ)=({b0[-1,0]:+.2f},{b0[-1,1]:+.2f},{b0[-1,2]:+.2f}) "
              f"| finite={np.isfinite(traj).all()}")
    except Exception as e:
        print(f"step {k:2d} FAILED: {type(e).__name__}: {str(e)[:80]}")
        break

print("\nRESULT: if you see traj(1, 24, 3) with finite endpoints -> the server IS generating trajectories.")
