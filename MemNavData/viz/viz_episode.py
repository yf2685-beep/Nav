import json, os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ep = "/home/asus/Research/Nav/memnav_viz/twoleg_proto/episode_0000"
m = json.load(open(os.path.join(ep, "meta/gen_meta.json")))
df = pd.read_parquet(os.path.join(ep, "data/chunk-000/episode_000000.parquet"))
P = np.stack([np.array(a.tolist(), float).reshape(4, 4)[:3, 3] for a in df["action"]])  # stored Zup
sw = m["switch_idx"]
legA, legB = P[:sw], P[sw:]

fig = plt.figure(figsize=(15, 4))
# top-down (stored frame x-y ground plane)
ax = fig.add_subplot(1, 4, 1)
ax.plot(legA[:, 0], legA[:, 1], "b.-", ms=3, label="leg A start->A")
ax.plot(legB[:, 0], legB[:, 1], "r.-", ms=3, label="leg B A->B (genuine)")
ax.plot(legA[::-1][:, 0], legA[::-1][:, 1], "g:", lw=1, label="reversed leg A")
for name, xy, c in [("start", legA[0], "b"), ("A", legA[-1], "k"), ("B", legB[-1], "r")]:
    ax.scatter(*xy[:2], c=c, s=60, zorder=5); ax.annotate(name, xy[:2])
ax.set_title(f"ep0 div={m['legB_vs_revlegA_mean_div_m']:.2f}m"); ax.legend(fontsize=6); ax.axis("equal")

# a leg-A frame where B is seen, with B marker
si = m["b_seen_idx"][len(m["b_seen_idx"]) // 2]
img = np.array(Image.open(os.path.join(ep, f"videos/chunk-000/observation.images.rgb/{si}.jpg")))
ax = fig.add_subplot(1, 4, 2); ax.imshow(img); ax.set_title(f"leg-A frame {si} (B seen)"); ax.axis("off")

# closest leg-A frame to B
ci = m["b_closest_idx"]
img2 = np.array(Image.open(os.path.join(ep, f"videos/chunk-000/observation.images.rgb/{ci}.jpg")))
ax = fig.add_subplot(1, 4, 3); ax.imshow(img2); ax.set_title(f"closest frame {ci}"); ax.axis("off")

# goal image (B viewed at pass heading)
g = np.array(Image.open(os.path.join(ep, "goal_image.jpg")))
ax = fig.add_subplot(1, 4, 4); ax.imshow(g); ax.set_title("goal image (B @ pass_yaw)"); ax.axis("off")

out = "/home/asus/Research/Nav/memnav_viz/twoleg_ep0_diag.png"
plt.tight_layout(); plt.savefig(out, dpi=110); print("saved", out)
print("depth sanity: frame0 range", end=" ")
d = np.array(Image.open(os.path.join(ep, "videos/chunk-000/observation.images.depth/0.png"))).astype(float) / 10000
print(f"[{d.min():.2f},{d.max():.2f}]m  nonzero={100*(d>0).mean():.0f}%")
