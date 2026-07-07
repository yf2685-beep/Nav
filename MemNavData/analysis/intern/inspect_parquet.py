import numpy as np, pandas as pd, json, os

base = "/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/matterport3d_d435i/17DRP5sb8fy/trajectory_10"
pq = os.path.join(base, "data/chunk-000/episode_000000.parquet")
df = pd.read_parquet(pq)
print("=== columns ===")
for c in df.columns:
    v = df[c].iloc[0]
    arr = np.array(v)
    print(f"  {c}: dtype-ish={type(v).__name__} shape={arr.shape}")
print("num rows (frames):", len(df))

# intrinsic
K = np.array(df['observation.camera_intrinsic'].iloc[0]).reshape(-1)
print("\n=== camera_intrinsic (row0) raw len", K.size, "===")
print(K.reshape(K.size//3 if K.size%3==0 else 1, -1) if K.size in (9,16) else K)

# extrinsic / action pose
for col in ['observation.camera_extrinsic','action']:
    if col in df.columns:
        a0 = np.array(df[col].iloc[0]); a1 = np.array(df[col].iloc[1])
        print(f"\n=== {col} row0 (shape {a0.shape}) ===")
        print(np.array_str(a0.reshape(-1)[:16], precision=4, suppress_small=True))
        d0 = a0.reshape(4,4) if a0.size==16 else None
        d1 = a1.reshape(4,4) if a1.size==16 else None
        if d0 is not None:
            print("translation row0:", np.array_str(d0[:3,3], precision=3))
            print("translation row1:", np.array_str(d1[:3,3], precision=3))
            print("step0->1 delta:", np.array_str(d1[:3,3]-d0[:3,3], precision=3))

# meta
info = json.load(open(os.path.join(base,"meta/info.json")))
print("\n=== info.json keys ===", list(info.keys()))
for k in ['fps','features','robot_type','total_frames','total_episodes']:
    if k in info: print(f"  {k}: {info[k] if k!='features' else list(info[k].keys())}")

with open(os.path.join(base,"meta/episodes.jsonl")) as f:
    ep = json.loads(f.readline())
print("\n=== episodes.jsonl[0] keys ===", list(ep.keys()))
print({k: ep[k] for k in ep if not isinstance(ep[k], list) or len(str(ep[k]))<200})
