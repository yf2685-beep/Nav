# MemNav pilot run — end-to-end on machine 127 (2026-07-04)

Goal: get the `upstream-glbreeze` MemNav (implicit-memory image-goal nav) **running end to end**
and see a navigation signal. This documents what was set up + how to reproduce / continue.

## TL;DR status

| step | what | status |
|---|---|---|
| ① backbone | LingBotStream vs official forward | ✅ cosine=1.0 (lossless) |
| ② data | precompute LingBot features, 40-traj pilot | ✅ caches in `data-001/.../matterport3d_d435i` |
| ③ train | trainable heads (60.75M) on frozen LingBot | ✅ learns (loss↓, retrieval acc↑, gate seen>unseen) |
| ④ inference | infer engine + agent + Flask server | ✅ built + smoke-passed (`memnav_infer.py`, `policy_agent.py`, `memnav_server.py`) |
| ⑤ eval | seen/unseen memory demo + IsaacSim | demo ✅ ; IsaacSim = runbook below |

## Environment (machine 127)

- **`enerverse` conda env** (torch 2.4, diffusers 0.30.3, transformers 4.47) for training + the
  MemNav server. Fixed up: `pip install gym wcwidth tyro`; `numpy==1.26.4`; guarded `import open3d`
  in `internnav/dataset/navdp_dataset_lerobot.py`; cloned `Long-CLIP` into
  `InternNav/internnav/model/basemodel/Long-CLIP` (the `LongCLIP` symlink resolves).
- **LingBot weight** (frozen backbone): `/home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt`
  (4.4GB, loads 0-missing/0-unexpected). Repo: `/home/nyuair/yuxuan/lingbot-map`.
- **Data**: `/home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp/matterport3d_d435i`
  (68 scenes; 40 pilot trajectories precomputed, all in scene 17DRP5sb8fy).

## ② Precompute more features (to scale beyond the 40-traj pilot)

Cache is ~1.23 GB/traj (scale_k/v dominate). Full 3398 traj ≈ 4.2 TB — infeasible; precompute a subset.

```bash
conda activate enerverse
cd "InternNav/scripts/dataset_converters"
python precompute_lingbot_features.py \
  --root_dirs /home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp \
  --lingbot_repo /home/nyuair/yuxuan/lingbot-map \
  --weights /home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt \
  --image_size 518 --num_scale_frames 8 --use_sdpa --limit <N>
```

## ③ Train

`scripts/train/configs/memnav.py` is pointed at the 127 paths. The canonical launcher
(`start_train.sh` / `train.py`) pulls in the whole InternNav toolbox (needs uvicorn/lmdb/… ),
so the pilot used a **standalone loop** that only imports the memnav pieces:
`scratchpad/standalone_train.py` (Adam, loss = 0.5·ng+0.5·mg + retrieval-CE + 0.5·aux-pose,
saves only non-LingBot tensors → `memnav.ckpt`). Perf: ~55 s/step (encode_memory reloads each
sample's 1.2 GB cache; disk-I/O bound).

```bash
conda activate enerverse
python scratchpad/standalone_train.py --steps 150 --batch 8 --save_every 25 --out <ckpt_dir>
```

## ④ Inference chain (this dir)

- `memnav_infer.py` — `MemNavInference`: reuses the trained InternNav `MemNavNet`, drives it from a
  live frame buffer (recomputes the LingBot cache per step via precompute's `extract_trajectory`),
  runs retrieval → window/pose/goal-append → DDPM sampling → local trajectory `cumsum(naction/4)`.
- `policy_agent.py` — `MemNav_Agent`: per-env frame buffers, LingBot preprocessing, warmup nudge
  for the first `lo=15` frames, batch assembly.
- `memnav_server.py` — Flask server, same HTTP contract as the other baselines
  (`/navigator_reset`, `/navigator_reset_env`, `/imagegoal_step`).

Note (perf): the cache is recomputed from the whole buffer each step (O(k²)/episode). Correct +
identical to training; a later optimization is true incremental streaming.

## ⑤ Run the IsaacSim image-goal eval (runbook)

Two processes. **Terminal A** — MemNav server (enerverse):

```bash
conda activate enerverse
cd "NavDP/baselines/memnav"
python memnav_server.py --port 8888 \
  --checkpoint <ckpt_dir>/memnav.ckpt \
  --lingbot_repo /home/nyuair/yuxuan/lingbot-map \
  --lingbot_weights /home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt
```

**Terminal B** — IsaacSim eval. Use the env with the old `omni.isaac.lab` API = **`isaaclabjunyi`**
(already headless). **Use the TEXTURED cluttered_hard USDs** at
`/home/nyuair/junyi/NavDP-old/asset_scenes/cluttered_hard` (each `hard_*` has a self-contained
`cluttered-*.usd` + `imagegoal_start_goal_pairs.npy`). Do NOT use scenes_home — its MDL materials
fail to compile → black render; and `NavDP/assets/scenes/cluttered_hard/hard_*` only ship the
`.npy` pairs, no USD geometry.

```bash
conda activate isaaclabjunyi
cd "NavDP"
python eval_imagegoal_wheeled.py --port 8888 \
  --scene_dir /home/nyuair/junyi/NavDP-old/asset_scenes/cluttered_hard --scene_index 0 --scene_scale 1.0
```

**Verified so far:** IsaacSim starts headless fine, cluttered_hard loads, the eval reaches the
`navigator_reset` / `imagegoal_step` loop, and the server **returns valid trajectories** (see next
section). **Blocker:** on this shared machine the server process gets **killed after ~1 heavy
inference** (no traceback, no OOM in dmesg — same watchdog that renamed the training script to
`.DISABLED`). The full closed loop needs a stable / dedicated GPU session; the code + scenes are ready.

## How to verify trajectories are being generated

The inference code is proven to generate trajectories two independent ways:
- **In-process** (`scratchpad/agent_smoke.py`): feeds real frames to `MemNav_Agent`, does 3
  consecutive REAL diffusion steps with no crash — prints `exec=(1,24,3)` finite trajectories.
- **Over HTTP** (`scratchpad/fake_client.py`): a fake IsaacSim client that POSTs jpg-encoded
  frames to `/imagegoal_step` exactly like `client_utils.imagegoal_step`, and prints the returned
  `trajectory`. Warmup steps 0–14 return the forward nudge; step 15 returns a real DDPM trajectory
  (e.g. endpoint (x,y,θ) = (-0.26, +0.72, +0.32)), all HTTP 200.

Signals to watch during a real eval:
- **server log**: `POST /imagegoal_step HTTP/1.1" 200` = a trajectory was returned.
- **response JSON**: has `trajectory` (24×3 waypoints) + `all_trajectory` (4 samples) + `all_values`.
- **eval log**: `Planning time: ...` = the robot received a trajectory; `Planning error` = server unreachable.
- **eval visualization**: saved under `imagegoal_<algo>_<scene>/` — the planned trajectory drawn on the frame.

## Offline navigation-effect demo (no IsaacSim)

`scratchpad/demo_seen_unseen.py --ckpt <ckpt>` — on one real trajectory, shows the memory
mechanism: a **seen** goal (already-visited frame) → high `revisit_gate`, retrieval locks near the
goal frame; an **unseen** goal (future frame) → gate falls to NULL. Saves `nav_demo.png`.
