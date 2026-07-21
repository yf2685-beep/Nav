# MemNav — session handoff (2026-07-06)

Read this + `SUMMARY.md` + `PILOT_RUN.md` to continue in a new session. Everything below is on
machine **127** (hostname `nyuair`, 10.224.36.127). Memory notes also auto-load:
`memnav-backbone-setup`, `memnav-train-env`, `memnav-pilot-results`.

## Where we are — DONE & verified
- **① backbone**: LingBotStream == official forward, cosine 1.0.
- **② data**: 130 precomputed caches across 10 scenes under
  `/home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp/matterport3d_d435i/`.
- **③ training**: single-scene 150-step checkpoint = `checkpoints/memnav_pilot.ckpt` (238MB,
  trained heads only). Trainer = `scratchpad/standalone_train.py` (env `enerverse`). A multi-scene
  400-step retrain was started but the training script got renamed to `.DISABLED` mid-run — see below.
- **④ inference chain**: `memnav_infer.py` + `policy_agent.py` + `memnav_server.py`. Proven to
  GENERATE TRAJECTORIES (HTTP 200 + real 24×3 waypoints) via `verify_fake_client.py`, and
  in-process via `verify_agent_inprocess.py` (3 real steps). Also `viz_trajectory.py` → `traj_viz.png`.
- **⑤ core contribution**: **seen-vs-unseen AUC = 0.924** (`eval_seen_unseen_stats.py`,
  `seen_unseen_stats.png`). seen gate 0.78 vs unseen 0.25. Fine localization weak (31% within ±5,
  `eval_retrieval_localization.py`).
- **Deliverables**: `SUMMARY.md`, `report.html` (published artifact), `traj_viz_page.html` (published).

## THE OPEN ITEM — IsaacSim closed-loop (server keeps getting KILLED)

**ROOT CAUSE FOUND (definitive):** a user-level GPU guard is killing the memnav processes:
`/home/nyuair/junyi/chatsign/phase6_augment/enhance_smpl/scripts/gpu_guard_chatsign.sh`
(runs as nyuair, 60s loop, log at `.../enhance_smpl/work/gpu_guard.log`). Every 60s it lists all
GPU compute PIDs and `kill -TERM`/`-9`s any that are NOT whitelisted — AND kills the parent .sh/python.
Whitelist (`is_ours()`) only passes cmd containing `avatar_api_server` / `/home/nyuair/junyi/` /
`/envs/GUAVA/bin/python`, or CWD under `/home/nyuair/junyi*` or `*/GUAVA-origin*`. The memnav server
(CWD `.../Nav/NavDP/...`, env enerverse) is NOT whitelisted → killed within 60s. This explains the
intermittent silent death (no traceback, nothing in dmesg — it's a userspace kill -9) and the training
launcher being taken out. **It is NOT a compute shortage — 34–41GB GPU is free.**

This is a COORDINATION decision (the guard protects a colleague's chatsign/GUAVA job). Options,
pick with the user:
1. **Whitelist memnav** — add one case to `is_ours()` in that guard script, e.g.
   `*/NavDP/baselines/memnav/*|*memnav_server*) return 0 ;;`  (lets memnav coexist; there's plenty
   of free GPU). Least disruptive; needs owner's OK to edit their script.
2. **Pause the guard while running** — `kill <guard_pid>` (pgrep -f gpu_guard_chatsign), run the eval,
   restart it. Only if the chatsign job can spare the GPU. Don't do this without the user's say-so.
3. Run on a genuinely free machine/GPU with no such guard.

**The threaded fix was also applied** (may still be good hygiene): `memnav_server.py` now builds the
agent once in the MAIN thread at startup and uses `app.run(host="127.0.0.1", port=args.port)` (dropped
`threaded=True`). But this was NOT the blocker — the guard was.

### First thing to do next session — verify the fix
```bash
# terminal A — server (enerverse)
conda activate enerverse
cd "/home/nyuair/yuxuan/1 robot navigation/Nav/NavDP/baselines/memnav"
CUDA_VISIBLE_DEVICES=0 python memnav_server.py --port 8888 \
  --checkpoint checkpoints/memnav_pilot.ckpt \
  --lingbot_repo /home/nyuair/yuxuan/lingbot-map \
  --lingbot_weights /home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt
# wait for "[memnav_server] agent ready — serving."

# terminal B — hammer it with many real steps; if it survives >20 steps, the crash is FIXED
conda activate enerverse
python scratchpad/verify_fake_client.py     # (or the copy verify_fake_client.py in this dir)
```
If it survives many steps → run the real closed loop:
```bash
# server still running in A; then in B:
conda activate isaaclabjunyi
cd "/home/nyuair/yuxuan/1 robot navigation/NavDP"
python eval_imagegoal_wheeled.py --port 8888 \
  --scene_dir /home/nyuair/junyi/NavDP-old/asset_scenes/cluttered_hard --scene_index 0 --scene_scale 1.0
```
Or the auto-runner (waits for free GPU, supervises, retries): `bash run_queue.sh` (see PILOT_RUN.md).
Vis images land in `NavDP/imagegoal_memnav_cluttered_hard/`.

### If it STILL crashes after the fix
- Compare against the **known-working** `logoplanner_server.py` in this baselines/ dir (it also uses
  LingBot) — match its structure exactly (it builds the agent in `navigator_reset`, uses imageio
  writers, no threaded arg). "参考之前的通路" = user has run IsaacSim before; the working path is
  navdp/logoplanner + `conda activate isaaclab`/`isaaclabjunyi`.
- Check `sudo dmesg -T | tail` right after a death for Xid/segfault (note: `kill -9` from another
  process does NOT log to dmesg).

## Key facts / gotchas
- Envs: **`enerverse`** = server+training (torch 2.4 + diffusers; fixed with gym/wcwidth/tyro,
  numpy==1.26.4, open3d guarded, Long-CLIP cloned). **`isaaclabjunyi`** = IsaacSim eval (has
  `omni.isaac.lab.app`). Have passwordless **sudo**.
- LingBot weight: `/home/nyuair/yuxuan/lingbot-map/weights/lingbot-map-long.pt` (4.4GB).
- **Textured** scenes (render OK): `/home/nyuair/junyi/NavDP-old/asset_scenes/cluttered_hard`
  (`--scene_scale 1.0`). scenes_home renders black (broken MDL). `NavDP/assets/scenes/cluttered_hard`
  has only .npy pairs, no USD.
- Two NavDP trees: eval runs from `/home/nyuair/yuxuan/1 robot navigation/NavDP/` (has the scenes-adjacent
  configs); the memnav baseline code lives in `.../Nav/NavDP/baselines/memnav/`. Server is location-independent (HTTP).
- Inference recomputes the LingBot cache from the live frame buffer each step (O(k²)/episode) — slow but correct.
- 186 machine: separate filesystem (no repo/data/weights/enerverse), GPU0 full, GPU1 only 2GB — not a quick option.

## Optional next steps (research)
- Resume multi-scene training (from `scratchpad/memnav_ckpt_multiscene/memnav.ckpt`, step 50) to lift
  fine retrieval. `standalone_train.py` now has `--resume` + fault-tolerance.
- Run closed-loop SR vs a memoryless baseline (NavDP), split by seen/unseen — the paper-grade result.
