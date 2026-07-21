# MP3D Revisit Dataset — end-to-end workflow

Goal: build & train on the professor's **revisit-focused** MP3D dataset (covisibility-labeled
2-leg/3-leg episodes, contrastive-loss ready). The professor's `MemNavData/` (upstream `glbreeze/Nav`)
ships the **generation pipeline**, not ready data — the actual v0 data (7 episodes, 1 scene) lives on
the professor's machines and is not reachable from 127/131. So we generate our own.

---

## Step 1 — Sign the Matterport3D license
1. Open **https://niessner.github.io/Matterport/** (official MP3D project page).
2. Download the **Terms of Use** PDF.
3. Fill/sign: name, institution = **NYU**, email = **yfang@nyu.edu** (academic email is the gate), advisor.
4. Email the signed PDF to **matterport3d@googlegroups.com** (per the page).
5. They reply (≈1–3 days) with **`download_mp.py`** (carries your download access).
   > Verify the exact contact/URL on the page — it can change.

## Step 2 — Download MP3D (Habitat version = `.glb` + `.navmesh`)
```bash
python download_mp.py --task habitat -o /path/to/mp3d     # habitat-ready scenes only (smaller)
```
Each scene → `<scene_id>/<scene_id>.glb` + `<scene_id>.navmesh` (v0 used `17DRP5sb8fy`).
Start with a few scenes to validate before pulling all ~90 houses.

## Step 3 — Generation env (habitat-sim)
```bash
conda create -n habitat python=3.9 -y && conda activate habitat
conda install habitat-sim headless -c conda-forge -c aihabitat -y   # headless for cluster
pip install numpy numpy-quaternion pillow
```
> ARM caveat: 131 compute nodes are aarch64 (GB10); habitat-sim aarch64 builds may not exist → may need
> a source build. x86 (186/127) is safer for habitat. See "Compute target" below.

## Step 4 — Generate revisit trajectories  (`MemNavData/generate_twoleg.py`)
```bash
# 2-leg (start -> A -> B, B is a backtrack near a past frame)
python MemNavData/generate_twoleg.py \
  --scene /path/to/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb \
  --navmesh /path/to/mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh \
  --out /path/to/gen/mp3d_2leg --n 50 --n_legs 2
# 3-leg (start -> A -> B -> C)
python MemNavData/generate_twoleg.py --scene ... --navmesh ... \
  --out /path/to/gen/mp3d_3leg --n 30 --n_legs 3
```
Key args (defaults are sane): `--covis_lo 0.20 --covis_hi 1.00` (revisit covisibility band),
`--covis_pos_hi 0.50 --covis_pos_lo 0.10` (contrastive pos/neg thresholds),
`--window 32 --num_scale 8` (**must match precompute**). Repeat per scene to build volume.
Output = InternData-N1 layout: `data/chunk-000/episode_000000.parquet`, `videos/.../rgb|depth`,
`meta/gen_meta.json`, `goal_image.jpg`.

## Step 5 — Precompute LingBot caches
Generated episodes have RGB/depth but training needs `lingbot_cache.npz`. Use the professor's
`InternNav/scripts/train_memnav/precompute_lingbot_mp3d.sbatch` (needs lingbot + GPU).
Professor uses `--skip_scale` (no scale_k/v stored, recomputed on the fly → smaller npz).
`window`/`num_scale` MUST equal Step 4.

## Step 6 — Train
Point memnav at the generated data (config now supports env vars):
```bash
export MEMNAV_ROOT_DIR=/path/to/gen/vln_n1/traj_data
export LINGBOT_REPO=/path/to/lingbot-map
export LINGBOT_WEIGHTS=$LINGBOT_REPO/weights/lingbot-map-long.pt
# then torchrun / sbatch scripts/train/train.py --name <run> --model-name memnav
```
Our retrieval-loss change (windowed soft-label + recall bias) is already in
`InternNav/internnav/trainer/memnav_trainer.py`. The revisit data's `covis_curve`/`covis_argmax`
also enables a proper **contrastive** retrieval loss later.

---

## Compute target (unresolved logistics)
| Machine | Arch | GPU | Blocker |
|---|---|---|---|
| 127 | x86 | 48G | chatsign GPU guard SIGKILLs memnav every 60s (policy, 2026-07-06) |
| 131 | **aarch64** | GB10 | env must be rebuilt for ARM; habitat-sim ARM build uncertain |
| 186 | x86 | 48G | currently only ~2G free (occupied); has `enerverse` already |

Generation (habitat) + training both need a usable **x86** GPU. Most realistic: wait for 186 to free,
or resolve a 127 guard exception. 131 needs a full ARM env rebuild first.

## Status / next (as of 2026-07-08)
- ✅ retrieval-loss change + reactive obstacle-avoidance governor + checkpoint-load fix (`core.` strip) done.
- ✅ diagnosed 127 guard; 131 arch mismatch (x86 pack won't run on ARM nodes) — 131 job failed on `Exec format error`.
- ⏳ user signing MP3D license.
- TODO: pick x86 compute, build habitat env, validate generate→precompute→train chain on a demo scene,
  then batch-generate once MP3D lands. Reconcile upstream merge (professor's trainer +30 lines vs our loss).
