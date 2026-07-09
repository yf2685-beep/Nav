# MemNavData — genuine multi-stop image-goal episode generator

`generate_twoleg.py` generates **genuine multi-stop navigation episodes** (2-leg `start→A→B`,
3-leg `start→A→B→C`) in Habitat on MP3D scenes, written in the **InternData-N1 layout** for
training the **memnav** implicit-memory image-goal policy. Unlike the manufactured "reverse-retrace"
data, each leg is an **independent geodesic**, so returns genuinely diverge from the outbound path.

- **2-leg**: B is a **revisit** goal on leg A (memory test).
- **3-leg**: B is a **novel** goal off leg A; C is a **revisit** of a leg-A place after the detour
  (long-range memory test).

Env: `conda activate habitat` (habitat-sim 0.3.3, python 3.9, EGL headless). GPU for rendering.

---

## Run

**Single scene, one leg type:**
```bash
python generate_twoleg.py --scene /path/<scene>.glb --navmesh "" \
    --out /path/out_dir --n 100 --n_legs 2          # or --n_legs 3
```

**Dual-leg (both types for one loaded scene — reuses sim + per-floor ESDF cache):**
```bash
python generate_twoleg.py --scene /path/<scene>.glb --navmesh "" \
    --out /path/OUT_ROOT --n2 100 --n3 100 --dA_min 4.0
# nests as  OUT_ROOT/mp3d_2leg/<scene_id>/episode_XXXX  and  OUT_ROOT/mp3d_3leg/<scene_id>/...
```
`--navmesh ""` recomputes the navmesh from the `.glb` at `--agent_radius` (recommended; the shipped
`.navmesh` is ignored either way since we re-bake with our radius/height). Resumable: a fully-populated
`out_dir` is skipped; the run is deterministic given `--seed`.

### HPC (SLURM array over scenes)
The generator does **one scene per process**; parallelize by putting the **scene loop in a SLURM array**:
```bash
#!/usr/bin/env bash
#SBATCH --job-name=memnav_gen
#SBATCH --array=0-N            # one task per scene
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
source ~/.bashrc; conda activate habitat
SCENES=(/scratch/mp3d/*.glb)                     # your scene list
SC=${SCENES[$SLURM_ARRAY_TASK_ID]}
python generate_twoleg.py --scene "$SC" --navmesh "" \
    --out /scratch/memnav_data --n2 200 --n3 200 --dA_min 4.0 \
    --window 32 --num_scale 8 --seed $SLURM_ARRAY_TASK_ID
```
Notes for bulk runs:
- The **safety gate adds retries** (rejected trajectories are resampled), so gen is slower in tight
  scenes. If acceptance is low, raise `--max_attempts` (default 60) or relax `--max_frame_turn`.
- `--window` **must match the LingBot precompute** you will run downstream (we use 32).

---

## Output (per episode, InternData-N1 layout)
```
episode_XXXX/
├── data/chunk-000/episode_000000.parquet   # per-frame: action (4x4 cam-to-world, Z-up), intrinsics
├── videos/chunk-000/observation.images.rgb/{i}.jpg
├── videos/chunk-000/observation.images.depth/{i}.png   # uint16, depth*10000
├── goal_1.jpg [, goal_2.jpg]  + goal_image.jpg          # goal views (B [, C])
└── meta/gen_meta.json
```
`meta.gen_meta.json` — `switches` (leg boundaries), `start`, `A`, and per goal:
`kind` (revisit/novel), `pos`, `yaw_habitat`, `covis`, `covis_argmax` (GT match frame / relocalization
anchor), `head_off_deg`, `recall_gap` (current−match; large = long-term), and **`covis_curve`**
(occlusion-aware co-visibility vs every history frame — the multi-positive retrieval label; the loader
thresholds it with `covis_pos_hi`/`covis_pos_lo`, recorded in meta).

---

## Key parameters
| flag | default | meaning |
|---|---|---|
| `--n2 / --n3` | — | dual-leg: #2-leg / #3-leg episodes for the scene |
| `--n / --n_legs` | 5 / 2 | single-mode count / leg type |
| `--dA_min/--dA_max` | 3 / 9 | geodesic start→A length (m). Use **4+** for W=32 |
| `--window / --num_scale` | 32 / 8 | LingBot streaming; sets `anchor_margin = num_scale+W−1 = 39` |
| `--covis_lo / --covis_hi` | 0.20 / 1.00 | revisit accept band on co-visibility |
| `--head_max_deg` | 45 | revisit: max heading offset vs matched frame |
| `--covis_pos_hi/lo` | 0.50 / 0.10 | retrieval positive/negative thresholds (loader-applied) |
| `--long_term_frac` | 0.7 | fraction of revisits forced outside the current window |
| `--goal_jitter_pos` | 1.50 | revisit goal disk radius around the anchor frame (m) |
| `--novel_covis` | 0.10 | 3-leg B: max co-visibility to count as novel |
| `--max_frame_turn` | 15 | safety gate: reject+retry if any frame turns more than this (deg) |
| `--agent_radius` | 0.30 | navmesh inflation = robot radius (collision margin) |
| `--max_attempts` | 60 | per-episode resample budget |

---

## Trajectory model (matches InternData-N1 dynamics)
geodesic → **ElasticBands** clearance smoothing → **pure-pursuit unicycle** tracking
(v≈0.0376 m/frame, lookahead 0.7 m, r_min 0.4 m, speed floor 0.48× in turns), per-floor confined
(`floor_tol` 0.8 m). Camera renders **forward-facing** (tracks realized motion); poses stored Z-up
cam-to-world (M_W).

---

## Changelog (2026-07-07 → 07-08)
- **Revisit sampling → anchor-centric**: pick a leg-A anchor frame X (index ≥ `anchor_margin`), sample
  B in a disk around X + heading cone; gate on co-visibility ∈ [lo,hi] **and** heading ≤ 45°.
- **Multi-positive retrieval label** `covis_curve` (occlusion-aware covis vs every history frame) +
  **stride-1 GT match** (argmax over all frames, not the sampling anchor). 3-leg C uses full history
  (leg1+leg2) so leg2 frames are hard negatives.
- **W=32 window**: `anchor_margin` auto = num_scale+W−1 = 39; `--long_term_frac`/`--min_recall_gap`
  control recent(in-view) vs long-term(implicit-memory) revisits.
- **Camera-facing fix**: `pursuit_track` integrates directly in one **habitat-yaw** convention →
  camera faces the **direction of travel** (was ~90° off — the old planar-θ-as-yaw bug). Stored yaw
  re-derived from the **realized displacement** (forward difference), gated for creep.
- **Clearance-aware turn direction**: at near-reversals, turn the *clear* side (probe both r_min arc
  centres) instead of the shorter way into a wall → eliminates creep/turn-in-place.
- **Trajectory safety gate**: reject+retry the episode if any frame turn > `--max_frame_turn` or any
  inter-frame segment leaves the navmesh (denser than the old midpoint check).
- **Viz** (`viz/viz_multileg.py`): heading arrow uses the camera-forward axis; occupancy height band
  fixed to reference the **floor** (`haby − cam_h`) so sloping/descending floors aren't mis-shown gray.

## Downstream (InternNav side — NOT in this repo)
1. **LingBot precompute** the generated episodes at the **same window (32)** —
   `InternNav/scripts/dataset_converters/precompute_lingbot_features.py`.
2. `memnav_dataset_lerobot.py` reads `covis_curve` to build positive/negative retrieval labels.
3. **PENDING policy change**: the RetrievalHead loss is single-target softmax-CE (`target=k_goal`) —
   genuine revisits need a **multi-positive** loss (soft-CE over positives, or per-frame BCE). See the
   `memnav-project` / `memnav-training-data` notes.

Validation set + BEV plots for spot-checking live under `/home/asus/Research/Nav/memnav_viz/validate_gated/`.
