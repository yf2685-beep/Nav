# LoGoPlanner Reproduction

Reproduction of **LoGoPlanner** (Peng et al., *Localization Grounded Navigation
Policy with Metric-aware Visual Geometry*, arXiv:2512.19629) — an end-to-end
navigation policy that pairs a Pi3/VGGT geometry backbone with a diffusion
trajectory head.

## Result Summary

| Setting | Checkpoint | Eval | SR | SPL |
|---|---|---|---|---|
| Our reproduction (baseline) | Stage-2 mini, step 849 | scene 0, 100 ep | 19% | ~0.16 |
| **Best mini-data config** | Stage-2 mini, **no-goal term removed** | scene 0, 100 ep | **23%** | **0.227** |
| Pipeline sanity check | official HF ckpt | scene 0, 100 ep | 60% | 0.59 |
| Paper (Table I) | official | 20 home scenes | 57.3% | 52.4% |

**Our trained checkpoint reaches SR = 19% on InternScenes home scene 0**
(100 episodes, PointGoal). This is a *trend-level* reproduction: the model
performs genuine goal-directed navigation (most successful episodes at
SPL = 1.0, i.e. optimal paths), but absolute SR is far below the paper because
the model was trained on ~0.04% of the paper's data/compute budget — see below.
A loss-function change (dropping the no-goal diffusion term) lifts SR to **23%**
— see *Loss-Weight Experiments*.

## Method

### Architecture (unchanged from paper)
- **Geometry backbone**: Pi3 / VGGT — a multi-frame visual-geometry transformer
  (ViT-L), producing camera pose, local point clouds and world point clouds.
- **Scale prior**: DepthAnythingV2-S features injected into the patch tokens so
  predictions carry absolute metric scale.
- **Heads**: local-point / camera-pose / world-point (auxiliary, Stage 1),
  diffusion action head + critic + sub-goal MLP (Stage 2).

### Training
Two stages, both on **InternData-N1 v0.1-mini** (`matterport3d_d435i`,
68 scenes, 3398 trajectories):

1. **Warm start** — `scripts/build_warmstart_ckpt.py` injects pretrained Pi3 +
   DepthAnythingV2-S weights into a LoGoPlanner checkpoint.
2. **Stage 1** — geometry losses (pose / local / world points), ViT frozen.
   Config: `scripts/train/configs/logoplanner_stage1.py`.
3. **Stage 2** — diffusion + critic + sub-goal losses, geometry backbone frozen.
   Config: `scripts/train/configs/logoplanner_stage2.py`. The evaluated
   checkpoint is `checkpoint-849` (849 optimizer steps).

### Training budget vs paper
| | Paper | Ours |
|---|---|---|
| Data | v0.1-full, 200k+ trajectories | v0.1-mini, 3398 trajectories (~1.7%) |
| Stage 2 | batch 32, 3 days (~250k steps) | batch 4, 849 steps (~0.3%) |
| Sample touches | ~8.3M | ~3.4k (~0.04%) |

The 19% / 57.3% gap is a training-budget gap, not a pipeline defect — confirmed
by the sanity check below.

## Evaluation

Client–server: `logoplanner_server.py` serves the policy; `eval_startgoal_wheeled.py`
runs IsaacSim and queries it. 100 random start–goal pairs per scene, 4–10 m apart.

```bash
# server (navdp env)
python baselines/logoplanner/logoplanner_server.py \
  --port 19999 --checkpoint <ckpt> --temporal_depth 8

# eval client (isaaclab env)
python eval_startgoal_wheeled.py --port 19999 \
  --scene_dir <Scene-N1/scenes_home> --scene_index 0 --scene_scale 0.01
```

`--temporal_depth` must match the checkpoint (8 for our trained ckpt, 16 for the
released HF ckpt). Our trained ckpt was trained with per-step goal updates, so
eval it with `--per_step_goal`; the HF release expects the default frozen goal.

### Three eval bugs fixed to get from SR=0% to SR=19%
1. **`policy.` prefix mismatch** — trainer saves weights under a `policy.`
   prefix; the agent loaded them into an unprefixed module with `strict=False`,
   silently dropping all 2098 tensors (model ran random weights). Fixed in
   `baselines/logoplanner/policy_agent.py`.
2. **`temporal_depth` mismatch** — eval defaulted to 16 decoder layers while
   training used 8, leaving half the decoder random. Added a `--temporal_depth`
   flag to `logoplanner_server.py`.
3. **Frozen goal** — `eval_startgoal_wheeled.py` froze the goal at episode start.
   Now controlled by `--per_step_goal` (default frozen, matching the release).

### Pipeline sanity check
Running the **official HF checkpoint** (`InternRobotics/LoGoPlanner`) through the
original eval code yields **SR = 60%** on scene 0 (100 episodes) — consistent
with the paper's 57.3% 20-scene average. This confirms the evaluation pipeline
is correct and the 19% is purely a function of our limited training budget.

## Loss-Weight Experiments

To probe whether SR can be improved purely by re-balancing the training loss
(no model/data/pipeline change), the trainer exposes per-component weights via
`--lambda-*` flags and logs per-component raw/weighted losses + grad-norm to
Weights & Biases. All runs below: mini data, Stage-2 config, 3000 steps,
warm-start init — differing only in loss weights.

### Critic weight sweep (`w_critic`: 0.0 → 2.0)
The Stage-2 total loss is `w_diffusion·action + w_critic·critic + w_subgoal·subgoal`.
Sweeping `w_critic` (baseline = 1.0) gives a clean dose-response in training
stability — grad-norm median rises monotonically with the critic weight:

| `w_critic` | 0.0 | 0.1 | 0.3 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| grad-norm (median) | 2 | 5 | 13 | 45 | 102 |
| diffusion raw loss | 0.07 | 0.07 | 0.09 | 0.16 | 0.35 |

A large critic weight destabilises training and competes with the diffusion
head for capacity. **But this did not move SR**: `w_critic=0.3` (the most stable
config) evaluated at SR 18% ≈ baseline 19%. On mini data, SR is limited by data
diversity (single scene group), not by loss-weight tuning.

### No-goal term removal — the change that helped
The action loss is `0.5·ng + 0.5·mg`, mixing a **no-goal** (exploration)
diffusion term and a **main-goal** (goal-conditioned) term. PointGoal evaluation
only needs goal-conditioned behaviour, so we set `w_nogoal=0, w_maingoal=1.0`
(via `--lambda-nogoal/--lambda-maingoal`) — putting all diffusion capacity into
goal-conditioned trajectory generation.

| Config | SR | SPL |
|---|---|---|
| baseline (`0.5·ng + 0.5·mg`) | 19% | ~0.16 |
| **no-goal removed (`1.0·mg`)** | **23%** | **0.227** |

SR by start distance (no-goal-removed run): ≤4.5 m 38%, 4.5–6 m 17%, 6–8 m 26%,
≥8 m 8% — long-range navigation remains the weak point. 21 of 23 successful
episodes are at SPL = 1.0 (optimal path).

This is the only loss change that measurably improved SR on mini data
(+4 SR points, SPL +0.07). With n=100 the SR delta is near the noise floor
(±~5%), but the SPL gain is more robust; multi-scene evaluation would harden it.

## LingBot-Map Backbone — Method 1 (this branch)

Branch `method1-lingbot-map` replaces the Pi3 geometry backbone with a **frozen
LingBot-Map** (`GCTStream`) + a trainable **Adapter** (AttnPool + Linear) that
pools the aggregator's per-frame patch tokens into 384-d `state_token` /
`scene_token` for the diffusion policy. This branch **drops** LoGoPlanner's
DA-S depth-prior fusion and resizes inputs to 518×518 (LingBot's native
resolution). Code: [`lingbot_map_geometry.py`](NavDP/baselines/logoplanner/lingbot_map_geometry.py).

Trigger with the `LOGO_BACKBONE` environment variable; default is the original
Pi3 backbone, so this branch's behaviour is unchanged for any caller that does
not opt in:

```bash
# Stage-1 (geometric supervision — heads on Adapter output, real preds)
LOGO_BACKBONE=lingbot_map LOGO_STAGE=1 \
  python InternNav/scripts/train/train.py --model-name logoplanner_stage1 ...

# Stage-2 / inference (no geometric heads, dummy zeros)
LOGO_BACKBONE=lingbot_map LOGO_STAGE=2 \
  python NavDP/baselines/logoplanner/logoplanner_server.py --port 19997 --checkpoint <ckpt>
```

> **Parallel work:** A complementary integration that **preserves the DA-S
> depth fusion** and uses `AggregatorStream` (instead of the frozen GCTStream
> + Adapter) lives on branch [`method2-lingbot-v2`](https://github.com/yf2685-beep/Nav/tree/method2-lingbot-v2),
> triggered by `LOGO_BACKBONE=lingbot_v2`. The two methods are intentionally
> parallel so the role of the depth prior can be isolated.

## Artifacts & Paths

Checkpoints and eval results are large and machine-local (not committed to git).
Locations (machine-specific):

### Checkpoints
| What | Machine | Path |
|---|---|---|
| Warm-start (Pi3 + DA-S injected) | 131 | `/media/cvpr/yuxuan/logoplanner/checkpoints/logoplanner_warmstart/` |
| Training runs (loss-weight / nogoal / Stage-1) | 131 | `/media/cvpr/yuxuan/logoplanner/Nav/InternNav/checkpoints/<run>/ckpts/checkpoint-*logoplanner.ckpt` |
| Eval copies (critic_lo / nogoal_off / nogoal_cri03) | 186 | `/home/nyuair/data-eval/checkpoints_lw/` |
| Official HF checkpoint | 186 | `/home/nyuair/data-eval/checkpoints_hf/logoplanner_policy.ckpt` |

`<run>` ∈ `logo_lw_{baseline,critic_lo,critic_hi,critic_off,cri0.1..cri0.6,subgoal_hi,diffusion_hi}`,
`logo_nogoal_off`, `logo_nogoal_off_cri0.3`, `logo_s1long`.

### Eval results
| What | Machine | Path |
|---|---|---|
| **nogoal_off SR=23%** (metric.csv + eval log) | 186 | `/home/nyuair/data-eval/eval_results_backup/nogoal_off_scene0_*` |
| nogoal_off metric.csv (mirror) | 127 | `1 robot navigation/eval_results/nogoal_off_scene0_metric.csv` |
| 20-scene paper eval (per-scene, archived live) | 186 | `/home/nyuair/data-eval/eval_results_backup/20scene/` |

`metric.csv` columns: `success, spl, ne, le, distance` — one row per episode.
Eval runs archive each scene's `metric.csv` into `eval_results_backup/` the moment
it finishes, so results are never lost to an output-dir overwrite.

## Reproducing

```bash
# 1. build warm-start checkpoint
python InternNav/scripts/build_warmstart_ckpt.py

# 2. stage 1
python InternNav/scripts/train/train.py --model-name logoplanner_stage1 \
  --ckpt-to-load <warmstart_dir> --load-from-ckpt True

# 3. stage 2  (loss weights / wandb optional, see below)
python InternNav/scripts/train/train.py --model-name logoplanner_stage2 \
  --ckpt-to-load <stage1_ckpt> --load-from-ckpt True

# 4. evaluate (see Evaluation section)
```

Loss weights are configurable via
`--lambda-{diffusion,critic,pose,local,world,subgoal,nogoal,maingoal}`;
`--report-to wandb` enables Weights & Biases logging of per-component raw and
weighted losses, grad-norm and learning rate. The best mini-data result so far
uses `--lambda-nogoal 0.0 --lambda-maingoal 1.0` (see *Loss-Weight Experiments*).
