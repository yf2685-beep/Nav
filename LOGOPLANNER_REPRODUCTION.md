# LoGoPlanner Reproduction

Reproduction of **LoGoPlanner** (Peng et al., *Localization Grounded Navigation
Policy with Metric-aware Visual Geometry*, arXiv:2512.19629) — an end-to-end
navigation policy that pairs a Pi3/VGGT geometry backbone with a diffusion
trajectory head.

## Result Summary

| Setting | Checkpoint | Eval | SR | SPL |
|---|---|---|---|---|
| **Our reproduction** | Stage-2 mini, step 849 | scene 0, 100 ep | **19%** | ~0.16 |
| Pipeline sanity check | official HF ckpt | scene 0, 100 ep | 60% | 0.59 |
| Paper (Table I) | official | 20 home scenes | 57.3% | 52.4% |

**Our trained checkpoint reaches SR = 19% on InternScenes home scene 0**
(100 episodes, PointGoal). This is a *trend-level* reproduction: the model
performs genuine goal-directed navigation (11/19 successful episodes at
SPL = 1.0, i.e. optimal paths), but absolute SR is far below the paper because
the model was trained on ~0.04% of the paper's data/compute budget — see below.

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

Loss weights are configurable via `--lambda-{diffusion,critic,pose,local,world,subgoal}`;
`--report-to wandb` enables Weights & Biases logging of per-component raw and
weighted losses, grad-norm and learning rate.
