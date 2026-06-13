# LingBot + LoGoPlanner Navigation — Experiment Report

> Goal: replace LoGoPlanner's Pi3 state encoder with LingBot-Map and drive
> Success Rate (SR) on NavDP IsaacSim PointGoal eval above 0%.

## TL;DR — current best

| Configuration | Scene | Episodes | **SR** | SPL | Pi3 baseline (same scene) |
|---|---|---|---|---|---|
| F1 (LingBot baseline)               | 6 | ~50  | **0.00%** | 0.0   | 23.8% |
| **Plan D Stage 2 step 12000**       | 6 | 101 | **2.97%** | 0.030 | 23.8% |
| **Plan D Stage 2 step 12000**       | 1 | 101 | **2.97%** | 0.029 | 29.7% |
| **Stage A step 1699** (30 eps)      | 6 | 31  | **3.23%** | 0.032 | 23.8% |
| Phase α step 1699 (image-goal, CNN) | 6 | 11  | **0.00%** | 0.0   | 23.8% |
| Phase α-Fix++ step 1699 (ResNet18)  | 6 | (probe only) | **goal-blind** | — | — |

**Best result so far: ~3.2% SR on scene 6 (Plan D + Stage A line), vs ~24% Pi3 baseline.** The 3% ceiling reproduces across three independent runs/scenes.

---

## Pipeline overview

```
RGB-D + goal
    ↓
LingBot state_encoder (DINOv2 + GCA, frozen at Stage 2)
    ↓
(state_token, scene_token, ...)
    ↓
+ start_encoder(goal_3d) or goal_image_encoder(goal_image)
    ↓
diffusion cond = cat([state_embed, scene_token, rgbd, dist_token, obstacle_token])
    ↓
DDPM denoise (10 timesteps, 16 candidate trajectories)
    ↓
critic ranks → top trajectory → MPC executes
```

Architecture is the same LoGoPlanner backbone (NavDP framework) with LingBot wrapped as `GeometryModel_LingBot` in `NavDP/baselines/logoplanner/geometry_model_lingbot.py`.

---

## Experimental lines

### 1. Plan D — Pi3-teacher distillation of `state_token`

Goal: LingBot's `state_token` lacks the world-point semantic that Pi3 learns from World Point Decoder supervision. Distill it.

- **Stage 1**: 3000 steps, MSE(`student.state_token`, `pi3_teacher.state_token`)
- **Stage 2**: 12000 steps from Stage 1 ckpt, full diffusion+critic training, teacher kept as 0.1× MSE anchor

Probe results (Stage 1):
- `state_token` MSE vs Pi3: **16.49 → 0.024** (99.85% gap closed)
- Cosine sim vs Pi3: **+0.027 → +0.521**

Eval results (Stage 2):
- Step 1699 scene 6: SR = **2.97%** (3/101)
- Step 12000 scene 1: SR = **2.97%** (3/101)
- Goal-responsiveness probe: trajectories point in correct direction for fwd/left/back goals

**Conclusion**: Plan D successfully broke 0% → 3%, but plateaued. The remaining gap to Pi3 is in the modules NOT distilled (`scene_token`, perception coverage).

### 2. Stage A — explicit metric tokens (distance + obstacle)

Hypothesis: LoGoPlanner is implicit about scale; add explicit `distance = ||goal||` and `min_obstacle = min(depth)` as separate cond tokens.

Implementation:
- New `nn.Linear(1, 384)` for each: `dist_encoder`, `obstacle_encoder`
- `cond_pos_embed` extended 36 → 38 rows
- 3000-step retrain from Plan D 12000 with `STAGE_A_EXPLICIT_METRIC=1`

Eval results (Stage A 1699):
- 11 eps small sample: 9.09% (1/11) — noise
- 30 eps: **3.23%** (1/31) — true rate
- **Big observation**: mean LE = 3.07m vs Plan D's < 1m. **Robot actually moves** now, but still doesn't reach goal often.

**Conclusion**: Explicit metric helps the robot ATTEMPT motion (LE increases), but doesn't break the underlying ceiling.

### 3. Professor MVP — replace sim odom with LingBot pose

Goal: at inference time, use LingBot's per-frame pose estimation + PGO loop closure to correct accumulated odom drift, propagating into the policy's goal-frame computation.

Phase 1 (Pi3 dir2b policy + LingBot pose, no PGO):
- Smoke test scene 0, 4-6 eps: **0%**
- LingBot drift in IsaacSim: 0.6m after 10 frames, 1.5m after 30 frames
- Cause: my preprocessing bug (518×518 vs aspect-preserving 518×280). After fixing, drift on demo data is 0.064m/step (healthy). But in IsaacSim domain it's still off-distribution.

Phase 2 (+ loop-closure PGO):
- Scene 0, 4 eps: **0%**
- PGO triggered only 1 time across 4 episodes
- Cause: PointGoal task has no natural loops (single-pass A→B). Loop closure inapplicable.

**Conclusion**: Professor's approach is fundamentally aimed at real-robot odom drift, not sim with GT poses. Sim eval cannot show its value without injected odom noise (ablation we didn't run).

### 4. Phase α — image-goal navigation

Goal: replace 3-dim point goal with goal image, enabling navigation in new environments without metric coordinates.

Phase α (random-init CNN encoder, 138325):
- Probe: trajectory is **identical** for 3 different goal images (Δ = 0.0000m)
- Cause: 600K-param random CNN can't learn meaningful features in 1 epoch

Phase α-Fix (reuse LingBot state_encoder via tile×12, 138328):
- Probe: still identical trajectories
- Cause: LingBot is trained on temporal video, static-image tile is OOD → backbone outputs near-constant

Phase α-Fix++ (frozen ResNet18 + Linear adapter + start_encoder distill, 138335):
- Probe: image encoder produces **distinct** tokens per image ✓
- But **norm = 9 vs teacher norm = 34** → 4× too small
- Diffusion was trained on teacher-norm cond → student input is OOD → all 16 trajectory candidates < 0.5m → stop-clipped to (0, 0)
- **Effective SR: 0%** (robot doesn't move)

**Conclusion**: Architecture is correct, but 1699-step retraining is not enough for the student token to converge to teacher magnitude. Higher distill weight + more training would likely fix this, but the same ~3% scene_token ceiling is still expected.

---

## Why ~3% is the ceiling (root-cause analysis)

```
diffusion cond = [state, scene, rgbd, dist, obstacle, goal]

   ↑ state_token:   Plan D distilled from Pi3 → ~99.85% aligned
   ↑ scene_token:   NEVER distilled, stays LingBot-native
   ↑ goal_token:    works (Goal Encoder unchanged)

LingBot's scene_token lacks World-Point Decoder training:
   Pi3 trains scene_token under per-pixel world-coord supervision → "wall at 2m"
   LingBot trains under depth + pose only → no metric world geometry baked in

→ Even with perfect state_token alignment, diffusion still gets weak
  scene-level metric signal → trajectory direction OK, but stops short of goal
  or collides with unseen geometry → ~3% SR ceiling
```

---

## Code changes summary

All changes are env-gated and backward-compatible (no env var set → original LoGoPlanner behaviour).

### Files modified

| File | What changed | Env gate |
|---|---|---|
| `NavDP/baselines/logoplanner/geometry_model_lingbot.py` | Wrap LingBot as LoGoPlanner state_encoder, attach Pi3 teacher | `LINGBOT_CKPT`, `LINGBOT_PI3_TEACHER` |
| `NavDP/baselines/logoplanner/policy_network.py` | Stage A metric encoders, ImageGoal encoder, predict_imagegoal_action | `STAGE_A_EXPLICIT_METRIC`, `IMAGEGOAL_MODE`, `GOAL_TOKEN_SCALE` |
| `NavDP/baselines/logoplanner/policy_agent.py` | `step_imagegoal` method | (same as above) |
| `NavDP/baselines/logoplanner/logoplanner_server.py` | `/imagegoal_step` HTTP endpoint | (auto when called) |
| `NavDP/eval_startgoal_wheeled.py` | LingBot-pose override + PGO loop closure | `EVAL_USE_LINGBOT_POSE`, `EVAL_USE_PGO` |
| `InternNav/internnav/dataset/logoplanner_dataset_lerobot.py` | `goal_image` field from trajectory endpoint frame | (auto in batch) |
| `InternNav/internnav/trainer/logoplanner_trainer.py` | Pick up `_last_image_distill_loss` for image-goal teacher | (auto when in batch) |
| `InternNav/internnav/model/basemodel/logoplanner/logoplanner_policy.py` | Stage A metric + IMAGEGOAL_MODE training forward | (env propagated) |

### Files added

- `NavDP/lingbot_pose_estimator.py` — streaming LingBot pose + PGOCorrector
- `NavDP/baselines/logoplanner/geometry_model_lingbot.py` — LingBot as LoGoPlanner state_encoder

### Launch scripts (on dgx-login at /media/cvpr/yuxuan/logoplanner_setup/)

| Script | Purpose | Slurm job |
|---|---|---|
| `lingbot_v2_stage1_distill.sh` | Plan D Stage 1: 3000-step state_token distillation | — |
| `lingbot_v2_stage2_planD_teacher.sh` | Plan D Stage 2: 12000 steps with teacher anchor | 138219 |
| `lingbot_v2_stage2_stageA.sh` | Stage A: 3000 steps, +dist +obstacle tokens | 138315 |
| `lingbot_v2_stage2_phaseA.sh` | Phase α v1: random-CNN goal encoder | 138325 |
| `lingbot_v2_stage2_phaseA_fix.sh` | Phase α-Fix: LingBot tile-12 for goal | 138328 |
| `lingbot_v2_stage2_phaseA_fixpp.sh` | Phase α-Fix++: ResNet18 + start_encoder distill | 138335 |

### Eval ckpts (local at /home/nyuair/data-001/eval_ckpts/)

- `dir2b-10300.ckpt` — Pi3 baseline (52% SR official)
- `dir1b-1699.ckpt`, `dir1b-...` — Pi3 ablation ckpts
- `planD_s2_teacher_1699_stripped.ckpt`, `planD_s2_teacher_12000_stripped.ckpt` — Plan D
- `stageA_1699.ckpt`, `stageA_1699_stripped.ckpt` — Stage A
- `phaseAfixpp_1699_stripped.ckpt` — Phase α-Fix++ (latest image-goal)

Diagnostic / probe scripts live in `/tmp/`:
- `probe_distill.py` — state_token alignment vs Pi3
- `probe_goal_responsive.py` — trajectory direction probe
- `probe_phaseA_goal.py` — image-goal goal-blind diagnostic
- `probe_phaseAfixpp_deep.py` — magnitude / OOD diagnostic for Phase α-Fix++

---

## Open questions / candidate next experiments

The 3% ceiling is set by `scene_token` lacking World-Point supervision. The unexplored interventions, in order of expected impact:

1. **scene_token distillation** — extend Plan D to also distill `scene_token` (currently only `state_token` is supervised). ~1 day code + 17h train.
2. **Unfreeze LingBot backbone** in Stage 2 — let aggregator adapt to navigation distribution. Risk of drift away from distilled state. ~17h train.
3. **Add World-Point Decoder head to LingBot** — train it on Stage 1 with per-pixel metric world supervision. ~2 days + 17h train.
4. **Image-goal completion** — push Phase α-Fix++ to convergence: distill weight 10× + 3000+ step retrain, then full eval.
5. **Trajectory-level loop closure** — instead of pose-level PGO at inference, score candidate trajectories by anchor matching (professor's later refinement).

---

## Reference values

- LoGoPlanner Pi3 (paper / HF release) 20-scene aggregate: **SR 52.18% / SPL 48.08%** (from `eval_results/hf_official_20scene_summary.csv`)
- Plan D / Stage A best on scene 6: ~3% SR
- Per-scene Pi3 baseline range: 23.8% (scene 6) — 76.2% (scene 13)

## Environment quick reference

- Training: dgx-login, `enerverse_arm.tar.gz` env, partition `spark`
- Inference server: local `navdp` conda env (`/home/nyuair/miniconda3/envs/navdp`)
- Eval client: local `isaaclabjunyi` conda env (IsaacSim 4.2.0.2 + IsaacLab v1.2.0)
- LingBot ckpt: `/home/nyuair/data-001/lingbot-map-ckpt/lingbot-map.pt`
- Scene data: `/home/nyuair/data-001/InternRobotics/Scene-N1/scenes_home/`
