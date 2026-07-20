# MemNav pilot — project summary (2026-07-05)

From "can you see the `upstream-glbreeze` branch?" to a **working end-to-end MemNav pipeline**
with a quantified demonstration of its core contribution.

## What MemNav is
Implicit-memory image-goal navigation. A **frozen LingBot-Map** (Geometric Context Transformer)
history encoder + trainable heads (RetrievalHead / RevisitMerge / NovelBranch / current_state) +
a DDPM trajectory decoder. **Core claim: seen ≫ unseen** — if the goal location was already
observed earlier in the episode, memory lets the policy exploit it; a memoryless policy can't.

## Pipeline status

| step | state | result |
|---|---|---|
| ① backbone validation | done | LingBotStream vs official forward **cosine = 1.0** (lossless); weight verified 0-missing, copied into `yuxuan/lingbot-map/weights/` |
| ② data + precompute | done | 40-traj single-scene pilot → **130 traj / 10 scenes**; cache ≈ 1.2 GB/traj |
| ③ training | single-scene done; multi-scene interrupted | standalone trainer (env `enerverse`); single-scene 150 steps; multi-scene stopped at step 94 (script disabled by resource governance) |
| ④ inference chain | done | wrote `memnav_infer.py` + `policy_agent.py` + `memnav_server.py` from scratch; all smoke-passed |
| ⑤ core-contribution eval | done | **seen-vs-unseen AUC = 0.924** |
| ⑤ IsaacSim closed-loop | blocked | headless starts fine; scenes_home renders black (broken MDL materials); no textured cluttered_hard USD |

## Headline result — the core contribution, quantified
On 1600 sampled (current, goal) pairs with the trained model:
- **SEEN goals: mean revisit_gate = 0.78** vs **UNSEEN: 0.25** (separation +0.53)
- **gate as a seen/unseen classifier: AUC = 0.924** (0.5 = no memory, 1.0 = perfect)
- unseen → correctly chose NULL: **99.4%**

The implicit-memory "is this goal in my memory?" signal works. Plot: `seen_unseen_stats.png`.

## Honest limitations
- **Fine localization weak**: the model knows *whether* a goal is in memory (coarse) but not *which
  exact frame* (best real frame within ±5 of the true goal = 31%, above the ~5% chance but weak).
  Partly a metric artifact (adjacent frames have near-identical DINO features → softmax mass
  spreads → NULL wins the argmax), partly genuinely weak.
- **Pilot scale**: 40–130 trajectories, far from a full training run.
- **IsaacSim closed-loop not achieved**: black rendering on scenes_home + weak pilot model.
- **Multi-scene training incomplete** (stopped at step 94; only a step-50 checkpoint, AUC 0.66).

## Where everything lives
- **Inference chain**: `memnav_infer.py`, `policy_agent.py`, `memnav_server.py` (this dir)
- **Trained checkpoint**: `checkpoints/memnav_pilot.ckpt` (single-scene step-150, trainable heads only)
- **Results**: `seen_unseen_stats.png`, `nav_demo.png`
- **Eval scripts**: `eval_seen_unseen_stats.py`, `eval_retrieval_localization.py`
- **Runbook**: `PILOT_RUN.md` (setup / precompute / train / infer / IsaacSim eval commands)
- **Environment**: `enerverse` (torch 2.4 + diffusers); LingBot weight in `yuxuan/lingbot-map/weights/lingbot-map-long.pt`

## Next steps (open)
1. Resume multi-scene training (from step-50, with checkpoint-resume + fault tolerance) when GPU frees up.
2. Fix IsaacSim rendering (MDL material search path) or obtain textured cluttered_hard USDs for a real closed-loop nav demo.
3. Improve fine retrieval (more data/scenes, or a top-k / windowed retrieval formulation).
