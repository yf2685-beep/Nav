# Multi-Stop Navigation Benchmark for NavDP — Design Doc

**Status:** proposal for review (no code written yet)
**Goal of this benchmark:** quantify whether NavDP can exploit having *already seen* goal B en route to goal A — i.e., measure the **memory gap** that motivates a memory-augmented next project.

---

## 1. Task definition

A single episode chains two stops:

```
start ──leg A──▶ goal A ──leg B──▶ goal B
```

with the deliberate spatial constraint: **B lies between start and A**, so the robot physically passes near B (and B enters its camera FoV) during leg A. By the time the robot reaches A and is retargeted to B, B is far outside NavDP's 1.2 s visual window and the robot must turn around and go back.

**Goal modality (decided): both A and B are image goals** (RGB renders from a canonical viewpoint). Image-goal is the focus of the next project, so the task is kept uniform. B's image-goal is what makes the memory question meaningful — a memoryless policy must *re-search* for B from scratch; a policy with spatial memory could head straight back.

**Gated, sequential evaluation (decided):** leg A is attempted first. **Only if A succeeds does the robot proceed to B.** If A fails (timeout/stuck before reaching A), the episode ends and is scored as an A-failure with no B leg. The final report is therefore **SR_A** and **SR_B | A** (B success rate *conditioned on* reaching A) — see §5.

---

## 2. Why NavDP is expected to show ~zero memory benefit

NavDP's only state is an **8-frame RGB memory queue** (`baselines/navdp/policy_agent.py`, `memory_size=8`). At `step_dt ≈ 0.15 s` that is ~1.2 s of history. There is no map, no persistent embedding of the scene. Consequences:

- For an **image-goal B**, the policy conditions on `[last 8 frames, current depth, goal image of B]` and servos toward matching the goal image. Whether it drove past B 40 seconds ago is invisible to it.
- Therefore we **predict** `SR_B|A(CH) ≈ SR_B(CO)` and `SPL_B(CH) ≈ SPL_B(CO)` — seeing B on leg A buys nothing.

The benchmark's value is precisely this: it is a testbed where a **memory-equipped** method would show `seen ≫ unseen`, and NavDP establishes the memoryless floor. The headline deliverable is the **memory-benefit delta** (§5), expected ≈ 0 for NavDP.

---

## 3. Experimental conditions

> **Revised after asset investigation (navmesh-free).** No scene ships a navmesh/occupancy map, so we cannot cheaply place a *matched-difficulty off-path* point. Instead the memory control fixes **the same B reached from the same A**, and varies only whether the robot saw B beforehand. This is a *cleaner* isolation of "did seeing B help" and needs no navmesh.

The two primary conditions reach an **identical** B from an **identical** A — the only difference is prior exposure:

| Arm | Description | Tests |
|-----|-------------|-------|
| **CH — Chained / Seen** | One episode: start → A → B, with B placed *on* the start→A path (robot passes B during leg A, ~40–60% along). Memory carries across the A→B switch. | The target case. |
| **CO — Cold / Unseen** | A standard single image-goal episode with **start = A, goal = B**, fresh memory. Same A, same B, but the robot has **never seen B**. | Difficulty-matched control (literally the same A→B leg). |
| **CH-reset — memory ablation** | Identical to CH, but call `navigator_reset_env` at the A→B switch to wipe the 8-frame queue. | Whether even short-term carryover matters at the transition. |

**Why this is rigorous:** CH and CO share the exact same A position, B position, and A→B leg. The *only* manipulated variable is whether leg A (and therefore exposure to B) preceded the B-leg. So the delta isolates exactly the benefit of prior exposure — no navmesh, no difficulty-matching guesswork. The cold arm CO is just a normal `imagegoal` episode (start=A, goal=B), runnable on the **existing** `eval_imagegoal_wheeled.py` with a generated `(start=A, goal=B)` `.npy`.

**Headline metric** = `SR_B|A(CH) − SR_B(CO)` and `SPL_B(CH) − SPL_B(CO)`. A memoryless policy → ≈ 0. A spatial-memory policy → large positive.

**Exposure validation (must-log):** for arm CH, confirm B was genuinely seen on leg A. Each step of leg A, log whether B's world point projects into the robot camera frustum (reuse the projection math in `pixel_projection_data`, `wheeled_task.py:56-70`) and the min robot→B distance. Record `B_seen_frames`, `B_min_dist_legA`. Drop/flag CH episodes where B never actually entered view — otherwise "seen" is mislabeled.

> **Optional future arm (needs navmesh): U — off-path Unseen.** Place B at matched distance from A but off the start→A corridor, testing whether a method can *localize a target it mapped while merely passing through the area* (vs the open scene). Deferred because it requires building an occupancy/navmesh from the USD (see §7). Not needed for the core memory-gap result.

---

## 4. Episode source — B is picked live from NavDP's own leg-A run (decided)

**Decision (5):** NavDP runs leg A, and **SR_A is NavDP's real success rate** (over all base `(start, A)` episodes, including failures). There is **no separate oracle generation pass** — B is chosen *during* the chained run, from NavDP's actual trajectory at the moment it reaches A. This guarantees "seen" by construction (the robot was physically at B) and makes B exist exactly for the episodes the user cares about (A-successes).

**Base episodes:** reuse the shipped `imagegoal_start_goal_pairs.npy` `(start, A, init_yaw)` rows — no new `.npy` needed for the chained arm.

**Per-episode flow (chained arm CH), single pass:**
1. NavDP navigates start → A (its own path). Record the world-frame trajectory `τ` (from `camera_pos` each step, `eval_imagegoal_wheeled.py:194`) and the heading at each point.
2. If A **fails** (timeout/stuck) → `reached_A=False`, episode ends, no B. (Counts in SR_A denominator.)
3. If A **succeeds** (dwell/stop at A, §6) → pick **B = a point on `τ`** at ~40–60% of arc length, with `B_yaw` = robot heading when it passed that point (so B's rendered goal image matches the view NavDP actually had). Constrain: `euclid(A,B) ≥ d_min` (e.g. 2 m) so leg B is a real backtrack, and B not within `arrive_thresh` of start.
4. Place `/Goal` + `goal_cam` at B, render B's goal image, continue A → B with memory retained.
5. **Log the frozen `(start, A, B, B_yaw)`** for every A-success → this list defines the cold arm.

**Cold arm (CO):** after the chained pass, take the frozen `(A, B, B_yaw)` pairs and run them as plain image-goal episodes (`start=A, goal=B`, fresh memory) on the **unmodified** `eval_imagegoal_wheeled.py` via a generated `(M,5)` `cold_imagegoal_start_goal_pairs.npy`. Same A, same B → isolates prior exposure.

**CH-reset arm:** replay the frozen pairs through the chained eval but call `navigator_reset_env` at the A→B switch.

- `find_usd_path` (`utils_tasks/basic_utils.py:28`) keys off the task substring, so the cold file name must contain `imagegoal`.
- **No navmesh, no pre-generation.** Reachability A→B is implicitly validated: B sits on a real start→A trajectory NavDP just drove. SPL is Euclidean (§5, parity with NavDP).
- *Selection note:* B exists only for A-successes — that's intended (SR_B|A is explicitly conditioned on reaching A). SR_A itself is measured over all base episodes.

---

## 5. Metrics

Per episode, record (extends the current `success/spl/distance` dict at `eval_imagegoal_wheeled.py:270-272`):

- `reached_A` (bool) — did leg A succeed. **Gates the rest of the episode.**
- `reached_B` (bool) — only defined when `reached_A`; episodes that fail A do **not** attempt B.
- **`SPL_A`** — oracle = `euclid(start→A)`, actual = leg-A path length. *(Euclidean, parity with existing NavDP SPL at `eval_imagegoal_wheeled.py:271` — no navmesh; see §7.)*
- **`SPL_B`** — leg-B only, over the conditioned set (`reached_A` episodes): oracle = `euclid(A→B)`, actual = path length accumulated *after* the switch. This is the cleanest memory-sensitive number.
- `overshoot_B` — max distance past B before arrival / min approach distance, captures "drove past it because it didn't recognize it."
- `excess_ratio_B` = `path_legB / euclid(A→B)`.
- Covariates: `B_seen_frames`, `B_min_dist_legA`, `euclid(A→B)`, `arm`.

**Primary report (the two headline numbers you asked for):**
- **`SR_A`** = fraction of episodes reaching A.
- **`SR_B | A`** = among A-successes only, fraction also reaching B.

Aggregate deliverables:
- Table: arm × {SR_B|A (CH) or SR_B (CO), SPL_B, excess_ratio_B}. `SR_A` is reported once for the chained arm (CO has start=A, so leg A doesn't apply).
- **Headline memory benefit:** `ΔSR_B = SR_B|A(CH) − SR_B(CO)`, `ΔSPL_B = SPL_B(CH) − SPL_B(CO)`. Memoryless NavDP → ≈ 0.
- CH vs CH-reset: isolates short-term-memory (8-frame) contribution at the transition.

> Note: the existing eval scores success at **1.5 m** (`:268`) while the sim `arrive_goal` DoneTerm fires at **1.0 m + velocity<0.5** (`wheeled_task.py:96`). For the chained arm we **disable the `arrive_goal` DoneTerm** and handle both arrivals manually in the eval loop (§6) — A-arrival = `dist<1.0 & velocity<0.5` (**dwell/stop required, decision 4**) to trigger the switch; B-arrival scored at `1.5 m` for parity with prior NavDP numbers.

---

## 6. Implementation plan (no model/server changes)

All logic lives in a new `eval_multistop_wheeled.py`, cloned from `eval_imagegoal_wheeled.py`. The sim, server, MPC, and planning thread are unchanged.

**Termination change:** in the multistop env config, **drop the `arrive_goal` DoneTerm** (keep `time_out` + `stuck`). All arrival logic is handled in the eval loop so A-arrival never auto-ends the episode. This removes the fragile "switch /Goal before the DoneTerm fires" ordering dependency.

**Per-env state machine** added to the main loop (`eval_imagegoal_wheeled.py:187-279`):

1. On reset, place robot at start, `/Goal` + `goal_cam` at **A** (`init_yaw`, as `wheeled_task.py:251-260`). Phase = `LEG_A`. Each step append `camera_pos` and heading to `τ[i]`.
2. **A-arrival (dwell required, decision 4):** when `dist(robot, A) < 1.0 & |velocity| < 0.5` for the dwell — i.e. NavDP brought the robot to a *stop* at A — set `reached_A=True`; pick B from `τ[i]` per §4 step 3; reposition `/Goal` to B (`set_world_poses`, `:259-260`) and `goal_cam` to B with `B_yaw` (`:252-256`); log frozen `(start,A,B,B_yaw)`; phase = `LEG_B`; start `legB_path_len`. *(No `navigator_reset_env` — keep memory — except CH-reset.)*
   - NavDP's own stop-in-place behavior (value < `stop_threshold`) naturally produces the dwell as it converges on A; if it lingers without stopping, a max-dwell-wait cap forces the switch and flags it.
3. **A-failure:** if `time_out`/`stuck` fires while phase==`LEG_A` → `reached_A=False`, no B leg; record, reset to next episode.
4. **B-arrival / B-failure:** in phase `LEG_B`, score `reached_B = dist(robot,B) < 1.5` on `time_out`/`stuck` or on a manual B-arrival check; record metrics, reset.

**Key extension points (file:line):**
- Prim retargeting: `wheeled_task.py:251-260` (pattern to copy for the in-loop switch).
- Relative-goal vector auto-updates from `/Goal`: `oracle_imu_pose_data`, `wheeled_task.py:49-53` (no code change — moving the prim is enough).
- Goal-image render source: `goal_cam` prim + `goal_image` obs (`eval_imagegoal_wheeled.py:190`).
- Memory reset hook (CH-reset only): `navigator_reset(env_id=i, port)` → `/navigator_reset_env` → `policy_agent.reset_env(i)`.
- Path-length integrator: `eval_imagegoal_wheeled.py:258` (split into leg-A + post-switch leg-B accumulators).
- Metrics writer: `:270-273` (extend dict; `write_metrics`, `basic_utils.py:39`).

**Tooling:** the chained eval `eval_multistop_wheeled.py` *emits* the frozen `(M,5)` cold `.npy` as a side-output (no separate generation script). A small `make_cold_npy.py` just collates the per-episode logs into the cold file.

**Scene config:** `internscenes_home` requires `--scene_scale 0.01` (per CLAUDE.md); B is placed/queried entirely in **world coords** (`camera_pos`, `set_world_poses`), so the scale only needs to be passed to the eval, not applied to B math. Verify on the first scene.

---

## 7. Risks / open dependencies

- **~~Navmesh/occupancy & geodesic SPL~~ — RESOLVED by investigation.** No scene ships a navmesh/occupancy map, and **NavDP's own SPL is already Euclidean** (`eval_imagegoal_wheeled.py:271`). We therefore use Euclidean SPL (parity, not a compromise) and place B on real rollout trajectories (§4) — no navmesh needed. The optional off-path arm U (§3) is the *only* thing that would require building one (deferred).
- **Difficulty matching CH vs CO — RESOLVED by construction.** CH and CO reach the *same* B from the *same* A, so the A→B leg is identical; no distance-matching guesswork.
- **`cluttered_easy` USD is missing locally** (only `.npy` present). Start the spike on `cluttered_hard/hard_0` or an `internscenes_home` scene (both have USD), or re-download `cluttered-0.usd` from Scene-N1. *Immediate practical blocker for that one category only.*
- **A→B reachability.** B is on the *same* start→A trajectory NavDP just drove, so a navigable A→B route provably exists (NavDP traversed B→A); reverse traversal is highly likely even in multi-room internscenes. No separate reachability filter needed.
- **Leg-A failures contaminating leg-B stats.** Now that A is an image goal, leg-A may fail more; mitigated by **gating** (leg-B metrics conditioned on `reached_A`) and by the cold arm CO, which sidesteps leg A entirely for the B measurement.
- **NavDP stop behavior at the switch.** When `value < stop_threshold` the agent zeros the trajectory (`policy_agent.py`). At the instant of A→B switch the new goal image may briefly produce low value → momentary stall. Acceptable, but log it; not a memory effect.
- **"Seen" actually seen.** Enforced by the exposure check (§3); don't trust geometry alone.
- **B exists only for A-successes (by design).** Since NavDP runs leg A and B is picked from its trajectory, B is undefined for A-failures. This is intended: `SR_B|A` is conditioned on reaching A, and `SR_A` is reported separately over all episodes. Consequence to keep in mind: the cold arm CO is therefore evaluated only on the A-success subset, so `ΔSR_B = SR_B|A(CH) − SR_B(CO)` compares like-for-like on the same `(A,B)` pairs.
- **`internscenes_home` scale.** Pass `--scene_scale 0.01`. B math stays in world coords so scale shouldn't leak in, but verify the rendered B goal image and the relative-goal vector look sane on scene 1 before scaling out.

---

## 8. Decisions — all locked

1. **Modality of A** — A and B both image goals; gated SR_A → SR_B|A reporting. ✓
2. **SPL** — Euclidean (parity with NavDP; no navmesh ships). ✓
3. **Memory control** — navmesh-free Cold (CO) arm (§3); off-path U deferred as optional. ✓
4. **Dwell at A** — **require a stop** (`dist<1.0 & |vel|<0.5`) before releasing to B. ✓
5. **Leg-A controller** — **NavDP itself**; SR_A is NavDP's real number; B picked live from NavDP's trajectory (§4). ✓
6. **Scenes** — **`internscenes_home`, 20 episodes/scene** (`--scene_scale 0.01`). ~18 scenes → ~360 chained episodes. ✓

---

## 9. Suggested build order (once decisions are settled)

1. `eval_multistop_wheeled.py` — clone `eval_imagegoal_wheeled.py`; drop `arrive_goal` DoneTerm; add the LEG_A→LEG_B state machine (dwell-at-A detect, live B-pick from `τ`, prim retarget). Run on **one** `internscenes_home` scene; eyeball the A→B switch in the fps video (does the goal image flip to B, does the robot turn back?).
2. Add metrics + logging: SR_A, SR_B|A, SPL_A/B, overshoot_B; emit frozen `(start,A,B,B_yaw)` per A-success.
3. `make_cold_npy.py` → collate frozen pairs into `cold_imagegoal_start_goal_pairs.npy`; run the **CO arm** on the unmodified `eval_imagegoal_wheeled.py`.
4. Add the CH-reset flag (call `navigator_reset_env` at switch).
5. Scale to all ~18 `internscenes_home` scenes × 20 episodes; produce the arm × metric table and the headline `ΔSR_B` / `ΔSPL_B`.
