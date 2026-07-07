
## Which data to use 

|                     | **MP3D (Matterport3D)** | **InternScenes** |
|---------------------|-------------------------|------------------|
| **Origin** | 90 real houses, 3D-scanned (photoreal, but scan holes/blur) | Synthetic furnished rooms (clean geometry, game-asset look) |
| **Format** | `.glb` + `.navmesh` (Habitat-native) | `.usd` (IsaacSim-native) |
| **Simulator** | Habitat | IsaacSim |
| **Planner ready?** | ✅ navmesh ships, `ShortestPathFollower` works | ❌ no navmesh, no planner |
| **Availability** | gated (waiting) | already on disk |
| **Relation to our data** | our `vln_n1` training data **IS** MP3D | our **eval benchmark IS** InternScenes |


# How to generate GT trajectory

### Measured InternData-N1 controller parameters based on 109 InternData-N1 trajectories
| Parameter | Value | How measured |
|-----------|-------|--------------|
| **Speed** | **0.0376 m/frame** (≈1.13 m/s @30 fps) | median step displacement |
| **Min turning radius** | ~0.35–0.5 m | p1–p5 of `R = Δs/Δθ` (tightest sustained turns) |
| Typical curvature radius | ~2.0–2.3 m | median `R` (gentle cruising curves) |
| **Effective lookahead** | ~0.66 m (range 0.34–1.28) | arc-distance until the path bends 15° |
| Max turn rate | ~3.5–4°/frame | p99 |

InternData-N1 motion fingerprint:
- Constant speed ~0.035 m/frame (median 0.035, max 0.049 very tight)
- Bounded turn rate ~1-4°/frame (median 1.1°, max 3.8°)
- Zero turn-in-place (0/1094 frames) 
- the robot always moves forward while turning

### what motion model, planner, can controller do we choose for GT trajectory
Motion model: what moves are physically possible
- nearly always unicycle / differential-drive model— state (x, y, yaw), commands (v, ω), no sideways slide.
- The constant-v, bounded-ω, no-turn-in-place pattern — N1 fingerprint — is a widespread convention. 

Planner: 

Controller: 
- pure-pursuit-style controller


### Sampling B and C

**Episode structure**

- 2-leg: `start → A → B`
- 3-leg: `start → A → B → C`

`A` is always the fresh outbound goal: a random navigable point that fixes the episode's floor (`floor_y = A.y`) and has clearance ≥ `r_min` (0.4 m). Every other waypoint is constrained to A's floor.

**Procedure (per goal B, and again for C)**

1. **Anchor frame.** Pick a random leg-A frame `X` with index ≥ `--anchor_margin` (15 = `num_scale + window − 1`), the earliest index with a valid match window behind it.
2. **Position.** Sample uniformly in a disk of radius `--goal_jitter_pos` (1.5 m) around `X`'s position; snap to the navmesh; reject on clearance and floor. For C, additionally require geodesic distance from B ≥ `c_min`.
3. **Heading.** Sample within a ± `--head_max_deg` (45°) cone around `X`'s heading.
4. **Covisibility gate.** Accept only if covis ∈ [0.2, 1.0] (max over history) **and** heading offset ≤ 45° relative to the covisibility-matched frame.

Note that `X` is the *sampling* anchor, not necessarily the best match — the position jitter means some other history frame may be more covisible with the placed goal. The gate is therefore evaluated against the best-covisibility frame over history, not against `X`.

### Labeling the matching frame

A goal has a *set* of acceptable matching frames, not a single one, so we don't store a hard match label. Instead, each goal's metadata records:

- `covis_curve` — covisibility of the goal against every history frame;
- `covis_argmax` — the index of the best-matching frame.

Positive vs. negative is then derived at train time by thresholding `covis_curve`: frames above the positive threshold form the positive set, frames below the negative threshold are negatives, and the band between is ignored. Storing the full curve rather than a binary label keeps the threshold a tunable knob and lets the same data support the contrastive loss (one sampled positive, masked negatives) without regeneration.