
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


