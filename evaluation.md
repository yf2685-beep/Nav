
## Scene Categories vs. Path Length for Point-Goal Navigation

| Scene | Path lengths | Format | Notes |
|---|---|---|---|
| `cluttered_easy/hard` | Short (single room) | Single tarball | Quick smoke tests |
| `internscenes_home` | Medium (multi-room) | HF tree, many files | Realistic homes |
| `internscenes_commercial` | Long (large buildings) | Single tarball | Best for long-horizon |

## Evaluation Metrics

| Metric | Full name | Definition (this codebase) | Range | Higher/Lower better |
|---|---|---|---|---|
| `success` | Success | 1 if final distance to goal < 1.5 m, else 0 | `{0, 1}` | higher |
| `SPL` | Success-weighted Path Length | `success × min(optimal_dist / actual_dist, 1)` | `[0, 1]` | higher |
| `NE` | Navigation Error | Euclidean distance from final robot position to goal (meters) | `≥ 0` | lower |
| `LE` | Last/Localization Error | Robot's belief vs. truth at termination — diverges when implicit localization drifts | `≥ 0` | lower |

## Result on cluttered_hard

The result is based on evaluation on hard_7 (one of 10 cluttered_hard scenes) with 20 episodes.  

### Results on official ckpt
```bash
/home/asus/Research/Nav/NavDP/startgoal_logoplanner_cluttered_hard_OFFICIAL/hard_7
```

| Metric | Value |
|---|---|
| Success rate | 23.8% (5/21) |
| Mean SPL | 0.227 |
| Mean NE | 7.82 m |

### Result on retrained ckpt
| Scene  | N  | SR    | SPL   | NE mean | NE med | LE mean | dist mean |
|--------|----|-------|-------|---------|--------|---------|-----------|
| hard_7 | 21 | 0.0 % | 0.0 % | 17.29   | 19.35  | 24.94   | 19.16     |
## Results on Internscenes Home

### Result on retrained ckpt
| Scene        | N  | SR     | SPL    | NE mean | NE med | LE mean | dist mean |
|--------------|----|--------|--------|---------|--------|---------|-----------|
|  (scene 0)   | 21 | 14.3 % | 12.1 % | 4.10    | 4.62   | 6.41    | 6.73      |
|  (scene 4)   | 21 | 4.8 %  | 3.5 %  | 3.77    | 3.67   | 4.54    | 5.69      |
| Combined     | 42 | 9.5 %  | 7.8 %  | 3.93    | 3.72   | 5.47    | 6.21      |

### Result on Official ckpt
| Scene                     | N  | SR     | SPL    | NE mean | NE med | LE mean | dist mean |
|---------------------------|----|--------|--------|---------|--------|---------|-----------|
| MVUCSQAKTKJ5...ABA8_usd   | 21 | 52.4 % | 52.3 % | 2.87    | 0.91   | 0.44    | 6.73      |
| MVUCSQAKTKJ5...ABY8_usd   | 21 | 19.0 % | 16.1 % | 3.37    | 3.68   | 0.96    | 5.69      |
| Combined                  | 42 | 35.7 % | 34.2 % | 3.12    | 3.10   | 0.70    | 6.21      |

### Comparison of official and retrained checkpoints

| Model | Split | N | SR | SPL | NE_mean | LE_mean |
|---|---|---:|---:|---:|---:|---:|
| Official | BA8 | 21 | 52.4% | 52.3% | 2.87 | 0.44 |
| Official | BY8 | 21 | 19.0% | 16.1% | 3.37 | 0.96 |
| Official | COMB | 42 | 35.7% | 34.2% | 3.12 | 0.70 |
| Retrain (Diffusion init) | BA8 | 21 | 14.3% | 12.1% | 4.10 | 6.41 |
| Retrain (Diffusion init)| BY8 | 21 | 4.8% | 3.5% | 3.77 | 4.54 |
| Retrain (Diffusion init)| COMB | 42 | 9.5% | 7.8% | 3.93 | 5.47 |
| Critic2 | BA8 | 21 | 9.5% | 8.1% | 4.58 | 6.09 |
| Critic2 | BY8 | 21 | 4.8% | 4.0% | 4.16 | 5.01 |
| Critic2 | COMB | 42 | 7.1% | 6.0% | 4.37 | 5.55 |
| Critic2_ng0 | BA8 | 21 | 23.8% | 20.9% | 3.77 | 5.69 |
| Critic2_ng0 | BY8 | 21 | 4.8% | 3.9% | 4.16 | 3.84 |
| Critic2_ng0 | COMB | 42 | 14.3% | 12.4% | 3.96 | 4.76 |
