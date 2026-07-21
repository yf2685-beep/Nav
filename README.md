# MemNav — memory-augmented visual navigation

Phase-2 line of work: a navigation policy that keeps an implicit memory of everywhere it
has been, decides whether the current goal is somewhere it has *seen before*, and — if so —
retrieves that frame and relocalizes against it instead of exploring from scratch.

This branch (`memnav`) = `upstream/main` + local run-time conveniences + the action-label
rotation-frame fix. `main` mirrors upstream and is never committed to; see
[GIT_WORKFLOW.md](GIT_WORKFLOW.md).

---

## Architecture in one paragraph

A frozen LingBot-Map backbone streams the episode and maintains a KV cache. Every observed
frame contributes a DINOv2 CLS vector to a growing memory. Given a goal image, a
**RetrievalHead** scores it against every candidate frame in `E(k) = [39 .. k-83]` (the
recent-approach window is excluded, which is what makes "have I been here" separable from
"am I looking at it right now"), producing `ret_logits [B, L]` plus a scalar **revisit gate**
`sigmoid(gate_a · max_cos + gate_b)`. A DDPM decoder (10 steps, 24 waypoints) is conditioned
on 17 memory tokens — `1 time + 8 current-state + 4 revisit + 4 novel` — with the gate
mixing the revisit and novel branches via an additive log-mask.

Retrieval and gating are **decoupled on purpose**: an earlier design used a joint softmax
with a null slot and collapsed to always-null (the easy shortcut). InfoNCE now ranks the
true co-visible frame among candidates, and a separate BCE trains the gate.

---

## Results

### The memory works

Trained bs4 on 2-leg MP3D revisit data (736 trajectories, 1700 steps). Gate separation
(`P(revisit|seen) − P(revisit|unseen)`), 100-step windowed means over revisit-bearing
batches:

| steps | seen | unseen | **sep** | action loss |
|---|---|---|---|---|
| 0–99 | 0.442 | 0.382 | +0.060 | 0.326 |
| 400–499 | 0.464 | 0.296 | +0.168 | 0.129 |
| 900–999 | 0.481 | 0.267 | +0.214 | 0.101 |
| 1400–1499 | 0.464 | 0.196 | +0.268 | 0.090 |
| **1600–1699** | 0.486 | 0.180 | **+0.305** | 0.088 |

Still rising at the end of training — no plateau.

A longer 8-node run (effective batch 32, 30 epochs) reproduces the trend independently:
`gate_sep` +0.207 at epoch 4.94 → +0.288 at epoch 7.53, again driven mostly by *unseen*
falling (0.30 → 0.19). Only the gate/retrieval half of that run is valid — it trains on the
uncorrected labels described below, so any navigation number from its checkpoints is
meaningless.

### Offline gate AUC

All 1296 goal-samples (490 seen / 806 unseen), identical sample set across checkpoints
(verified by hashing the label array):

| checkpoint | **ROC AUC** | seen / unseen | acc@0.5 | retrieval top-1 |
|---|---|---|---|---|
| 500 | 0.703 | 0.611 / 0.492 | 0.587 | 0.618 |
| 1000 | 0.816 | 0.534 / 0.320 | 0.745 | 0.704 |
| **1500** | **0.874** | 0.557 / 0.285 | 0.797 | 0.720 |
| 1700 | 0.862 | 0.473 / 0.195 | 0.776 | **0.771** |

AUC peaks at 1500 and dips slightly by 1700 (+0.023/100 steps → +0.012 → −0.006), so 1500
is the operating point. Two independent signals rise together — AUC and retrieval top-1 —
which a shortcut would not produce.

**Distribution separation** matters more than the means. Up to step 1000 the seen 25th
percentile sat *below* the unseen 75th (overlapping); from 1500 that flips:

| | seen p25 | unseen p75 | |
|---|---|---|---|
| step 1000 | 0.457 | 0.472 | overlapping |
| step 1500 | 0.497 | 0.439 | **separated** |
| step 1700 | 0.360 | 0.321 | **separated** |

Mechanistically the gain comes almost entirely from pushing *unseen* down (0.49 → 0.20)
rather than pulling *seen* up (0.61 → 0.47).

### Closed-loop Habitat (revisit protocol)

Three-leg protocol: drive past a future goal, continue until it clears `exclude_recent` and
lands inside `E(k)`, then hand control to the policy to return. Each revisit goal is paired
with a distance-matched novel goal provably off the traversed path.

| metric | value |
|---|---|
| `match_hit_revisit` | **0.90 – 0.95** |
| `c_in_ek_rate` | 1.00 |
| gate revisit / novel | 0.53 / 0.23 |
| `SR_revisit` / `SR_novel` | 0.0 / 0.0 |

Retrieval finds the correct frame ~9 times in 10 *in closed loop*, and the goal is always in
the candidate set. The success rate is nevertheless zero — see below.

### Why SR was 0: the action labels had no forward channel

Not a tuning problem, and not a failure of the memory. `MemNavData/generate_twoleg.py:361`
applies the Y-up→Z-up matrix to the camera **rotation** as a plain left-multiply instead of
the conjugation `M_W @ R @ M_W.T`. A left-multiply re-expresses the world basis but leaves
the *local axis ordering* in Habitat's `(right, up, forward)`, so column 1 of every stored
rotation is the world vertical axis. `relative_pose` then resolves displacement onto those
columns and `xyz_to_xyt` keeps only channels (0, 1) — making the label
**`[vertical, lateral, theta]`, with forward discarded.**

Measured on the real parquets, 12 trajectories across 12 scenes: label `ch0` std =
**0.000000** while the robots travelled 6–14 m; the largest `|ch0|` anywhere is
**0.19999993 m**, exactly the navmesh `agent_max_climb` — pure stair jitter. The discarded
channel correlates **+1.000** with the dominant-motion body axis.

Consequences, all observed: the policy's predicted heading is pinned at ≈90° at *every*
step regardless of where the goal is; `d_goal` rises monotonically; both `MEMNAV_SWAP_XY`
settings fail for complementary reasons (`=1` feeds lateral in as forward → creeps and never
steers; `=0` feeds the identically-zero channel in → spins in place). The surviving
channel's sign is unlearnable (42% positive) because deleting the perpendicular component
strips any visual meaning from the remaining coordinate's sign.

It hid for months because **a constant-zero channel is trivial to fit** — action loss fell
smoothly 0.33 → 0.088 and looked like textbook convergence.

### The fix

Load-time, not a data regeneration: the parquets store the full 4×4, so the pose was
mis-projected rather than lost. `_fix_stored_rotation` applies the missing `@ M_W.T` via a
`MemNav_Dataset.process_data_parquet` override — deliberately not in `NavDP_Base_Datset`,
which navdp/logoplanner share with data from other generators.

A/B through the real dataset class, toggled by `MEMNAV_LEGACY_ROT_FRAME`:

| | LEGACY (old) | FIXED |
|---|---|---|
| ch0 std | 0.0000 | 0.3096 |
| ch0 frac nonzero | 0.000 | 0.597 |
| ch0 frac positive | 0.000 | **0.875** |
| theta frac nonzero | 0.045 | 0.597 |
| 2-D endpoint | 0.853 m | **1.796 m** |

1.796 m matches the independently measured true horizontal displacement (1.548 m mean).
**ch1 (lateral) is byte-identical across both modes** — the change restores the missing
channel and disturbs nothing that already worked. Theta was collateral damage too: the old
4.5% nonzero values were all ±4π artifacts, since collinear points on a single surviving
axis have no meaningful bearing.

Upstream PR: [glbreeze/Nav#1](https://github.com/glbreeze/Nav/pull/1).

---

## What these numbers do and do not show

- **No held-out split.** This pipeline trains on all 736 trajectories and there is no
  validation set, so the AUC table is a training-fit diagnostic, not generalization.
- **`is_revisit` is a heuristic label**, thresholded from `covis_curve`
  (`covis_pos_hi/lo`, `exclude_recent=83`). The AUC measures agreement with that label, not
  with ground-truth "have I been here".
- **The closed-loop numbers are n=2 episodes on one scene.** `match_hit` is robust at that
  size; the gate figures are not — a single-episode `gate_sep` ranged 0.13–0.30.
- **Every checkpoint above was trained on the broken labels.** The gate/retrieval results
  stand (they never touch the action labels), but nothing about navigation quality can be
  concluded until a model is retrained on corrected labels.

---

## Running

Training and evaluation both run on machine 186 out of a versioned checkout — see
`186:/home/nyuair/memnav_src/RUN_HERE.md`. The entry point refuses to start from a
non-git tree, so every result is attributable to a commit.

    NAME=my_run BATCH_SIZE=4 MAX_STEPS=1700 SAVE_STEPS=500 bash train_memnav_src.sh

---

## The clean baseline configuration

Every component upstream-original, nothing forked or hand-patched — established so that any
later difference is attributable to a deliberate change rather than to drift.

| element | what | where |
|---|---|---|
| **code** | `glbreeze/Nav` **main**, not a fork | `186:/home/nyuair/memnav_src` (branch `memnav` = main + fix; `git checkout main` for pristine upstream) |
| **data** | `mp3d_revisit_v0`, pt1 squashfs, 121 GB | `mp3d_revisit_v0/mp3d_revisit_v0_pt1.sqf` (present on both 127 and 186) |
| ↳ extracted | 2-leg only, 26 GB | `186:/home/nyuair/mp3d_pt1_extract/mp3d_revisit_v0/vln_n1/traj_data` |
| **cache** | upstream `precompute_lingbot_features.py`, with `dino_cls` | `186:/home/nyuair/memnav/cache/vln_n1/traj_data` (166 GB) |
| **VRAM** | batch 2–4 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, no OOM | A6000 48 GB; peak ~23 GB at bs4 |

### Dataset accounting — read this before quoting a number

The pt1 squashfs holds **1944 episodes**, listed directly from its directory table:

| split | episodes | status |
|---|---|---|
| `mp3d_2leg` | **736** | extracted, precomputed, **this is what every result above was trained on** |
| `mp3d_3leg` | 1208 | in the archive only — not extracted, not precomputed, never trained on |

So the working set is 736 trajectories over 50 scenes → **1296 goal-samples**, which is the
`Num examples` HuggingFace reports. Any claim of "the full ~1900 episodes" refers to the
archive's contents, not to what has been trained: **3-leg has never been used.** That matters
because the documented reference run reached `gate_sep` +0.37 on 2leg+3leg at batch 8, while
2-leg-only tops out around +0.31 — the gap is plausibly the missing 3-leg data, and closing
it is the obvious next experiment.

The cache stores visual tensors only — `dino_cls`, `anchor_k/v`, `scale_k/v`, `cam_k/v`,
`cam_pose_enc`. It contains **no action labels**, which is why the rotation-frame fix needed
no cache regeneration.

### Other paths

| what | where |
|---|---|
| LingBot repo + weights | `186:/home/nyuair/yuxuan/lingbot-map`, `weights/lingbot-map-long.pt` |
| checkpoints (bs4 2leg) | `186:/home/nyuair/memnav/checkpoints_keep/memnav_mp3d_2leg_bs4/checkpoint-{500,1000,1500,1700}` |
| Habitat closed-loop eval | `186:/home/nyuair/habitat_eval/` — `env_186.sh`, `run_server.sh`, `run_client.sh`, 61 MP3D scenes |
| offline gate AUC probe | `dgx:/media/cvpr/yuxuan/memnav_cluster/eval_gate_stats.py`, results in `eval_out/gate_*.json` |
| multi-node cluster tree | `dgx:/media/cvpr/yuxuan/memnav_cluster/{code,cache,frames,lingbot-map}`; ARM env `/media/cvpr/yuxuan/envs/enerverse_arm` |
| full cache on dgx | `dgx:/media/cvpr/yuxuan_memnav/caches_pt1/mp3d_2leg/` (579 GB, 50 scenes). Note `caches_glbreeze/` is **empty** despite older notes describing it as the full cache. |

---

## Multi-node training (8 × GB10)

The dgx SLURM cluster (`spark` partition) has ~25 NVIDIA GB10 Grace-Blackwell nodes, one
GPU each with ~128 GB **unified** memory. That much memory means batch size is never the
constraint — bs8 fits trivially where the A6000 needs care. `train.py` reads
`RANK`/`LOCAL_RANK`/`WORLD_SIZE` and wraps in DDP natively, so multi-node needs no code
change, only a launcher: `InternNav/scripts/train_memnav/train_memnav_mp3d.sbatch`.

### The recipe

**8 nodes × 4 per device = effective batch 32.** Launch via `srun` + `python -m
torch.distributed.run` (`--nnodes=$SLURM_NNODES --nproc_per_node=1
--node_rank=$SLURM_NODEID`). The cluster is ethernet-only, so `NCCL_SOCKET_IFNAME=enP7s7`
and `NCCL_IB_DISABLE=1` are required; the resulting `ibvwrap` NCCL warning is harmless.

GB10 is `sm_121` Blackwell on aarch64, so it needs an ARM env with CUDA 12.8+
(`/media/cvpr/yuxuan/envs/enerverse_arm`, torch 2.13+cu130). The x86 `enerverse` will not
run there. Do **not** use the env's own `torchrun` — its shebang points at a dead build
path.

### Measured scaling — more nodes does *not* help

Throughput at effective batch 32 unless noted:

| config | median min/step | samples/min |
|---|---|---|
| 8 nodes × 4/dev = b32 | 2.79 – 2.88 | 11.1 – 11.5 |
| 8 nodes × 4/dev = b32 (slow run) | 6.62 | 4.8 |
| 16 nodes × 2/dev = b32 | 3.71 | 8.6 |
| 16 nodes × 4/dev = b64 | 6.99 | 9.2 |

**8 nodes beats 16, at both batch 32 and 64.** Doubling nodes *and* batch predicted 2×
throughput and delivered 0.83× of the 8-node figure. Against single-GPU bs4, 8 nodes gives
about **3.5×**, not 8× — well short of linear. Default to 8 × 4/dev and do not spend time
scaling out.

Two jobs with **identical config differed 2.3×** (6.62 vs 2.88 min/step); 2 of 3 runs
landed in the fast regime. The cause was never identified — so measure the actual step rate
of each run rather than predicting it from the config.

### Bottleneck hypotheses ruled out by measurement

Recorded so they are not re-investigated:

- **CIFS sequential bandwidth** — 112–115 MB/s per node, uniform. A 239 MB cache file reads
  in ~2 s against a 170 s step.
- **Small-file latency on frames** — 64 JPEGs in 0.39 s.
- **"Some nodes are slower"** — disproved, speeds uniform; fast and slow runs both showed
  96% GPU util.
- **Staging the cache to node-local NVMe** — pointless, I/O is not the bottleneck.
- **NCCL all-reduce** — 224 MB gradient at 4 nodes = 2867 ms/iter (~78 MB/s on ethernet).
  Seconds per step, not minutes; does not explain a 7-minute step on its own.

Note that 96% GPU utilisation *includes NCCL kernels*, so it does not prove compute-bound.
`py-spy` on a live rank is the unfinished next step.

### What scaling buys, and what it does not

**It buys throughput only.** At comparable sample counts, bs32 mean `gate_sep` (+0.109) is
statistically the same as bs4 (+0.100); bs32 is merely more consistently positive (92% vs
76% of recent batches), because a larger batch is less likely to contain zero revisit rows.
Scaling out does **not** improve gate supervision — do not expect a bigger cluster to fix a
learning problem.

### Traps worth knowing

- **HuggingFace prints the wrong batch size.** It reports `Total train batch size = 64` when
  the real value is 32, because it mis-detects `n_gpu=2`. `MemNavTrainer` overrides
  `get_train_dataloader` with its own `DistributedSampler(num_replicas=world)`, so the real
  effective batch is `per_device × world_size`. Verify via
  `epoch = step × eff_batch / 1296`.
- **Resuming across a batch-size change permanently corrupts HF's `epoch` field** — it
  re-derives from the new dataloader length. Compute real epochs by hand, per phase.
- **`max_steps` must be recomputed when the batch changes.** 30 epochs = 38 880 samples:
  1215 steps at b32, but only 708 at b64.
- **CIFS cannot** preserve times (`rsync -a` warns; use `-rlD --no-perms --no-times
  --omit-dir-times`), do pip's atomic install (`pip install --target` + PYTHONPATH), or
  create symlinks (make real directories). Compute nodes see `/media/cvpr` but **not** the
  login node's `/home/cvpr`, so code, data and env must all live under `/media/cvpr`.
- **Reading `gate_sep` wrongly is easy.** Batches logging `no revisit rows this batch` have
  `seen=0.00` by construction, so their `sep = -unseen` is structurally negative and
  meaningless — exclude them before averaging. And never sample every-Nth log line to judge
  a trend; that silently cherry-picks. Correct form:

      grep 'sep=' LOG | grep -v 'no revisit rows' | grep -oE 'sep=[-+][0-9.]+'

The closed-loop Habitat eval lives at `186:/home/nyuair/habitat_eval/` (61 MP3D scenes,
py3.9 habitat-sim client + py3.10 torch policy server over HTTP — they cannot share an env,
and initializing CUDA before EGL breaks rendering, so the split is required, not stylistic).

**Known cost:** ~24 s per policy step on an A6000, dominated by replaying 64 stored frames
through the GCT on every retrieval query. A statistically useful run (50+ episodes) is not
feasible without optimizing that. Upstream's `MEMNAV_STREAM_GROUP` batching is the obvious
first thing to try.

## Layout

| path | what |
|---|---|
| `InternNav/internnav/model/basemodel/memnav/` | policy (`memnav_policy.py`), streaming backbone (`lingbot_stream.py`) |
| `InternNav/internnav/dataset/memnav_dataset_lerobot.py` | dataset + label construction (the rotation fix lives here) |
| `InternNav/internnav/trainer/memnav_trainer.py` | losses + the gate/retrieval diagnostics |
| `MemNavData/generate_twoleg.py` | episode generation (`pursuit_track`, `make_sim`) |
| `NavDP/baselines/memnav/` | first-generation inference scaffolding + pilot notes |
