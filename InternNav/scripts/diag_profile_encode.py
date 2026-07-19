"""Profile encode_memory throughput to find why MEMNAV_STREAM_GROUP>1 regressed.

Runs one REAL training-sized batch (BATCH_SIZE, default 16) through
``core.encode_memory`` under three configurations and reports wall time + a
per-op breakdown + the history-length padding waste:

  (a) G=1                — original per-sample scalar loop (baseline)
  (b) G=Grp random-chunk — current batched path, consecutive-index chunks
  (c) G=Grp k-SORTED     — same batched path, but the batch is reordered by k
                           so each chunk holds adjacent-depth samples
                           (length bucketing). Padding to max_n_hist per chunk
                           then collapses toward zero.

The padding-waste ratio = sum_over_chunks(max_n_hist * chunk_len) / sum(n_hist).
1.0 = no waste; higher = more attention computed over masked pad columns. This
is the ceiling that bucketing can recover.

Run inside the training apptainer overlay on a GPU (see
scripts/train_memnav/run_profile_encode.sh).
"""
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "diffusion-policy"))
sys.path.insert(0, REPO)

from scripts.train.configs.memnav import memnav_exp_cfg
from internnav.model.basemodel.memnav.memnav_policy import MemNavPolicy, MemNavModelConfig
from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _pad_waste(ks, W, S, G):
    """Analytic padding waste for consecutive-index chunking of the given k order."""
    n_hist = [max(0, (k - W + 1) - S) for k in ks]
    used = sum(n_hist)
    computed = 0
    for s in range(0, len(ks), G):
        chunk = n_hist[s:s + G]
        computed += max(chunk) * len(chunk)
    return computed / max(1, used), n_hist


def _reorder(batch, perm):
    """Return a shallow-copied batch dict with all per-sample fields permuted by `perm`."""
    out = dict(batch)
    for key, val in batch.items():
        if torch.is_tensor(val) and val.shape[0] == len(perm):
            out[key] = val[perm]
        elif isinstance(val, list) and len(val) == len(perm):
            out[key] = [val[i] for i in perm]
    return out


def _timed_encode(core, batch, G, reps):
    """Wall time (median of `reps`) for encode_memory at MEMNAV_STREAM_GROUP=G,
    plus a per-op breakdown captured by monkeypatching the lingbot methods."""
    os.environ["MEMNAV_STREAM_GROUP"] = str(G)
    lb = core.lingbot
    acc = {"window_b": 0.0, "window_s": 0.0, "goal_b": 0.0, "goal_s": 0.0,
           "goal_b_streams": 0, "goal_s_calls": 0, "depth": 0.0}
    orig = {n: getattr(lb, n) for n in
            ["window_forward", "window_forward_batched", "depth_feature",
             "depth_feature_batched", "goal_append_warm", "goal_append_warm_batched"]}

    def wrap(name, key, count_streams=False, count_calls=False):
        f = orig[name]
        def inner(*a, **kw):
            _sync(); t = time.perf_counter()
            r = f(*a, **kw)
            _sync(); acc[key] += time.perf_counter() - t
            if count_streams:
                acc["goal_b_streams"] += len(a[1])           # caches is 2nd positional
            if count_calls:
                acc["goal_s_calls"] += 1
            return r
        return inner

    lb.window_forward = wrap("window_forward", "window_s")
    lb.window_forward_batched = wrap("window_forward_batched", "window_b")
    lb.depth_feature = wrap("depth_feature", "depth")
    lb.depth_feature_batched = wrap("depth_feature_batched", "depth")
    lb.goal_append_warm = wrap("goal_append_warm", "goal_s", count_calls=True)
    lb.goal_append_warm_batched = wrap("goal_append_warm_batched", "goal_b", count_streams=True)

    times = []
    try:
        with torch.no_grad():
            for _ in range(reps):
                for k in acc:
                    acc[k] = 0.0 if isinstance(acc[k], float) else 0
                _sync(); t0 = time.perf_counter()
                core.encode_memory(batch)
                _sync(); times.append(time.perf_counter() - t0)
    finally:
        for n, f in orig.items():
            setattr(lb, n, f)
    times.sort()
    return times[len(times) // 2], acc


def main():
    G = int(os.environ.get("PROFILE_GROUP", "4"))
    BS = int(os.environ.get("BATCH_SIZE", "16"))
    REPS = int(os.environ.get("PROFILE_REPS", "3"))
    cfg = memnav_exp_cfg
    il = cfg.il
    W, S = il.window_size, il.num_scale

    print(f"root={il.root_dir}\nwindow={W} num_scale={S} goal_warm={il.goal_warm} "
          f"G={G} BATCH_SIZE={BS} reps={REPS}")

    model_cfg = MemNavModelConfig(model_cfg=cfg.model_dump())
    model = MemNavPolicy.from_pretrained(pretrained_model_name_or_path="", config=model_cfg)
    model.to(model._device).eval()
    core = model.core

    ds = MemNav_Dataset(
        il.root_dir, predict_size=il.predict_size, image_size=il.image_size,
        lingbot_repo=il.lingbot_repo, feature_root=getattr(il, "feature_root", None),
        window_size=il.window_size, num_scale=il.num_scale,
    )
    print(f"dataset trajectories with cache: {len(ds)}")

    # a real shuffled batch (like training) — spread picks across the dataset so k varies
    torch.manual_seed(int(os.environ.get("PROFILE_SEED", "0")))
    n = len(ds)
    picks = torch.randperm(n)[:BS].tolist()
    items = [ds[i] for i in picks]
    batch = memnav_collate_fn(items)
    ks = [int(x) for x in batch["cur_steps"]]

    # ---- padding-waste analysis (analytic) ----
    waste_rand, n_hist = _pad_waste(ks, W, S, G)
    perm = sorted(range(BS), key=lambda i: ks[i])
    ks_sorted = [ks[i] for i in perm]
    waste_sort, _ = _pad_waste(ks_sorted, W, S, G)
    print(f"\nk (cur_step)   : min={min(ks)} max={max(ks)} "
          f"mean={sum(ks)/BS:.0f}  values={sorted(ks)}")
    print(f"n_hist         : min={min(n_hist)} max={max(n_hist)} "
          f"mean={sum(n_hist)/BS:.0f}")
    print(f"pad-waste ratio: random-chunk={waste_rand:.2f}x   k-sorted={waste_sort:.2f}x "
          f"  (1.0 = no wasted attention over pad columns)")

    def show(tag, wall, acc):
        gb = acc["goal_b_streams"]; gs = acc["goal_s_calls"]
        print(f"\n[{tag}] wall={wall*1000:.0f} ms/step  "
              f"(x{60/wall:.0f}/min)" if wall > 0 else f"\n[{tag}] wall=0")
        print(f"   window: batched={acc['window_b']*1000:.0f}ms scalar={acc['window_s']*1000:.0f}ms | "
              f"depth={acc['depth']*1000:.0f}ms")
        print(f"   goal  : batched={acc['goal_b']*1000:.0f}ms ({gb} streams) "
              f"scalar={acc['goal_s']*1000:.0f}ms ({gs} calls)")

    # warmup (allocator + cudnn autotune) at G=1 then G
    with torch.no_grad():
        os.environ["MEMNAV_STREAM_GROUP"] = "1"; core.encode_memory(batch)
        os.environ["MEMNAV_STREAM_GROUP"] = str(G); core.encode_memory(batch)
    _sync()

    print("\n==================== TIMING (median of reps) ====================")
    w1, a1 = _timed_encode(core, batch, 1, REPS)
    show("G=1  scalar baseline", w1, a1)

    wg, ag = _timed_encode(core, batch, G, REPS)
    show(f"G={G}  random-chunk (current)", wg, ag)

    batch_sorted = _reorder(batch, perm)
    ws, as_ = _timed_encode(core, batch_sorted, G, REPS)
    show(f"G={G}  k-SORTED (bucketed)", ws, as_)

    print("\n==================== SUMMARY ====================")
    print(f"G=1                : {w1*1000:.0f} ms/step  (1.00x)")
    print(f"G={G} random-chunk : {wg*1000:.0f} ms/step  ({w1/wg:.2f}x vs G=1)")
    print(f"G={G} k-sorted     : {ws*1000:.0f} ms/step  ({w1/ws:.2f}x vs G=1, "
          f"{wg/ws:.2f}x vs random-chunk)")
    print("(>1.00x = speedup. k-sorted isolates the gain bucketing would add to training.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
