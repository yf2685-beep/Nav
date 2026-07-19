"""Split goal_append_warm's cost into IO (image load+decode) vs GPU compute.

At G=1 the ONLY live image loads inside encode_memory come from goal_append_warm
(``lingbot_stream.py:543`` — window frames are already prefetched in the batch).
So wrapping ``load_images`` isolates goal-warm IO exactly, and the rest of
goal_append_warm's wall time is GPU compute (the warm-recompute forwards + the
goal stream).

Reports, over a real BS batch at G=1:
  goal IO      = sum(load_images wall)              [hideable by worker prefetch]
  goal compute = sum(goal_append_warm wall) - IO   [the transformer forwards]
  window       = sum(window_forward wall)          [already prefetched → all compute]
plus per-frame IO cost and frames loaded, to size the prefetch win.

Run inside the training apptainer overlay on a GPU.
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


def main():
    BS = int(os.environ.get("BATCH_SIZE", "16"))
    REPS = int(os.environ.get("PROFILE_REPS", "2"))
    cfg = memnav_exp_cfg
    il = cfg.il
    os.environ["MEMNAV_STREAM_GROUP"] = "1"   # real training path

    print(f"window={il.window_size} num_scale={il.num_scale} goal_warm={il.goal_warm} "
          f"BATCH_SIZE={BS} reps={REPS} (G=1 path)")

    model_cfg = MemNavModelConfig(model_cfg=cfg.model_dump())
    model = MemNavPolicy.from_pretrained(pretrained_model_name_or_path="", config=model_cfg)
    model.to(model._device).eval()
    core = model.core
    lb = core.lingbot

    ds = MemNav_Dataset(
        il.root_dir, predict_size=il.predict_size, image_size=il.image_size,
        lingbot_repo=il.lingbot_repo, feature_root=getattr(il, "feature_root", None),
        window_size=il.window_size, num_scale=il.num_scale,
    )
    torch.manual_seed(int(os.environ.get("PROFILE_SEED", "0")))
    picks = torch.randperm(len(ds))[:BS].tolist()
    batch = memnav_collate_fn([ds[i] for i in picks])
    ks = [int(x) for x in batch["cur_steps"]]
    print(f"dataset={len(ds)}  k: min={min(ks)} max={max(ks)} mean={sum(ks)/BS:.0f}")

    acc = {"io": 0.0, "io_frames": 0, "io_calls": 0, "goal_wall": 0.0, "window_wall": 0.0}
    o_load = lb.load_images
    o_goal = lb.goal_append_warm
    o_win = lb.window_forward

    def load_images(paths, *a, **kw):
        t = time.perf_counter()            # IO+decode is CPU/synchronous — no cuda sync needed
        r = o_load(paths, *a, **kw)
        acc["io"] += time.perf_counter() - t
        acc["io_frames"] += len(paths); acc["io_calls"] += 1
        return r

    def goal_append_warm(*a, **kw):
        _sync(); t = time.perf_counter()
        r = o_goal(*a, **kw)
        _sync(); acc["goal_wall"] += time.perf_counter() - t
        return r

    def window_forward(*a, **kw):
        _sync(); t = time.perf_counter()
        r = o_win(*a, **kw)
        _sync(); acc["window_wall"] += time.perf_counter() - t
        return r

    lb.load_images = load_images
    lb.goal_append_warm = goal_append_warm
    lb.window_forward = window_forward

    with torch.no_grad():
        core.encode_memory(batch)          # warmup
    _sync()

    rows = []
    with torch.no_grad():
        for _ in range(REPS):
            for k in acc:
                acc[k] = 0 if isinstance(acc[k], int) else 0.0
            _sync(); t0 = time.perf_counter()
            core.encode_memory(batch)
            _sync(); wall = time.perf_counter() - t0
            rows.append((wall, dict(acc)))

    rows.sort(key=lambda r: r[0])
    wall, a = rows[len(rows) // 2]         # median rep
    goal_io = a["io"]
    goal_compute = a["goal_wall"] - a["io"]
    window = a["window_wall"]

    print("\n==================== GOAL-WARM IO vs COMPUTE (median rep) ====================")
    print(f"encode_memory wall : {wall*1000:8.0f} ms")
    print(f"  goal IO (decode) : {goal_io*1000:8.0f} ms  ({100*goal_io/wall:4.1f}%)  "
          f"{a['io_frames']} frames over {a['io_calls']} calls  "
          f"-> {1000*goal_io/max(1,a['io_frames']):.2f} ms/frame")
    print(f"  goal compute     : {goal_compute*1000:8.0f} ms  ({100*goal_compute/wall:4.1f}%)")
    print(f"  window (compute) : {window*1000:8.0f} ms  ({100*window/wall:4.1f}%)")
    other = wall - a["goal_wall"] - window
    print(f"  other (pose/etc) : {other*1000:8.0f} ms  ({100*other/wall:4.1f}%)")
    print("\nInterpretation: 'goal IO' is what worker-prefetch could overlap with GPU.")
    print(f"If fully hidden, floor ~= {(wall-goal_io)*1000:.0f} ms  "
          f"({wall/(wall-goal_io):.2f}x speedup on encode_memory).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
