"""Numerical-equivalence check: batched window_forward (MEMNAV_STREAM_GROUP=G)
vs. the original one-at-a-time loop (G=1).

Builds the real MemNavPolicy + MemNav_Dataset (env-configured, same as training),
collates G samples, and runs ``core.encode_memory`` twice — once with
MEMNAV_STREAM_GROUP=1 (reference, scalar per-sample path) and once with
MEMNAV_STREAM_GROUP=G (batched streams on the batch dim). The window path
(``current``, ``depth_feat``) must match to bf16 tolerance; the goal/pose path is
unbatched in both, so ``cur_pose``/``goal_pose`` must match near-exactly.

Run inside the training apptainer overlay (frames) on a GPU. See
scripts/train_memnav/run_batched_window_check.sbatch.
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "diffusion-policy"))
sys.path.insert(0, REPO)

from scripts.train.configs.memnav import memnav_exp_cfg
from internnav.model.basemodel.memnav.memnav_policy import MemNavPolicy, MemNavModelConfig
from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn


def _stats(name, a, b):
    a = a.float(); b = b.float()
    diff = (a - b).abs()
    # relative Frobenius norm — robust to single-element bf16 outliers, unlike
    # max_abs/max_val. This is the pass metric.
    relF = ((a - b).norm() / a.norm().clamp_min(1e-6)).item()
    # per-sample cosine over the flattened feature
    af = a.reshape(a.shape[0], -1); bf = b.reshape(b.shape[0], -1)
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=1)
    print(f"  {name:12s} shape={tuple(a.shape)} "
          f"max_abs={diff.max().item():.4e} mean_abs={diff.mean().item():.4e} "
          f"relF={relF:.4e} cos[min/mean]={cos.min().item():.6f}/{cos.mean().item():.6f}")
    return cos.min().item(), relF


def main():
    G = int(os.environ.get("CHECK_GROUP", "4"))
    torch.manual_seed(0)
    cfg = memnav_exp_cfg
    il = cfg.il

    print(f"root={il.root_dir}\nfeat={getattr(il,'feature_root',None)}")
    print(f"window={il.window_size} num_scale={il.num_scale} goal_warm={il.goal_warm} G={G}")

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

    # take G samples spread across the dataset to vary k / history length
    n = len(ds)
    picks = [int(i * n / G) for i in range(G)]
    items = [ds[i] for i in picks]
    batch = memnav_collate_fn(items)
    ks = [int(x) for x in batch["cur_steps"]]
    print(f"picked idx={picks} cur_steps(k)={ks} "
          f"n_hist={[max(0,(k-il.window_size+1)-il.num_scale) for k in ks]}")

    # count batched-goal invocations so we can confirm the goal path is exercised
    _orig_goal_batched = core.lingbot.goal_append_warm_batched
    _n_goal_batched = {"calls": 0, "streams": 0}
    def _counting_goal(goal_imgs, caches, ms, *a, **kw):
        _n_goal_batched["calls"] += 1
        _n_goal_batched["streams"] += len(caches)
        return _orig_goal_batched(goal_imgs, caches, ms, *a, **kw)
    core.lingbot.goal_append_warm_batched = _counting_goal

    with torch.no_grad():
        os.environ["MEMNAV_STREAM_GROUP"] = "1"
        ref = core.encode_memory(batch)
        os.environ["MEMNAV_STREAM_GROUP"] = str(G)
        test = core.encode_memory(batch)

    # report the goal warm-length groups (from the anchors the model actually used)
    lo = il.num_scale + il.window_size - 1
    anc = ref["anchor_idx"].tolist()
    Ls = []
    for b, a in enumerate(anc):
        m = min(max(a, lo), ks[b] - 1)
        Ls.append(m - max(il.num_scale, m - il.goal_warm + 1) + 1)
    print(f"goal anchors m-clamped -> warm-lengths L={Ls}")
    print(f"batched-goal: calls={_n_goal_batched['calls']} streams={_n_goal_batched['streams']} "
          f"(0 => goal path fell back to scalar singletons; window path still tested)")

    print("\n=== batched (G) vs scalar (1) ===")
    worst_cos, worst_relF = 1.0, 0.0
    for key in ["current", "depth_feat", "cur_pose", "goal_pose"]:
        c, r = _stats(key, ref[key], test[key])
        worst_cos = min(worst_cos, c); worst_relF = max(worst_relF, r)
    # anchors must be identical (integer)
    same_anchor = torch.equal(ref["anchor_idx"], test["anchor_idx"])
    print(f"\nanchor_idx identical: {same_anchor}")
    print(f"WORST cos={worst_cos:.6f}  WORST relF={worst_relF:.4e}")
    ok = worst_cos > 0.999 and worst_relF < 2e-2 and same_anchor
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
