"""Break ONE aggregator streaming forward into rope / SDPA / linear / kv-cache /
other, to decide whether torch.compile / a rope kernel is worth it.

The memnav hot path is ~96 sequential single-frame ``_aggregate_features`` calls
(32 window + up to 64 goal-warm) per sample. Each call, per depth-block, does:
  qkv-Linear -> q/k norm -> rope(apply_rotary_emb) -> kv-cache cat/clone/evict
  -> SDPA over (specials + window KV) -> proj-Linear -> MLP.

We wrap ``apply_rotary_emb`` and ``F.scaled_dot_product_attention`` in
record_function scopes and profile a real deep sample's window_forward (32
forwards) with torch.profiler, then bucket CUDA time:
  ROPE  = apply_rotary_emb scope
  SDPA  = scaled_dot_product_attention scope
  LINEAR= addmm/mm/bmm/linear (qkv, proj, MLP)   [torch.compile won't shrink these]
  KV    = cat/clone/copy_ (cache growth+eviction) [candidate for the real overhead]
  other = norms/elementwise/launch

Interpretation: if LINEAR+SDPA dominate, the forward is already GEMM-bound and
compile/rope work buys little. If ROPE or KV or 'other' (launch overhead) is a
big slice, compile / CUDA-graph / a rope kernel could pay off.

Run inside the training apptainer overlay on a GPU.
"""
import os
import sys

import torch
from torch.profiler import profile, record_function, ProfilerActivity

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "diffusion-policy"))
sys.path.insert(0, REPO)

from scripts.train.configs.memnav import memnav_exp_cfg
from internnav.model.basemodel.memnav.memnav_policy import MemNavPolicy, MemNavModelConfig
from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn
import lingbot_map.layers.attention as attn_mod
import lingbot_map.layers.rope as rope_mod


def main():
    cfg = memnav_exp_cfg
    il = cfg.il
    os.environ["MEMNAV_STREAM_GROUP"] = "1"

    model_cfg = MemNavModelConfig(model_cfg=cfg.model_dump())
    model = MemNavPolicy.from_pretrained(pretrained_model_name_or_path="", config=model_cfg)
    model.to(model._device).eval()
    core = model.core
    lb = core.lingbot
    dev = model._device

    ds = MemNav_Dataset(
        il.root_dir, predict_size=il.predict_size, image_size=il.image_size,
        lingbot_repo=il.lingbot_repo, feature_root=getattr(il, "feature_root", None),
        window_size=il.window_size, num_scale=il.num_scale,
    )
    # pick a DEEP sample (long history -> realistic KV) by scanning a handful
    torch.manual_seed(0)
    cand = torch.randperm(len(ds))[:24].tolist()
    items = [ds[i] for i in cand]
    batch = memnav_collate_fn(items)
    ks = [int(x) for x in batch["cur_steps"]]
    bsel = max(range(len(ks)), key=lambda i: ks[i])
    k = ks[bsel]
    print(f"profiling window_forward on deep sample: k={k} n_hist={(k-il.window_size+1)-il.num_scale} "
          f"(W={il.window_size} forwards)")

    cache = core._load_cache(batch["cache_paths"][bsel], batch["rgb_dirs"][bsel])
    wimg = batch["batch_window_images"][bsel].to(dev)   # [W,3,H,W]

    # ---- record_function markers around rope + SDPA ----
    _orig_rope = rope_mod.apply_rotary_emb
    def rope_marked(*a, **kw):
        with record_function("ROPE_apply_rotary_emb"):
            return _orig_rope(*a, **kw)
    rope_mod.apply_rotary_emb = rope_marked
    attn_mod.apply_rotary_emb = rope_marked   # attention.py imported it by name

    _orig_sdpa = torch.nn.functional.scaled_dot_product_attention
    def sdpa_marked(*a, **kw):
        with record_function("SDPA_scaled_dot_product_attention"):
            return _orig_sdpa(*a, **kw)
    torch.nn.functional.scaled_dot_product_attention = sdpa_marked
    attn_mod.F.scaled_dot_product_attention = sdpa_marked

    def run_window():
        return lb.window_forward(cache, wimg, k, return_multilayer=False)

    with torch.no_grad():
        for _ in range(2):      # warmup (allocator + autotune)
            run_window()
        torch.cuda.synchronize()

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=False) as prof:
            with record_function("WINDOW_FORWARD"):
                run_window()
            torch.cuda.synchronize()

    torch.nn.functional.scaled_dot_product_attention = _orig_sdpa
    rope_mod.apply_rotary_emb = _orig_rope

    ka = prof.key_averages()
    def cuda_us(e):
        return getattr(e, "self_cuda_time_total", getattr(e, "self_device_time_total", 0))
    def cuda_tot(e):
        return getattr(e, "cuda_time_total", getattr(e, "device_time_total", 0))

    total_self = sum(cuda_us(e) for e in ka)
    rope_us = sum(cuda_tot(e) for e in ka if e.key == "ROPE_apply_rotary_emb")
    sdpa_us = sum(cuda_tot(e) for e in ka if e.key == "SDPA_scaled_dot_product_attention")

    LIN = ("addmm", "mm", "bmm", "linear", "matmul")
    KVO = ("cat", "clone", "copy_", "slice", "narrow", "index")
    lin_us = sum(cuda_us(e) for e in ka if any(t in e.key.lower() for t in LIN))
    kv_us = sum(cuda_us(e) for e in ka if any(t in e.key.lower() for t in KVO))
    other_us = max(0, total_self - lin_us - kv_us
                   - sum(cuda_us(e) for e in ka
                         if e.key in ("ROPE_apply_rotary_emb", "SDPA_scaled_dot_product_attention")))

    def pct(x):
        return f"{x/1e3:8.2f} ms  ({100*x/max(1,total_self):4.1f}%)"

    print("\n==================== AGGREGATOR FORWARD CUDA BREAKDOWN ====================")
    print(f"total self-CUDA over {il.window_size} forwards: {total_self/1e3:.2f} ms "
          f"(~{total_self/1e3/il.window_size:.3f} ms/forward)")
    print(f"  ROPE   (apply_rotary_emb) : {pct(rope_us)}")
    print(f"  SDPA   (attention)        : {pct(sdpa_us)}")
    print(f"  LINEAR (qkv/proj/mlp)     : {pct(lin_us)}")
    print(f"  KV     (cat/clone/evict)  : {pct(kv_us)}")
    print(f"  other  (norm/elem/launch) : {pct(other_us)}")

    # CPU-vs-CUDA: if CPU wall >> CUDA, the forward is launch/overhead-bound
    total_cpu = sum(getattr(e, "self_cpu_time_total", 0) for e in ka)
    print(f"\ntotal self-CPU: {total_cpu/1e3:.2f} ms  (CPU>CUDA => launch/Python-bound "
          f"=> compile/CUDA-graph helps)")

    print("\n---- top 15 ops by self-CUDA ----")
    for e in sorted(ka, key=cuda_us, reverse=True)[:15]:
        print(f"  {e.key[:48]:48s} {cuda_us(e)/1e3:8.2f} ms  x{e.count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
