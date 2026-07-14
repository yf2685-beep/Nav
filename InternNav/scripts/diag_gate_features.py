"""Gate-feature probe: how do revisit vs novel score distributions differ, and
which POOLED statistic best separates them — so we design the BCE gate from data,
not a guess.

Two design facts this checks (both flagged before writing the gate):
  (1) A covis positive is a CONTIGUOUS BAND (an earlier pass co-observes the goal),
      so revisit is a PLATEAU, not a spike. `top1 - top2` ~ 0 for both a plateau
      and a flat novel floor -> peak-sharpness is a bad feature. We measure real
      candidate features instead.
  (2) At the current step the robot is APPROACHING the goal, so recent frames see
      it in BOTH revisit and novel. The revisit/novel signal lives ONLY in the
      pre-approach RETRIEVABLE region (idx >= anchor_margin, before this leg) --
      the loop-closure frames. So we score EXACTLY that eligible region E, the same
      candidate set the retrieval label is drawn from, NOT the full [0..k].

For each covis goal we take raw-CLS cosine over E, standardize within-sample
(z = (s - mean_E)/std_E, to kill the per-scene absolute-cosine offset the first
probe exposed), and compute a menu of candidate gate features. We then rank each
feature by its SAMPLE-LEVEL AUC for the revisit(1) / novel(0) decision, dump the
mean sorted-z curve per group (plateau vs floor, eyeball), and save a PNG + npz.

Run inside the same apptainer overlay as training (see run_gate_probe.sbatch).
"""
import argparse
import os

import numpy as np
import torch

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from internnav.model.basemodel.memnav.memnav_policy import MemNavNet


def _auc(pos, neg):
    if pos.size == 0 or neg.size == 0:
        return np.nan
    d = pos[:, None] - neg[None, :]
    return float((d > 0).mean() + 0.5 * (d == 0).mean())


def _band_max(z, b):
    """max mean-z over any contiguous window of length b (plateau elevation)."""
    if z.size < b:
        return float(z.mean())
    c = np.convolve(z, np.ones(b) / b, mode="valid")
    return float(c.max())


def _features(s):
    """Candidate pooled gate features over the eligible-region score vector s."""
    s = np.sort(s)[::-1]                                   # descending
    mu, sd = float(s.mean()), float(s.std() + 1e-6)
    z = (s - mu) / sd                                     # within-sample standardize
    def tk(a, k):
        return float(a[: min(k, a.size)].mean())
    med = float(np.median(s))
    return {
        # absolute (anisotropy-prone) — kept as baselines
        "top1": float(s[0]),
        "top1_minus_top2": float(s[0] - (s[1] if s.size > 1 else s[0])),
        "top5_mean_minus_median": tk(s, 5) - med,
        # within-sample standardized (recommended family)
        "z_top1": float(z[0]),
        "z_top5_mean": tk(z, 5),
        "z_top10_mean": tk(z, 10),
        "z_band3": _band_max(z, 3),
        "z_band5": _band_max(z, 5),
        "z_band8": _band_max(z, 8),
        "z_frac_gt2": float((z > 2.0).mean()),
        "z_frac_gt1_5": float((z > 1.5).mean()),
    }, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_per_group", type=int, default=300)
    ap.add_argument("--dino_batch", type=int, default=16)
    ap.add_argument("--min_elig", type=int, default=8, help="min eligible frames |E|")
    ap.add_argument("--topk_curve", type=int, default=32)
    ap.add_argument("--out_prefix", default="/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/gate_features")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]; feat = os.environ["MEMNAV_FEATURE_ROOT"]
    repo = os.environ["LINGBOT_REPO"];    wts = os.environ["LINGBOT_WEIGHTS"]
    W = int(os.environ.get("MEMNAV_WINDOW", 32)); NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    MFN = int(os.environ.get("MEMNAV_MAX_FRAME_NUM", 2048))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)

    # eligible region E = pre-approach, retrievable frames [amargin .. leg_start)
    # (covis goals only — that's where the loop-closure revisit/novel decision lives)
    buckets = {"revisit": [], "novel": []}
    for s in ds.samples:
        if not s["has_covis"]:
            continue
        leg = int(s["leg_start"]); am = int(s["amargin"])
        if leg - am < args.min_elig:
            continue
        grp = "novel" if s["null_pos"] else "revisit"
        buckets[grp].append((s["traj_idx"], s["goal_img_path"], am, leg))
    for g in buckets:
        rng.shuffle(buckets[g]); buckets[g] = buckets[g][: args.max_per_group]
    print("[gate] group sizes:", {g: len(v) for g, v in buckets.items()}, "| device:", device)

    net = MemNavNet(
        token_dim=384, heads=8, predict_size=24, temporal_depth=8, num_diffusion_iters=10,
        lingbot_kwargs=dict(lingbot_repo=repo, weights=wts, window=W, num_scale=NS, max_frame_num=MFN),
        device=device,
    ).to(device).eval()

    dino_cache = {}

    def mem_of(ti):
        if ti not in dino_cache:
            dino_cache[ti] = ds._load_dino_cls(ti)
        return dino_cache[ti]

    @torch.no_grad()
    def goal_cls_of(paths):
        out = []
        for i in range(0, len(paths), args.dino_batch):
            imgs = torch.stack([ds._load_image_path(p) for p in paths[i : i + args.dino_batch]]).to(device)
            out.append(net.lingbot.dino(imgs)["cls"].float().cpu().numpy())
        return np.concatenate(out, 0)

    feats = {"revisit": [], "novel": []}
    zcurves = {"revisit": [], "novel": []}
    raw_examples = {"revisit": [], "novel": []}
    for grp, items in buckets.items():
        if not items:
            continue
        gcls = goal_cls_of([it[1] for it in items])
        gcls = gcls / (np.linalg.norm(gcls, axis=1, keepdims=True) + 1e-8)
        for (ti, _p, am, leg), gc in zip(items, gcls):
            mem = mem_of(ti)[am:leg].astype(np.float32)          # eligible region E
            mem = mem / (np.linalg.norm(mem, axis=1, keepdims=True) + 1e-8)
            s = mem @ gc                                          # [|E|]
            fd, z = _features(s)
            feats[grp].append(fd)
            zs = np.sort(z)[::-1][: args.topk_curve]
            zpad = np.full(args.topk_curve, np.nan); zpad[: zs.size] = zs
            zcurves[grp].append(zpad)
            if len(raw_examples[grp]) < 6:
                raw_examples[grp].append(np.sort(s)[::-1][: args.topk_curve])

    # ---- rank candidate features by revisit-vs-novel AUC ----
    keys = list(feats["revisit"][0].keys())
    print("\n============ GATE-FEATURE SEPARATION (revisit=1 vs novel=0) ============")
    print(f"{'feature':<26}{'AUC':>7}{'rev med':>10}{'nov med':>10}{'rev p10':>10}{'nov p90':>10}")
    ranked = []
    for k in keys:
        rv = np.array([f[k] for f in feats["revisit"]], dtype=np.float64)
        nv = np.array([f[k] for f in feats["novel"]], dtype=np.float64)
        auc = _auc(rv, nv)
        ranked.append((auc, k, rv, nv))
    for auc, k, rv, nv in sorted(ranked, key=lambda x: -x[0]):
        print(f"{k:<26}{auc:>7.3f}{np.median(rv):>10.3f}{np.median(nv):>10.3f}"
              f"{np.percentile(rv,10):>10.3f}{np.percentile(nv,90):>10.3f}")
    best_auc, best_k, best_rv, best_nv = max(ranked, key=lambda x: x[0])
    print(f"\nBEST separating feature: {best_k}  (AUC={best_auc:.3f})")
    print("  -> if a BAND/z-feature wins over top1/top1_minus_top2, the plateau intuition holds:")
    print("     gate on 'how far a contiguous band stands above THIS scene's background'.")

    # ---- mean sorted-z curve per group (plateau vs floor) ----
    print("\nmean sorted-z curve (top-{}), revisit vs novel:".format(args.topk_curve))
    mr = np.nanmean(np.stack(zcurves["revisit"]), 0)
    mn = np.nanmean(np.stack(zcurves["novel"]), 0)
    for r in range(0, args.topk_curve, 4):
        print(f"  rank {r:2d}: revisit z={mr[r]:+.2f}   novel z={mn[r]:+.2f}")

    # ---- optional plot ----
    png = args.out_prefix + ".png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        x = np.arange(args.topk_curve)
        ax[0].plot(x, mr, "-o", ms=3, label="revisit"); ax[0].plot(x, mn, "-o", ms=3, label="novel")
        ax[0].set_title("mean sorted-z over eligible E"); ax[0].set_xlabel("rank"); ax[0].set_ylabel("z"); ax[0].legend()
        ax[1].hist(best_rv, bins=30, alpha=0.6, label="revisit"); ax[1].hist(best_nv, bins=30, alpha=0.6, label="novel")
        ax[1].set_title(f"best feature: {best_k} (AUC={best_auc:.2f})"); ax[1].legend()
        for c in raw_examples["revisit"]:
            ax[2].plot(c, "-", color="C0", alpha=0.5)
        for c in raw_examples["novel"]:
            ax[2].plot(c, "-", color="C1", alpha=0.5)
        ax[2].set_title("example raw-cos sorted curves (blue=revisit, orange=novel)"); ax[2].set_xlabel("rank")
        fig.tight_layout(); fig.savefig(png, dpi=110)
        print(f"\nsaved figure -> {png}")
    except Exception as e:
        print(f"\n[plot skipped: {e}]")

    np.savez(args.out_prefix + ".npz",
             **{f"revisit__{k}": np.array([f[k] for f in feats["revisit"]]) for k in keys},
             **{f"novel__{k}": np.array([f[k] for f in feats["novel"]]) for k in keys},
             zcurve_revisit=np.stack(zcurves["revisit"]), zcurve_novel=np.stack(zcurves["novel"]))
    print(f"saved per-sample features -> {args.out_prefix}.npz")


if __name__ == "__main__":
    main()
