"""t-sweep: does the unified revisit candidate rule E(k)=[anchor_margin .. k-t]
retain enough positives, or does it over-suppress real loop closures?

For the unified design, a frame is a revisit candidate only if it is >= t frames
behind the current step k (anything nearer is 'approach' -> novel branch). We must
check that genuine B/C revisit goals still keep >=1 positive across their sampled
k-range for a candidate t; if most old matches sit only ~30-80 frames behind k,
t=83 would silently turn revisits into novels.

Pure covis/index arithmetic (no DINO, no GPU): for every covis goal, over ALL
k in [k_lo..k_hi], count for each t whether pos survives in [amargin..k-t].
Also reports, per revisit goal, the temporal gap (k - last_positive_index) so we
see how far behind the old match actually is.
"""
import argparse
import os

import numpy as np

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="0,32,68,83,103")
    ap.add_argument("--max_goals", type=int, default=100000)
    args = ap.parse_args()
    ts = [int(x) for x in args.ts.split(",")]

    root = os.environ["MEMNAV_ROOT_DIR"]; feat = os.environ["MEMNAV_FEATURE_ROOT"]
    repo = os.environ["LINGBOT_REPO"]
    W = int(os.environ.get("MEMNAV_WINDOW", 32)); NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)

    # per-t, per-k tallies over covis-REVISIT goals (static null_pos False)
    # rev_goal := a covis goal that has >=1 pos in the FULL pre-approach region.
    surv = {t: [0, 0] for t in ts}           # [k-with-pos, k-total] pooled over revisit goals
    goal_frac = {t: [] for t in ts}          # per-goal fraction of k that keep a pos
    gaps = []                                 # (k - last_pos_idx) at the MIDDLE k, revisit goals
    n_rev = n_covis = 0

    for s in ds.samples:
        if not s["has_covis"]:
            continue
        n_covis += 1
        if s["null_pos"]:                     # novel covis goal: no pos ever -> skip survival
            continue
        n_rev += 1
        if n_rev > args.max_goals:
            break
        am = int(s["amargin"]); leg = int(s["leg_start"])
        pos_pre = np.asarray(s["pos_pre"], dtype=bool)     # length leg, over [0..leg)
        pos_idx = np.where(pos_pre)[0]                      # positive frame indices (< leg, >= am)
        klo, khi = int(s["k_lo"]), int(s["k_hi"])
        ks = np.arange(klo, khi + 1)
        if ks.size == 0:
            continue
        # gap at middle k
        kmid = int(ks[ks.size // 2])
        gaps.append(kmid - int(pos_idx.max()))             # frames from current back to nearest old pos
        pmin = int(pos_idx.min())                          # earliest positive (all pos_idx >= am)
        for t in ts:
            # a positive survives at k  <=>  exists pos in [am .. k-t]  <=>  k >= pmin + t
            keep = int(np.count_nonzero(ks >= pmin + t))
            surv[t][0] += keep; surv[t][1] += ks.size
            goal_frac[t].append(keep / ks.size)

    print(f"\n[t-sweep] covis goals={n_covis}  revisit goals used={min(n_rev, args.max_goals)}  W={W} NS={NS}")
    gaps = np.array(gaps)
    if gaps.size:
        print(f"[gap] k - nearest_old_pos (at mid-k): median={np.median(gaps):.0f} "
              f"p25={np.percentile(gaps,25):.0f} p75={np.percentile(gaps,75):.0f} "
              f"p90={np.percentile(gaps,90):.0f}  (how far behind the old match sits)")

    print(f"\n{'t':>5}{'k-with-pos%':>14}{'goals w/ pos-at-any-k%':>26}{'goal frac med':>16}")
    for t in ts:
        kw = 100.0 * surv[t][0] / max(surv[t][1], 1)
        gf = np.array(goal_frac[t])
        any_pos = 100.0 * float((gf > 0).mean()) if gf.size else 0.0
        med = float(np.median(gf)) if gf.size else 0.0
        print(f"{t:>5}{kw:>13.1f}%{any_pos:>25.1f}%{med:>16.2f}")

    print("\nRead: 'k-with-pos%' = across all sampled k of revisit goals, fraction that "
          "still have >=1 candidate positive after excluding the last t frames. "
          "'goals w/ pos-at-any-k%' = fraction of revisit goals that keep at least SOME "
          "trainable k. If t=83 tanks these vs t=32/68, the old matches sit too close "
          "behind k -> drop t (use ~68) or make it covis-adaptive.")


if __name__ == "__main__":
    main()
