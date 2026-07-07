"""Stage 3/3 of the revisit-definition sweep: turn revisit_sweep_eval.py results into
the plots/tables that pin down the operational definition of "revisit".

Success (per sample) := pos_err_m < --pos_thr AND rot_err_deg < --rot_thr.

Outputs (PNG + summary.md into --out_dir):
  1. envelope_heatmaps.png  — success rate / median pos err / median rot err over the
       (requested forward distance x heading offset) grid: the insertion envelope.
       If success tracks columns (dyaw) more than rows (fwd), heading is the binding
       constraint; if it tracks covisibility bins better than either, content is.
  2. covis_curves.png       — success rate vs GT covisibility (goal-in-anchor), split
       by |dyaw|; plus pos/rot err scatter vs covisibility.
  3. observables.png        — success vs DINO sim-at-anchor (the deployable gate
       signal); retrieval top-1 accuracy over the grid; negatives-vs-positives DINO
       top-1 similarity distributions with suggested gate threshold (max-F1).
  4. summary.md             — headline numbers + per-cell table.

Run (any env with pandas/matplotlib):
  python revisit_sweep_analyze.py \
      --results /home/asus/Research/Nav/memnav_viz/revisit_sweep/results.parquet \
      --out_dir /home/asus/Research/Nav/memnav_viz/revisit_sweep
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def heat(ax, piv, title, fmt="{:.2f}", cmap="viridis", vmin=None, vmax=None):
    im = ax.imshow(piv.values.astype(float), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)), [f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), [f"{i:g}" for i in piv.index])
    ax.set_xlabel("heading offset |dyaw| (deg)")
    ax.set_ylabel("forward dist (m)")
    ax.set_title(title)
    for (r, c), v in np.ndenumerate(piv.values.astype(float)):
        if np.isfinite(v):
            ax.text(c, r, fmt.format(v), ha="center", va="center", fontsize=8,
                    color="white" if (vmax or 1) and v < ((vmax or 1) * 0.6) else "black")
    plt.colorbar(im, ax=ax, fraction=0.046)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pos_thr", type=float, default=0.5,
                    help="metric fallback threshold, used only where direction is undefined")
    ap.add_argument("--rot_thr", type=float, default=30.0)
    ap.add_argument("--dir_thr", type=float, default=30.0,
                    help="translation-direction error threshold (scale-free primary)")
    ap.add_argument("--max_local_rmse", type=float, default=0.3,
                    help="drop anchors whose local similarity fit is worse than this (m)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.results)
    n_traj = df.traj.nunique()
    n_bad = int((df.local_rmse > args.max_local_rmse).sum())
    bad = sorted(df[df.local_rmse > args.max_local_rmse]
                 .groupby(["traj", "anchor_idx"]).size().index.tolist())
    if n_bad:
        print(f"[warn] dropping {n_bad} rows from {len(bad)} anchors with "
              f"local_rmse > {args.max_local_rmse}m: {bad}")
        df = df[df.local_rmse <= args.max_local_rmse]
    # SCALE-FREE success: LingBot is scale-ambiguous, so metric position error
    # conflates scale with relocalization. Primary criterion = heading + translation
    # DIRECTION. Metric position is only the fallback where direction is undefined
    # (zero-displacement goals, or a collapsed near-zero prediction — a real failure
    # that the metric fallback correctly rejects).
    df["ok"] = (df.yaw_err_deg < args.rot_thr) & np.where(
        df.t_dir_err_deg.notna(), df.t_dir_err_deg < args.dir_thr,
        df.rel_pos_err_m < args.pos_thr)
    grid = df[df.kind == "grid"].copy()
    neg = df[df.kind == "neg"].copy()
    ctrl = df[df.kind == "control"].copy()
    grid["adyaw"] = grid.req_dyaw.abs()
    # pooled positives: grid + covis-stratified random goals; heading band from the
    # ACTUAL offset so both kinds join the covisibility analysis
    pool = df[df.kind.isin(["grid", "rand"])].copy()
    pool["adyaw_act"] = pool.dyaw_deg.abs()

    # ---------------- 1. insertion envelope over (fwd x |dyaw|) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    piv_s = grid.pivot_table(index="req_fwd", columns="adyaw", values="ok", aggfunc="mean")
    piv_p = grid.pivot_table(index="req_fwd", columns="adyaw", values="t_dir_err_deg", aggfunc="median")
    piv_r = grid.pivot_table(index="req_fwd", columns="adyaw", values="yaw_err_deg", aggfunc="median")
    heat(axes[0], piv_s, f"success rate (pos<{args.pos_thr}m & rot<{args.rot_thr}deg)",
         vmin=0, vmax=1)
    heat(axes[1], piv_p, "median translation direction err (deg)", fmt="{:.0f}", cmap="magma_r")
    heat(axes[2], piv_r, "median yaw err (deg)", fmt="{:.0f}", cmap="magma_r")
    fig.suptitle("LingBot goal-insertion relocalization envelope (grid samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "envelope_heatmaps.png"), dpi=140)

    # ---------------- 2. covisibility as the predictor ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    bins = np.array([0, .02, .05, .1, .2, .3, .5, .7, 1.0])
    pool["covbin"] = pd.cut(pool.covis_goal_in_anchor, bins, include_lowest=True)
    for a_lo, a_hi, cl in [(0, 45, "C0"), (45, 91, "C1"), (91, 181, "C3")]:
        sub = pool[(pool.adyaw_act >= a_lo) & (pool.adyaw_act < a_hi)]
        if not len(sub):
            continue
        sr = sub.groupby("covbin", observed=False).ok.mean()
        cx = [(i.left + i.right) / 2 for i in sr.index]
        axes[0].plot(cx, sr.values, "-o", color=cl, label=f"|dyaw| in [{a_lo},{a_hi})")
    axes[0].set_xlabel("GT covisibility goal-in-anchor")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].set_title("success vs covisibility, by heading offset")
    sc = axes[1].scatter(pool.covis_goal_in_anchor, pool.rel_pos_err_m.clip(0, 5),
                         c=pool.adyaw_act, cmap="coolwarm", s=8, alpha=0.6)
    axes[1].set_xlabel("covisibility goal-in-anchor")
    axes[1].set_ylabel("rel pos err (m, clipped 5)")
    plt.colorbar(sc, ax=axes[1], label="|dyaw| deg")
    sc = axes[2].scatter(pool.covis_goal_in_anchor, pool.yaw_err_deg,
                         c=pool.fwd.abs(), cmap="viridis", s=8, alpha=0.6)
    axes[2].set_xlabel("covisibility goal-in-anchor")
    axes[2].set_ylabel("yaw err (deg)")
    plt.colorbar(sc, ax=axes[2], label="fwd (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "covis_curves.png"), dpi=140)

    # ---------------- 3. deployable observables (DINO) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    sbins = np.linspace(min(grid.dino_sim_at_anchor.min(), neg.dino_top1_sim.min() if len(neg) else 1) - .01,
                        1.0, 12)
    grid["simbin"] = pd.cut(grid.dino_sim_at_anchor, sbins)
    sr = grid.groupby("simbin", observed=False).ok.mean()
    axes[0].plot([(i.left + i.right) / 2 for i in sr.index], sr.values, "-o")
    axes[0].set_xlabel("DINO CLS cos(goal, anchor)")
    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-.05, 1.05)
    axes[0].set_title("gate signal -> insertion success")
    piv_ret = grid.pivot_table(index="req_fwd", columns="adyaw",
                               values="dino_top1_near_anchor", aggfunc="mean")
    heat(axes[1], piv_ret, "retrieval top-1 within +/-3 of anchor", vmin=0, vmax=1)
    if len(neg):
        axes[2].hist(grid.dino_top1_sim, bins=30, alpha=0.6, density=True,
                     label=f"grid (n={len(grid)})")
        axes[2].hist(neg.dino_top1_sim, bins=30, alpha=0.6, density=True,
                     label=f"neg (n={len(neg)})")
        # max-F1 threshold: positives = successful grid samples
        cand = np.linspace(min(neg.dino_top1_sim.min(), grid.dino_top1_sim.min()),
                           grid.dino_top1_sim.max(), 200)
        pos_sim = grid[grid.ok].dino_top1_sim.values
        best = max(cand, key=lambda t: 2 * (pos_sim >= t).sum() /
                   max(1, (grid.dino_top1_sim >= t).sum() + (neg.dino_top1_sim >= t).sum()
                       + len(pos_sim)))
        axes[2].axvline(best, color="k", ls="--", label=f"thr~{best:.3f}")
        axes[2].set_xlabel("DINO top-1 sim")
        axes[2].legend()
        axes[2].set_title("false-positive separation (negatives)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "observables.png"), dpi=140)

    # ---------------- 4. summary ----------------
    tbl = grid.groupby(["req_fwd", "adyaw"], observed=False).agg(
        n=("ok", "size"), success=("ok", "mean"),
        dir_med=("t_dir_err_deg", "median"), yaw_med=("yaw_err_deg", "median"),
        mag_med=("t_mag_ratio", "median"), pos_med=("rel_pos_err_m", "median"))
    lines = [
        "# Revisit-definition sweep summary", "",
        f"- results: `{args.results}`  ({n_traj} trajectories, "
        f"{len(grid)} grid samples, {len(neg)} negatives; "
        f"dropped {n_bad} rows with bad local fit)",
        f"- success := yaw_err < {args.rot_thr} deg AND translation-direction err < "
        f"{args.dir_thr} deg (SCALE-FREE; metric fallback rel_pos_err < {args.pos_thr} m "
        f"only where direction is undefined). LingBot translation magnitude is "
        f"scale-ambiguous — see the mag ratio column, do not read it as failure.",
        f"- overall grid success: **{grid.ok.mean():.1%}**",
        (f"- self-insertion controls (upper bound): success {ctrl.ok.mean():.1%}, "
         f"rel pos err median {ctrl.rel_pos_err_m.median():.2f} m, "
         f"yaw err median {ctrl.yaw_err_deg.median():.1f} deg (n={len(ctrl)})"
         if len(ctrl) else "- no control rows"),
        f"- calibration: local fit RMSE median {df.local_rmse.median():.3f} m "
        f"(global {df.calib_pos_rmse.median():.3f} m — scale drifts), "
        f"local scale median {df.local_scale.median():.3f}, "
        f"interp: {df.pose_interp.mode().iat[0]}, "
        f"convention: {df.cam_convention.mode().iat[0]}", "",
        "## PRIMARY: success by covisibility x heading (grid + rand pooled)", "",
        "| covis(goal,anchor) | dyaw 0-45 | dyaw 45-95 | dyaw 95-180 |",
        "|---|---|---|---|",
    ]
    cbins = [0, .02, .05, .1, .2, .5, 1.01]
    pool["cb2"] = pd.cut(pool.covis_goal_in_anchor, cbins, include_lowest=True)
    for iv in pool.cb2.cat.categories:
        row = [f"| {iv.left:g}-{iv.right:g}"]
        for a_lo, a_hi in [(0, 45), (45, 95), (95, 181)]:
            c = pool[(pool.cb2 == iv) & (pool.adyaw_act >= a_lo) & (pool.adyaw_act < a_hi)]
            row.append(f" {c.ok.mean():.2f} (n={len(c)})" if len(c) else " -")
        lines.append(" |".join(row) + " |")
    lines += ["",
        "## distance is NOT the variable: grid success by (fwd, covis >= 0.1)", "",
        "| fwd (m) | covisible: success (n) | not covisible: success (n) |",
        "|---|---|---|",
    ]
    for f_, gsub in grid.groupby("req_fwd"):
        seen = gsub[gsub.covis_goal_in_anchor >= 0.1]
        uns = gsub[gsub.covis_goal_in_anchor < 0.1]
        lines.append(f"| {f_:g} | " +
                     (f"{seen.ok.mean():.2f} (n={len(seen)})" if len(seen) else "-") + " | " +
                     (f"{uns.ok.mean():.2f} (n={len(uns)})" if len(uns) else "-") + " |")
    lines += ["",
        "## secondary: per-cell results by (fwd, |dyaw|) — fwd is confounded with"
        " visibility via placement, read the primary table instead", "",
        "| fwd (m) | \\|dyaw\\| | n | success | dir err med (deg) | yaw err med (deg) "
        "| mag ratio med | metric pos med (m) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (f_, a_), r in tbl.iterrows():
        dm = "-" if np.isnan(r.dir_med) else f"{r.dir_med:.1f}"
        mm = "-" if np.isnan(r.mag_med) else f"{r.mag_med:.2f}"
        lines.append(f"| {f_:g} | {a_:g} | {int(r.n)} | {r.success:.2f} "
                     f"| {dm} | {r.yaw_med:.1f} | {mm} | {r.pos_med:.2f} |")
    # data-driven envelope suggestion
    good = tbl[tbl.success >= 0.7]
    if len(good):
        idx = good.index.to_frame(index=False)
        lines += ["", f"## suggested envelope (cells with success >= 0.70)",
                  f"- max forward distance: **{idx.req_fwd.max():g} m**",
                  f"- max heading offset:  **{idx.adyaw.max():g} deg**"]
    cthr = None
    cs = pool.groupby("covbin", observed=False).ok.mean()
    for iv, v in cs.items():
        if v >= 0.7:
            cthr = iv.left
            break
    if cthr is not None:
        lines.append(f"- covisibility threshold where success crosses 0.70: **>= {cthr:g}**")
    with open(os.path.join(args.out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nplots + summary.md -> {args.out_dir}")


if __name__ == "__main__":
    main()
