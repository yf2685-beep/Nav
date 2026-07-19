"""Diagnostic: is the frozen-LingBot pose scale a PER-TRAJECTORY constant that
varies across trajectories, or a GLOBAL constant?

This decides how to fix the stuck aux-pose loss (relative goal pose):

  * If s_traj varies a lot across trajectories  -> the metric (x,y) target is
    NOT recoverable from the frozen poses by any fixed head (the per-episode
    scale is unobservable from values already in LingBot units). Switch the aux
    target to SCALE-INVARIANT bearing + relative yaw.
  * If s_traj is ~constant across trajectories  -> a single global scale absorbs
    it; metric (x,y) is recoverable and the real bug is the head architecture /
    loss (relative-geometry input, theta definition, Huber+geodesic split).

Two probes
----------
Probe B (headline, clean):  per-trajectory Umeyama SIMILARITY alignment of
    LingBot camera centers -> GT metric camera centers over several REAL frames.
    Umeyama returns (scale s_traj, rotation R, residual). Small normalized
    residual => LingBot poses are a similarity transform of GT (metric-up-to-
    scale), so the ONLY missing quantity is the per-trajectory scalar s_traj.
    The ACROSS-trajectory spread (CV) of s_traj is THE decision number. This
    probe is free of the goal-image relocalization and the 2D/3D-plane issue
    (full 3D camera centers on both sides).

Probe A (the real aux signal): ||goal_pose[:3] - cur_pose[:3]|| (LingBot, incl.
    the goal_append relocalization the head actually sees) vs
    ||goal_rel_pose[:2]|| (GT metric, the exact target magnitude). Shows the
    regressable floor including relocalization noise.

Run (inside the same apptainer overlay training uses):
    sbatch scripts/train_memnav/run_pose_scale_probe.sbatch
or in an existing GPU alloc:
    python scripts/diag_pose_scale.py --n_traj 40 --n_samples 256
"""
import argparse
import json
import os

import numpy as np
import torch

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn
from internnav.model.basemodel.memnav.memnav_policy import MemNavNet


# --------------------------------------------------------------------------- #
def umeyama(X, Y):
    """Least-squares similarity Y ~= s R X + t  (Umeyama 1991).
    X, Y: [n, 3]. Returns (s, R[3,3], t[3], resid_rmse)."""
    n = X.shape[0]
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    Sigma = (Yc.T @ Xc) / n
    U, Dvec, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_x = (Xc ** 2).sum() / n
    s = float(np.trace(np.diag(Dvec) @ S) / max(var_x, 1e-12))
    t = my - s * R @ mx
    Yhat = (s * (R @ X.T)).T + t
    resid = float(np.sqrt(((Y - Yhat) ** 2).sum(1).mean()))
    return s, R, t, resid


def poses_to_centers(pe, to_extri):
    """pose_enc [N,9] (absT_quaR_FoV, camera-FROM-world) -> camera centers [N,3] (numpy).
    LingBot's T is the world->cam translation, so the world-frame center is C = -R^T T."""
    E = to_extri(pe[None], image_size_hw=None,
                 pose_encoding_type="absT_quaR_FoV", build_intrinsics=False)[0][0]  # [N,3,4]
    R = E[:, :3, :3]
    T = E[:, :3, 3]
    C = -torch.einsum("nij,nj->ni", R.transpose(1, 2), T)   # -R^T T
    return C.float().cpu().numpy()


def corr_report(name, lbv, gtv):
    """Log-log Pearson r + GT/LB ratio for one relative-distance component."""
    lbv = np.asarray(lbv, dtype=np.float64)
    gtv = np.asarray(gtv, dtype=np.float64)
    m = (lbv > 1e-6) & (gtv > 1e-6) & np.isfinite(lbv) & np.isfinite(gtv)
    r = float(np.corrcoef(np.log(lbv[m]), np.log(gtv[m]))[0, 1]) if m.sum() > 4 else float("nan")
    ratio = gtv[m] / lbv[m]
    print(f"  [{name}] n={int(m.sum())} r(log)={r:.3f}  ratio GT/LB {fmt(stats(ratio))}")
    print(f"        LB dist {fmt(stats(lbv[m]))}")
    print(f"        GT dist {fmt(stats(gtv[m]))}")
    return dict(name=name, n=int(m.sum()), r_log=r, ratio=stats(ratio),
                lb=stats(lbv[m]), gt=stats(gtv[m]))


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()), cv=float(a.std() / (abs(a.mean()) + 1e-12)),
                p10=float(np.percentile(a, 10)), p90=float(np.percentile(a, 90)),
                min=float(a.min()), max=float(a.max()))


def fmt(d):
    if d.get("n", 0) == 0:
        return "(no data)"
    return (f"n={d['n']} mean={d['mean']:.4g} median={d['median']:.4g} "
            f"std={d['std']:.4g} CV={d['cv']:.3f} p10={d['p10']:.4g} p90={d['p90']:.4g}")


# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_B(ds, net, to_extri, n_traj, frames_per_traj, W, NS, seed):
    """Per-trajectory Umeyama alignment of LingBot centers -> GT metric centers."""
    dev = net.device
    lo = NS + W - 1
    rng = np.random.default_rng(seed)
    # unique trajectory indices that have samples
    tijs = sorted({int(s["traj_idx"]) for s in ds.samples})
    rng.shuffle(tijs)
    rows = []
    for ti in tijs:
        if len(rows) >= n_traj:
            break
        try:
            _, _, extr, tlen = ds.process_data_parquet(ti)   # extr: [T,4,4] cam->world
        except Exception as e:
            print(f"  [traj {ti}] parquet fail: {e}")
            continue
        rgb_dir = ds.trajectory_rgb_dir[ti]
        cache_path = ds.trajectory_feature_path[ti]
        T = int(min(tlen, extr.shape[0]))
        hi = T - 1
        if hi - lo < frames_per_traj:            # not enough valid frames
            continue
        frames = np.linspace(lo, hi, frames_per_traj).round().astype(int)
        frames = np.unique(frames)
        if frames.size < 4:
            continue
        # rgb frames must exist
        if not all(os.path.isfile(os.path.join(rgb_dir, f"{int(f)}.jpg")) for f in frames):
            continue
        try:
            cache = net._load_cache(cache_path, rgb_dir)
        except Exception as e:
            print(f"  [traj {ti}] cache fail: {e}")
            continue
        ck, cv = cache["cam_k"], cache["cam_v"]
        gt_c, lb_pe = [], []
        ok = True
        for f in frames:
            f = int(f)
            paths = [os.path.join(rgb_dir, f"{i}.jpg") for i in range(f - W + 1, f + 1)]
            try:
                win = net.lingbot.load_images(paths).to(dev)
                _, agg, _ = net.lingbot.window_forward(cache, win, f, return_multilayer=True)
                pose = net.lingbot.camera_pose(ck, cv, f, agg)[-1]     # [9] frame f pose_enc
            except Exception as e:
                print(f"  [traj {ti}] frame {f} pose fail: {e}")
                ok = False
                break
            lb_pe.append(pose)
            gt_c.append(extr[f][:3, 3].astype(np.float64))             # GT is cam->world: [:3,3] IS center
        if not ok or len(gt_c) < 4:
            continue
        X = poses_to_centers(torch.stack(lb_pe), to_extri).astype(np.float64)  # LingBot centers (-R^T T)
        Y = np.stack(gt_c).astype(np.float64)                                  # GT metric centers
        s, R, t, resid = umeyama(X, Y)
        scene = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))   # GT RMS radius (m)
        # per-traj within consistency: does ONE scalar fit all pairwise dists?
        gd, ld = [], []
        for i in range(len(Y)):
            for j in range(i + 1, len(Y)):
                gd.append(np.linalg.norm(Y[i] - Y[j]))
                ld.append(np.linalg.norm(X[i] - X[j]))
        ratios = np.asarray(gd) / (np.asarray(ld) + 1e-12)   # meters per lingbot-unit
        rows.append(dict(ti=ti, n_frames=len(Y), s_traj=abs(s), resid=resid,
                         scene=scene, resid_norm=resid / (scene + 1e-9),
                         within_cv=float(ratios.std() / (ratios.mean() + 1e-12))))
        print(f"  [traj {ti}] frames={len(Y)} s_traj={abs(s):.4g} "
              f"resid={resid:.3g}m scene={scene:.3g}m resid/scene={resid/(scene+1e-9):.3f} "
              f"within_pair_CV={rows[-1]['within_cv']:.3f}")
    return rows


@torch.no_grad()
def probe_A(ds, net, to_extri, n_samples, bs, seed):
    """Real aux magnitudes: LingBot ||goal_pose-cur_pose|| vs GT ||goal_rel_pose[:2]||.

    Teacher-forces the goal_append anchor to a GT positive (net.training=True) —
    exactly the TRAIN path — so this measures scale, not the untrained retrieval
    head's (random) match_idx. The frozen backbone is kept in eval (deterministic)."""
    net.train()
    net.lingbot.eval()          # frozen backbone deterministic; only the anchor branch flips
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(ds), size=n_samples).tolist()
    lb, gt, tid = [], [], []
    for st in range(0, len(idx), bs):
        items = [ds[i] for i in idx[st:st + bs]]
        batch = memnav_collate_fn(items)
        enc = net.encode_memory(batch)
        cur_c = poses_to_centers(enc["cur_pose"], to_extri)           # [B,3] LingBot centers
        goal_c = poses_to_centers(enc["goal_pose"], to_extri)
        lb_d = np.linalg.norm(goal_c - cur_c, axis=1)                 # center distance (LingBot units)
        gt_d = np.linalg.norm(batch["batch_goal_rel_pose"][:, :2].numpy(), axis=1)  # meters
        is_rev = batch["batch_is_revisit"].numpy() > 0.5
        for b in range(len(items)):
            if is_rev[b]:
                lb.append(lb_d[b]); gt.append(gt_d[b]); tid.append(batch["rgb_dirs"][b])
        print(f"  probe A: {len(lb)} revisit samples so far", end="\r")
    print()
    return np.asarray(lb), np.asarray(gt), tid


@torch.no_grad()
def probe_C(ds, net, to_extri, n_decomp, bs, W, NS, seed):
    """Decompose the current->goal relative distance error into two causes:

      C1 (cross-leg DRIFT):  LingBot ||C_m - C_k||   vs GT ||G_m - G_k||
          two REAL frames (current k on leg 2, anchor m on leg 1). Pure track drift.
      C2 (OOD goal, LOCAL):  LingBot ||C_goal - C_m|| vs GT ||G_g - G_m||
          m co-observes the goal, so this is a LOCAL displacement -> isolates the
          goal-image relocalization quality (drift-free).
      full (x-check A):      LingBot ||C_goal - C_k|| vs GT ||G_g - G_k||
          should reproduce Probe A's r ~ 0.21, validating the centers pipeline.

    Decision:
      * C1 r LOW (and <= C2)  -> drift dominates. LoRA on the goal branch CANNOT fix
        it (cur is frozen & independently drifted). Supervise goal rel-to-anchor-m,
        or use point-map geometry instead of the drifted camera-pose track.
      * C2 r LOW (and < C1)   -> goal relocalization is the weak link, real track OK.
        LoRA on the goal-frame pass is well-targeted; cur frozen is fine.
    """
    net.train()                 # teacher-forced anchor path (same as training / probe A)
    net.lingbot.eval()          # frozen backbone deterministic
    dev = net.device
    lo = NS + W - 1
    rev_idx = [i for i, s in enumerate(ds.samples) if s.get("has_covis")]
    rng = np.random.default_rng(seed + 2)
    rng.shuffle(rev_idx)
    rev_idx = rev_idx[:n_decomp]
    extr_cache = {}
    rows = []
    for st in range(0, len(rev_idx), bs):
        chunk = rev_idx[st:st + bs]
        items = [ds[i] for i in chunk]
        smpls = [ds.samples[i % len(ds.samples)] for i in chunk]
        batch = memnav_collate_fn(items)
        enc = net.encode_memory(batch)
        cur_c = poses_to_centers(enc["cur_pose"], to_extri)     # [B,3] LingBot center at k
        goal_c = poses_to_centers(enc["goal_pose"], to_extri)   # [B,3] LingBot center of relocalized goal
        anchor = enc["anchor_idx"]
        for b in range(len(items)):
            k = int(batch["cur_steps"][b])
            ti = int(smpls[b]["traj_idx"])
            rgb_dir = batch["rgb_dirs"][b]
            cache_path = batch["cache_paths"][b]
            m = int(anchor[b].clamp(lo, k - 1).item())
            if ti not in extr_cache:
                try:
                    _, _, extr, tlen = ds.process_data_parquet(ti)
                    extr_cache[ti] = (extr, int(tlen))
                except Exception as e:
                    print(f"  [C traj {ti}] parquet fail: {e}")
                    extr_cache[ti] = None
            if extr_cache[ti] is None:
                continue
            extr, tlen = extr_cache[ti]
            T = int(min(tlen, extr.shape[0]))
            g = min(int(smpls[b]["goal_step"]), T - 1)
            if not (0 <= k < T and lo <= m < T and 0 <= g < T):
                continue
            # LingBot REAL camera pose at anchor frame m (no goal image)
            try:
                cache = net._load_cache(cache_path, rgb_dir)
                ck, cv = cache["cam_k"], cache["cam_v"]
                paths = [os.path.join(rgb_dir, f"{i}.jpg") for i in range(m - W + 1, m + 1)]
                win = net.lingbot.load_images(paths).to(dev)
                _, m_agg, _ = net.lingbot.window_forward(cache, win, m, return_multilayer=True)
                m_pose = net.lingbot.camera_pose(ck, cv, m, m_agg)[-1]
            except Exception as e:
                print(f"  [C traj {ti}] m-pose fail: {e}")
                continue
            C_m = poses_to_centers(m_pose[None], to_extri)[0]
            C_k, C_goal = cur_c[b], goal_c[b]
            G_k = extr[k][:3, 3].astype(np.float64)
            G_m = extr[m][:3, 3].astype(np.float64)
            G_g = extr[g][:3, 3].astype(np.float64)
            rows.append(dict(
                ti=ti, k=k, m=m, g=g,
                c1_lb=float(np.linalg.norm(C_m - C_k)),   c1_gt=float(np.linalg.norm(G_m - G_k)),
                c2_lb=float(np.linalg.norm(C_goal - C_m)), c2_gt=float(np.linalg.norm(G_g - G_m)),
                full_lb=float(np.linalg.norm(C_goal - C_k)), full_gt=float(np.linalg.norm(G_g - G_k)),
            ))
        print(f"  probe C: {len(rows)} samples", end="\r")
    print()
    return rows


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_traj", type=int, default=40)
    ap.add_argument("--frames_per_traj", type=int, default=8)
    ap.add_argument("--n_samples", type=int, default=256)
    ap.add_argument("--n_decomp", type=int, default=80, help="probe C revisit samples")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only_C", action="store_true", help="run only the decomposition probe C")
    ap.add_argument("--skip_C", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]
    feat = os.environ.get("MEMNAV_FEATURE_ROOT")
    repo = os.environ["LINGBOT_REPO"]
    weights = os.environ.get("LINGBOT_WEIGHTS", os.path.join(repo, "weights/lingbot-map-long.pt"))
    W = int(os.environ.get("MEMNAV_WINDOW", 32))
    NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    MFN = int(os.environ.get("MEMNAV_MAX_FRAME_NUM", 2048))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} root={root} feat={feat}\nwindow={W} num_scale={NS} max_frame_num={MFN}")

    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)
    print(f"dataset: {len(ds.samples)} goal-samples")

    net = MemNavNet(
        token_dim=384, heads=8, predict_size=24, temporal_depth=8,
        lingbot_kwargs=dict(lingbot_repo=repo, weights=weights,
                            window=W, num_scale=NS, max_frame_num=MFN),
        device=dev)
    net.eval()

    # LingBot pose_enc -> extrinsics (uses the SAME quat convention the poses were emitted in).
    # Importable now: LingBotStream.__init__ inserted lingbot_repo onto sys.path.
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri as to_extri

    rows, s_all, resid_norm, within_cv = [], [], [], []
    s_stat = {}
    lb = np.asarray([])
    a_summary = {}

    # ---------------- Probe B (headline) ----------------
    if not args.only_C:
        print("\n===== Probe B: per-trajectory Umeyama (LingBot centers -> GT metric) =====")
        rows = probe_B(ds, net, to_extri, args.n_traj, args.frames_per_traj, W, NS, args.seed)
        s_all = [r["s_traj"] for r in rows]
        resid_norm = [r["resid_norm"] for r in rows]
        within_cv = [r["within_cv"] for r in rows]
        s_stat = stats(s_all)
        print(f"\n  s_traj (meters per LingBot-unit) ACROSS {len(rows)} trajectories:\n    {fmt(s_stat)}")
        print(f"  Umeyama residual / scene_scale (per traj):\n    {fmt(stats(resid_norm))}")
        print(f"  within-traj pairwise-ratio CV (single-scalar fit quality):\n    {fmt(stats(within_cv))}")

        # global-scale floor: if we used ONE scale for all, how wrong is each traj?
        if s_stat.get("n", 0):
            s_glob = s_stat["median"]
            rel_err = [abs(r["s_traj"] - s_glob) / s_glob for r in rows]
            print(f"  |s_traj - median|/median (metric error floor for a FIXED-scale head):\n    {fmt(stats(rel_err))}")

    # ---------------- Probe A (real aux signal) ----------------
    if not args.only_C:
        print("\n===== Probe A: real aux magnitudes (revisit rows) =====")
        lb, gt, tid = probe_A(ds, net, to_extri, args.n_samples, args.bs, args.seed)
    if lb.size:
        s_a = gt / (lb + 1e-9)
        print(f"  LingBot ||goal-cur||:   {fmt(stats(lb))}")
        print(f"  GT ||goal_rel[:2]|| (m): {fmt(stats(gt))}")
        print(f"  ratio s=GT/LingBot:     {fmt(stats(s_a))}")
        # log-log correlation (a single global scale => tight line, slope 1)
        m = (lb > 1e-6) & (gt > 1e-6)
        if m.sum() > 4:
            r_log = float(np.corrcoef(np.log(lb[m]), np.log(gt[m]))[0, 1])
            print(f"  Pearson r(log lb, log gt) = {r_log:.3f}  (1.0 => perfect global scale)")
            a_summary["r_log"] = r_log
        a_summary.update(dict(ratio=stats(s_a), lb=stats(lb), gt=stats(gt)))

    # ---------------- Probe C (decomposition: drift vs OOD-goal) ----------------
    c_summary = {}
    crows = []
    if not args.skip_C:
        print("\n===== Probe C: decomposition of current->goal error (drift vs OOD-goal) =====")
        crows = probe_C(ds, net, to_extri, args.n_decomp, args.bs, W, NS, args.seed)
        if crows:
            c_summary["C1_drift"] = corr_report("C1 drift  k<->m (real frames)",
                                                [r["c1_lb"] for r in crows], [r["c1_gt"] for r in crows])
            c_summary["C2_goal"] = corr_report("C2 goal   m<->goal (local reloc)",
                                               [r["c2_lb"] for r in crows], [r["c2_gt"] for r in crows])
            c_summary["full"] = corr_report("full      k<->goal (x-check A)",
                                            [r["full_lb"] for r in crows], [r["full_gt"] for r in crows])
            r1 = c_summary["C1_drift"]["r_log"]
            r2 = c_summary["C2_goal"]["r_log"]
            print("  --- decomposition verdict ---")
            if r1 < 0.4 and r1 <= r2:
                print(f"  DRIFT dominates (C1 r={r1:.2f} <= C2 r={r2:.2f}): cross-leg pose drift corrupts "
                      "the relative. LoRA on the goal branch CANNOT fix it (cur frozen & drifted).\n"
                      "     => supervise goal RELATIVE TO ANCHOR m (local), or use point-map geometry.")
            elif r2 < 0.4 and r2 < r1:
                print(f"  OOD-GOAL dominates (C2 r={r2:.2f} < C1 r={r1:.2f}): goal-image relocalization is "
                      "the weak link; real-frame track is OK.\n"
                      "     => LoRA on the goal-frame pass is well-targeted; cur frozen is fine.")
            else:
                print(f"  BOTH contribute (C1 r={r1:.2f}, C2 r={r2:.2f}): expect LoRA to help partly; "
                      "the drift floor caps how much.")

    # ---------------- verdict ----------------
    print("\n===== VERDICT =====")
    if s_stat.get("n", 0):
        across_cv = s_stat["cv"]
        wcv = stats(within_cv)["median"]
        rn = stats(resid_norm)["median"]
        print(f"  across-traj CV(s_traj) = {across_cv:.3f} | within-traj ratio CV (median) = {wcv:.3f} "
              f"| Umeyama resid/scene (median) = {rn:.3f}")
        if rn > 0.25:
            print("  -> LingBot poses are NOT a clean similarity of GT (high residual): pose signal "
                  "itself is noisy; even bearing may be weak. Inspect before trusting either target.")
        elif across_cv > 0.20:
            print("  -> SCALE VARIES per trajectory (across_cv high) while each traj fits ONE scalar "
                  "(low within CV). Metric (x,y) is UNRECOVERABLE from frozen poses by a fixed head.\n"
                  "     => switch aux target to SCALE-INVARIANT bearing + relative yaw.")
        else:
            print("  -> Scale is ~GLOBALLY constant. Metric (x,y) IS recoverable with a single scale.\n"
                  "     => keep metric target; fix head architecture + loss (relative-geometry input, "
                  "theta = relative camera yaw, Huber+geodesic split).")

    # ---------------- save ----------------
    out = args.out or os.path.join(
        os.environ.get("MEMNAV_LOG_DIR", "/scratch/lg154/Research/Nav/InternNav/logs/train_memnav"),
        "pose_scale")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(dict(config=vars(args), window=W, num_scale=NS,
                       probeB_rows=rows, probeB_s_stat=s_stat,
                       probeB_resid_norm=stats(resid_norm), probeB_within_cv=stats(within_cv),
                       probeA=a_summary, probeC=c_summary, probeC_rows=crows), f, indent=2)
    print(f"\nwrote {out}/summary.json")

    # scatter plots (best-effort)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        if s_all:
            ax[0].hist(s_all, bins=min(30, len(s_all)))
            ax[0].set_title(f"Probe B: s_traj across trajectories (CV={s_stat['cv']:.3f})")
            ax[0].set_xlabel("s_traj = meters per LingBot-unit")
        if lb.size:
            ax[1].scatter(lb, gt, s=8, alpha=0.5)
            if s_stat.get("n", 0):                      # global-scale reference line gt = s*lb
                xl = np.array([0.0, float(lb.max())])
                ax[1].plot(xl, s_stat["median"] * xl, "r--", lw=1,
                           label=f"gt = {s_stat['median']:.3g}·lb (global scale)")
                ax[1].legend()
            ax[1].set_xlabel("LingBot ||goal-cur||")
            ax[1].set_ylabel("GT ||goal_rel[:2]|| (m)")
            ax[1].set_title("Probe A: aux magnitudes (revisit)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "pose_scale.png"), dpi=110)
        print(f"wrote {out}/pose_scale.png")
    except Exception as e:
        print(f"(plot skipped: {e})")

    # Probe C decomposition scatter (log-log, GT vs LingBot for each component)
    if crows:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            comps = [("C1 drift k<->m", "c1_lb", "c1_gt"),
                     ("C2 goal m<->goal", "c2_lb", "c2_gt"),
                     ("full k<->goal", "full_lb", "full_gt")]
            fig, ax = plt.subplots(1, 3, figsize=(16, 5))
            for i, (title, kl, kg) in enumerate(comps):
                x = np.array([r[kl] for r in crows]); y = np.array([r[kg] for r in crows])
                m = (x > 1e-6) & (y > 1e-6)
                ax[i].scatter(x[m], y[m], s=10, alpha=0.5)
                ax[i].set_xscale("log"); ax[i].set_yscale("log")
                rr = np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1] if m.sum() > 4 else float("nan")
                ax[i].set_title(f"{title}  r(log)={rr:.2f}")
                ax[i].set_xlabel("LingBot dist"); ax[i].set_ylabel("GT dist (m)")
            fig.tight_layout()
            fig.savefig(os.path.join(out, "pose_decomp.png"), dpi=110)
            print(f"wrote {out}/pose_decomp.png")
        except Exception as e:
            print(f"(decomp plot skipped: {e})")


if __name__ == "__main__":
    main()
