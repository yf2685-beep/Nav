"""Stage 2/3 of the revisit-definition sweep: run the LingBot insertion path on every
(leg-A, goal-B) pair from revisit_sweep_gen.py and measure relocalization quality.

Prerequisite: KV caches next to each trajectory, produced by
InternNav/scripts/dataset_converters/precompute_lingbot_features.py (lingbot-map env).
The cam cache MUST contain ``cam_pose_enc`` (native per-frame poses) — re-run the
precompute if it predates that field.

Per sweep sample this executes the memnav revisit path (memnav_policy.encode_memory):
inject the precomputed cache, recompute a local window ending at the anchor m, stream
the goal image at temporal slot m+1, read the goal's absolute pose from the frozen
camera head (camera_pose at m+1). The aggregator cache is snapshotted after the
window warmup and restored between goals so each goal sees an identical stream.

Empirical calibration facts baked in here (validated on MP3D smoke data, 2026-07):
  * pose9 decodes as **cam-to-world** (raw [R|t]), NOT world-to-cam as the
    VGGT-inherited pose_enc.py docstring claims (native relative translations match
    GT at ~6 deg direction error only under the c2w reading). The per-trajectory
    calibration still auto-checks both interpretations and records which fits.
  * the default 8-frame window recompute is a POOR approximation of native streaming
    for the camera head (pose cos 0.47 at k=40); ``--warm 32`` recovers native
    fidelity (cos 1.0000). Anchor poses are read from the native ``cam_pose_enc``.
  * monocular scale is ~0.5x metric and drifts; a per-trajectory Umeyama similarity
    (native leg camera centers -> GT centers) supplies (s, R, t) for metric errors.

Each anchor also gets SELF-INSERTION CONTROLS (kind="control"): actual leg frames
j = m+off re-inserted as goals — perfect-covisibility baselines that upper-bound
what any real goal image can achieve. Negatives ("neg") get their insertion anchor
from DINO top-1 (what retrieval would do at deployment).

Run (memnav env, GPU):
  python revisit_sweep_eval.py --sweep_root /home/asus/Research/datasets/memnav_sweep \
      --out /home/asus/Research/Nav/memnav_viz/revisit_sweep/results.parquet
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

INTERNNAV = "/home/asus/Research/Nav/InternNav"
if INTERNNAV not in sys.path:
    sys.path.insert(0, INTERNNAV)

# habitat optical (-Z fwd, +Y up) -> OpenCV (+Z fwd, +Y down) change of basis
C_GL2CV = np.diag([1.0, -1.0, -1.0])
CONTROL_OFFSETS = [0, -3, -8, -15]
_RESTORE_CHECKED = False


# --------------------------------------------------------------------------- #
# SE(3) helpers
# --------------------------------------------------------------------------- #
def decode_pose9(pose9_batch, pose_dec, interp):
    """[N,9] -> [N,4,4] cam-to-world under the chosen interpretation of the decoded
    3x4 extrinsic: 'c2w' takes it verbatim, 'w2c' inverts it."""
    enc = torch.as_tensor(np.asarray(pose9_batch), dtype=torch.float32)[None]
    extri, _ = pose_dec(enc)
    E = extri[0].numpy()                                   # [N,3,4]
    out = np.tile(np.eye(4), (len(E), 1, 1))
    if interp == "c2w":
        out[:, :3, :4] = E
    else:
        out[:, :3, :3] = E[:, :3, :3].transpose(0, 2, 1)
        out[:, :3, 3] = -np.einsum("nji,nj->ni", E[:, :3, :3], E[:, :3, 3])
    return out


def umeyama(src, dst):
    """Similarity fit dst ~= s*R@src + t. Returns (s, R, t, rmse)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sv, Vt = np.linalg.svd(cov)
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = float((sv * np.diag(E)).sum() / (S ** 2).sum() * len(src))
    t = mu_d - s * R @ mu_s
    rmse = float(np.sqrt(((s * src @ R.T + t - dst) ** 2).sum(1).mean()))
    return s, R, t, rmse


def fit_similarity(T_pred, T_gt, C, delta=0.5):
    """Similarity fit pred map -> GT world from matched pose lists (both cam-to-world;
    pred in its own map frame/axes, GT habitat). Scale from pairwise center distances
    (robust median); rotation+translation by Kabsch on centers AUGMENTED with
    forward-offset virtual points, so short straight segments stay conditioned.
    Returns (s, R, t, rmse)."""
    P = np.stack([T[:3, 3] for T in T_pred])
    G = np.stack([T[:3, 3] for T in T_gt])
    ratios = []
    for a in range(len(P)):
        for b in range(a + 1, len(P)):
            dp, dg = np.linalg.norm(P[a] - P[b]), np.linalg.norm(G[a] - G[b])
            if dg > 0.15 and dp > 1e-4:
                ratios.append(dg / dp)
    s = float(np.median(ratios)) if ratios else 1.0
    fwd_cam = C @ np.array([0.0, 0.0, -1.0])                   # GL forward in pred cam axes
    Pv = s * P + delta * np.stack([T[:3, :3] @ fwd_cam for T in T_pred])
    Gv = G + delta * np.stack([T[:3, :3] @ np.array([0.0, 0.0, -1.0]) for T in T_gt])
    src = np.concatenate([s * P, Pv])
    dst = np.concatenate([G, Gv])
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _sv, Vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    t = mu_d - R @ mu_s
    rmse = float(np.sqrt(((s * P @ R.T + t - G) ** 2).sum(1).mean()))
    return s, R, t, rmse


def rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def yaw_of_R(R_c2w_gl):
    """Ground-plane heading of a habitat-convention cam-to-world rotation."""
    f = R_c2w_gl @ np.array([0.0, 0.0, -1.0])
    return float(np.arctan2(-f[0], -f[2]))


def wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def rel_in_anchor_frame(a_T_gt, g_pos, g_R):
    """(fwd, lat, dyaw_deg) of a goal pose relative to the anchor's GT ground frame."""
    a_R, a_p = a_T_gt[:3, :3], a_T_gt[:3, 3]
    fwd_dir = a_R @ np.array([0.0, 0.0, -1.0])
    left_dir = a_R @ np.array([-1.0, 0.0, 0.0])
    d = g_pos - a_p
    dyaw = wrap_deg(np.degrees(yaw_of_R(g_R) - yaw_of_R(a_R)))
    return float(d @ fwd_dir), float(d @ left_dir), float(dyaw)


# --------------------------------------------------------------------------- #
# aggregator-cache snapshot/restore (rewind the stream between goals).
# Truncation-based: streaming ONE goal frame only appends one slot along the frame
# axis of every cache tensor (the model's eviction window is set larger than the
# warmup depth, so nothing is compressed) — restoring is slicing back to the
# warmup frame count. A clone-based snapshot would double the ~8.6 GB live cache
# (64 full frames) and OOM alongside other GPU tenants.
# --------------------------------------------------------------------------- #
def snap_cache(agg):
    shapes = {k: (v.shape[2] if torch.is_tensor(v) else None)
              for k, v in agg.kv_cache.items()}
    return shapes, int(agg.total_frames_processed)


def restore_cache(agg, snap):
    shapes, tot = snap
    kv = agg.kv_cache
    for k in [k for k in kv if k not in shapes]:
        del kv[k]
    for k, f0 in shapes.items():
        if f0 is not None and torch.is_tensor(kv.get(k)) and kv[k].shape[2] > f0:
            kv[k] = kv[k][:, :, :f0]
    agg.total_frames_processed = tot


# --------------------------------------------------------------------------- #
# discovery / cache loading
# --------------------------------------------------------------------------- #
def find_sweep_trajs(root):
    out = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "sweep" and "sweep_meta.json" in filenames:
            out.append(os.path.dirname(dirpath))
    return sorted(out)


def load_cache(chunk_dir, device):
    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
    c = np.load(os.path.join(chunk_dir, "lingbot_cache.npz"))
    sk, sv, ak, av = LingBotStream._cache_to_layered(
        c["scale_k"], c["scale_v"], c["anchor_k"], c["anchor_v"], device)
    cc = np.load(os.path.join(chunk_dir, "lingbot_cam_cache.npz"))
    if "cam_pose_enc" not in cc:
        raise RuntimeError(f"{chunk_dir}: cam cache lacks cam_pose_enc — re-run "
                           "precompute_lingbot_features.py (updated version)")
    ck, cv = LingBotStream._cam_to_device(cc["cam_k"], cc["cam_v"], device)
    dino = torch.as_tensor(c["dino_cls"], device=device, dtype=torch.float32)
    return (dict(scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av),
            ck, cv, dino, cc["cam_pose_enc"])


# --------------------------------------------------------------------------- #
# warm forward: inject cache up to m-warm, recompute [.. m] live (native-fidelity
# context for the goal stream; window_forward's default 8 is NOT enough for the
# camera head — see module docstring)
# --------------------------------------------------------------------------- #
def warm_forward(ls, cache, rgb_dir, m, warm, cal_step=3, cal_skip=8):
    """Warm the stream to [0..m]; return {frame_idx: agg_list} for calibration frames —
    the frames whose LIVE camera poses anchor the metric alignment (the live context's
    map frame drifts away from the native one at deep anchors, so the reference must
    come from the same context as the goal). The FIRST `cal_skip` warm frames are
    excluded from calibration when there is compressed (specials-only) history: they
    see no full-KV predecessors and their poses are the least reliable. When the warm
    window starts at the scale block the context is exact and all frames are kept."""
    start = max(ls.num_scale, m - warm + 1)
    n_hist = start - ls.num_scale
    skip = cal_skip if n_hist > 0 else 0
    ls._inject(cache["scale_k"], cache["scale_v"], cache["anchor_k"], cache["anchor_v"],
               n_hist=n_hist, total_frames=start)
    imgs = ls.load_images([os.path.join(rgb_dir, f"{i}.jpg") for i in range(start, m + 1)])
    aggs = {}
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for j in range(len(imgs)):
            a, _ = ls.model._aggregate_features(
                imgs[j:j + 1][None].to(ls.device),
                num_frame_for_scale=ls.num_scale, num_frame_per_block=1)
            k = start + j
            if (j >= skip and (j - skip) % cal_step == 0) or k == m:
                aggs[k] = [layer for layer in a]
    return aggs


# --------------------------------------------------------------------------- #
# per-trajectory evaluation
# --------------------------------------------------------------------------- #
def eval_trajectory(ls, traj_dir, args, pose_dec):
    device = ls.device
    sweep = json.load(open(os.path.join(traj_dir, "sweep", "sweep_meta.json")))
    chunk_dir = os.path.join(traj_dir, "videos", "chunk-000")
    rgb_dir = os.path.join(chunk_dir, "observation.images.rgb")
    cache, ck, cv, dino_mem, cam_pose = load_cache(chunk_dir, device)

    n = sweep["n_frames"]
    lo = sweep["num_scale"] + sweep["window"] - 1
    anchors = [a for a in sweep["anchors"] if lo <= a < n - 1]
    gt_T = [np.array(p) for p in sweep["leg_T_wc_hab"]]
    gt_c = np.stack([T[:3, 3] for T in gt_T])

    # ---- calibration from NATIVE poses: interpretation + similarity + axis conv ----
    fits = {}
    for interp in ("c2w", "w2c"):
        Tn = decode_pose9(cam_pose, pose_dec, interp)
        s, Ra, ta, rmse = umeyama(Tn[:, :3, 3], gt_c)
        fits[interp] = (Tn, s, Ra, ta, rmse)
    interp = min(fits, key=lambda k: fits[k][4])
    Tn, s, Ra, ta, calib_rmse = fits[interp]
    conv_res = {}
    for name, C in (("gl2cv", C_GL2CV), ("identity", np.eye(3))):
        errs = [rot_angle_deg(Ra @ Tn[k, :3, :3] @ C, gt_T[k][:3, :3])
                for k in range(lo, n, max(1, n // 20))]
        conv_res[name] = float(np.median(errs))
    conv = min(conv_res, key=conv_res.get)
    C = C_GL2CV if conv == "gl2cv" else np.eye(3)
    calib_rot_med = conv_res[conv]

    calib_cols = dict(calib_pos_rmse=calib_rmse, calib_rot_med_deg=calib_rot_med,
                      calib_scale=s, pose_interp=interp, cam_convention=conv)

    # ---- DINO CLS of all goal images (batched) ----
    recs = sweep["records"]
    goal_imgs = ls.load_images([os.path.join(traj_dir, "sweep", r["goal_file"]) for r in recs])
    sims_all = []
    for i in range(0, len(recs), 32):
        cls = ls.dino(goal_imgs[i:i + 32].to(device))["cls"]
        sims_all.append(torch.nn.functional.cosine_similarity(
            cls[:, None], dino_mem[None], dim=-1).cpu())
    sims_all = torch.cat(sims_all)                                # [n_goals, n_frames]

    # ---- group work items by insertion anchor ----
    # grid -> its generating anchor; neg -> DINO top-1 (deployment retrieval);
    # controls -> every anchor gets CONTROL_OFFSETS leg frames.
    by_anchor = {m: [] for m in anchors}
    for i, r in enumerate(recs):
        m = (r["anchor_idx"] if r["kind"] in ("grid", "rand")
             else int(sims_all[i].argmax().clamp(lo, n - 2).item()))
        by_anchor.setdefault(m, []).append(("rec", i))
    for m in anchors:
        for off in CONTROL_OFFSETS:
            if 0 <= m + off < n:
                by_anchor[m].append(("ctrl", m + off))

    rows = []
    for m, items in sorted(by_anchor.items()):
        ls.model.clean_kv_cache()
        torch.cuda.empty_cache()
        aggs = warm_forward(ls, cache, rgb_dir, m, args.warm)
        # LIVE poses of the warm calibration frames (same context the goal will see)
        live_T = {k: decode_pose9(ls.camera_pose(ck, cv, k, aggs[k])[-1]
                                  .float().cpu().numpy()[None], pose_dec, interp)[0]
                  for k in sorted(aggs)}
        cal_idx = sorted(live_T)
        s_l, R_l, t_l, local_rmse = fit_similarity(
            [live_T[k] for k in cal_idx], [gt_T[k] for k in cal_idx], C)
        aT_gt = gt_T[m]
        aT_map = live_T[m]                                        # live anchor pose

        def aligned(T_map):
            """pred cam-to-world (live map frame) -> (pos in GT world, R with habitat
            cam axes), via the per-anchor live similarity fit."""
            return s_l * R_l @ T_map[:3, 3] + t_l, R_l @ T_map[:3, :3] @ C

        snap = snap_cache(ls.agg)

        def insert(img):
            restore_cache(ls.agg, snap)
            with torch.no_grad():
                _tok, gagg = ls._stream_one(img.to(device), return_agg=True)
            p9 = ls.camera_pose(ck, cv, m + 1, gagg)[-1]
            return decode_pose9(p9.float().cpu().numpy()[None], pose_dec, interp)[0]

        # one-time self-check: truncation restore must make insertion idempotent
        global _RESTORE_CHECKED
        if not _RESTORE_CHECKED and items:
            img0 = (ls.load_images([os.path.join(rgb_dir, f"{m}.jpg")])[0]
                    if items[0][0] == "ctrl" else goal_imgs[items[0][1]])
            d = np.abs(insert(img0) - insert(img0)).max()
            assert d < 1e-3, f"truncation restore not idempotent (max delta {d})"
            _RESTORE_CHECKED = True

        for tag, idx in items:
            if tag == "ctrl":
                j = idx
                img = ls.load_images([os.path.join(rgb_dir, f"{j}.jpg")])[0]
                gT_map = insert(img)
                T_goal_gt = gt_T[j]
                base = dict(sid=-1 - j, kind="control",
                            covis_goal_in_anchor=np.nan, covis_anchor_in_goal=np.nan,
                            covis_max=np.nan, covis_argmax=j,
                            dino_top1_sim=np.nan, dino_top1_idx=j,
                            dino_sim_at_anchor=np.nan, dino_top1_near_anchor=True,
                            req_fwd=np.nan, req_lat=np.nan, req_dyaw=np.nan)
            else:
                r = recs[idx]
                gT_map = insert(goal_imgs[idx])
                T_goal_gt = np.array(r["gt_T_wc_hab"])
                sims = sims_all[idx]
                top1 = int(sims.argmax())
                base = dict(sid=r["sid"], kind=r["kind"],
                            covis_goal_in_anchor=r["covis_goal_in_anchor"],
                            covis_anchor_in_goal=r["covis_anchor_in_goal"],
                            covis_max=r["covis_max"], covis_argmax=r["covis_argmax"],
                            dino_top1_sim=float(sims[top1]), dino_top1_idx=top1,
                            dino_sim_at_anchor=(float(sims[r["anchor_idx"]])
                                                if r["kind"] in ("grid", "rand") else np.nan),
                            dino_top1_near_anchor=(abs(top1 - r["anchor_idx"]) <= 3
                                                   if r["kind"] in ("grid", "rand") else False),
                            req_fwd=(r["req"]["fwd"] if r["req"] else np.nan),
                            req_lat=(r["req"]["lat"] if r["req"] else np.nan),
                            req_dyaw=(r["req"]["dyaw_deg"] if r["req"] else np.nan))

            # ---- PRIMARY metrics: goal relative to the LIVE anchor pose (same
            # context). The map->world alignment rotation cancels; only the
            # per-anchor scale s_l enters. Robust to deep-anchor map distortion. ----
            gt_fwd, gt_lat, gt_dyaw = rel_in_anchor_frame(aT_gt, T_goal_gt[:3, 3],
                                                          T_goal_gt[:3, :3])
            rel_map = np.linalg.inv(aT_map) @ gT_map              # cam_goal in cam_anchor
            R_rel_gl = C @ rel_map[:3, :3] @ C                    # pred cam axes -> GL
            t_rel_gl = s_l * (C @ rel_map[:3, 3])
            pr_fwd, pr_lat = -float(t_rel_gl[2]), -float(t_rel_gl[0])   # GL: fwd=-Z, left=-X
            f = R_rel_gl @ np.array([0.0, 0.0, -1.0])
            pr_dyaw = float(np.degrees(np.arctan2(-f[0], -f[2])))
            rel_pos_err = float(np.hypot(pr_fwd - gt_fwd, pr_lat - gt_lat))
            yaw_err = abs(wrap_deg(pr_dyaw - gt_dyaw))
            # scale-invariant translation metrics (LingBot is scale-ambiguous, so the
            # metric magnitude is untrustworthy; direction is the robust signal)
            gtv, prv = np.array([gt_fwd, gt_lat]), np.array([pr_fwd, pr_lat])
            ng, np_ = np.linalg.norm(gtv), np.linalg.norm(prv)
            t_dir_err = (float(np.degrees(np.arccos(np.clip(gtv @ prv / (ng * np_), -1, 1))))
                         if ng > 0.15 and np_ > 0.05 else np.nan)
            t_mag_ratio = float(np_ / ng) if ng > 0.15 else np.nan
            # ---- secondary: absolute pose after the per-anchor live alignment ----
            p_hat, R_hat = aligned(gT_map)
            rows.append(dict(
                traj=os.path.join(os.path.basename(os.path.dirname(traj_dir)),
                                  os.path.basename(traj_dir)),
                scene=sweep["scene"],
                anchor_idx=int(m), n_frames=n,
                fwd=gt_fwd, lat=gt_lat, dyaw_deg=gt_dyaw,
                rel_pos_err_m=rel_pos_err, yaw_err_deg=yaw_err,
                t_dir_err_deg=t_dir_err, t_mag_ratio=t_mag_ratio,
                pred_fwd=pr_fwd, pred_lat=pr_lat, pred_dyaw_deg=pr_dyaw,
                pos_err_m=float(np.linalg.norm(p_hat - T_goal_gt[:3, 3])),
                rot_err_deg=rot_angle_deg(R_hat, T_goal_gt[:3, :3]),
                rel_t_map=float(np.linalg.norm(rel_map[:3, 3])),
                local_rmse=local_rmse, local_scale=s_l,
                **base, **calib_cols,
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True)
    ap.add_argument("--out", required=True, help="results parquet path")
    ap.add_argument("--lingbot_repo", default="/home/asus/Research/lingbot-map")
    ap.add_argument("--weights",
                    default="/home/asus/Research/lingbot-map/weights/lingbot-map-long.pt")
    ap.add_argument("--window", type=int, default=64,
                    help="kv_cache_sliding_window for LingBot. Its default is 64 and "
                         "quality degrades sharply below 32 — do NOT leave this at the "
                         "memnav wrapper's 8. Must match the precompute setting.")
    ap.add_argument("--warm", type=int, default=64,
                    help="window-recompute depth before the anchor (deep warmup; the "
                         "first frames of it are excluded from pose calibration). "
                         "Keep >= --window so the anchor sees a full live window.")
    ap.add_argument("--limit", type=int, default=0, help="max trajectories (0 = all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
    # eviction window > warm: the per-anchor session (warm frames + 1 goal) must
    # never evict, so truncation restore stays exact. History beyond the warm window
    # is injected as specials, matching the precompute's native window-64 eviction.
    ls = LingBotStream(lingbot_repo=args.lingbot_repo, weights=args.weights,
                       window=max(args.window, args.warm) + 8, device=args.device)
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    def pose_dec(enc):
        return pose_encoding_to_extri_intri(enc, image_size_hw=(518, 518),
                                            build_intrinsics=False)

    trajs = find_sweep_trajs(args.sweep_root)
    if args.limit:
        trajs = trajs[:args.limit]
    print(f"{len(trajs)} sweep trajectories under {args.sweep_root}  (warm={args.warm})")

    all_rows = []
    for t in trajs:
        if not os.path.exists(os.path.join(t, "videos/chunk-000/lingbot_cache.npz")):
            print(f"[skip] no lingbot_cache.npz: {t} (run precompute_lingbot_features.py)")
            continue
        print(f"[traj] {t}")
        rows = eval_trajectory(ls, t, args, pose_dec)
        r0 = rows[0]
        ctrl = [r for r in rows if r["kind"] == "control" and abs(r["fwd"]) < 0.02
                and abs(r["dyaw_deg"]) < 2]
        ctrl_pos = np.median([r["rel_pos_err_m"] for r in ctrl]) if ctrl else float("nan")
        print(f"  {len(rows)} samples  calib: rmse={r0['calib_pos_rmse']:.3f}m "
              f"rot_med={r0['calib_rot_med_deg']:.1f}deg scale={r0['calib_scale']:.3f} "
              f"interp={r0['pose_interp']} conv={r0['cam_convention']} | "
              f"identity-control pos_err median={ctrl_pos:.3f}m")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out)
    print(f"DONE: {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
