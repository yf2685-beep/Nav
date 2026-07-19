"""Visualize LingBot predicted camera trajectory vs GT for ONE episode.

Hypothesis under test: LingBot's pose track drifts at SHARP TURNS (fast rotation,
small translation — the classic monocular failure), and the 2-leg episodes have an
in-place align_turn at the leg boundary. If the predicted track diverges from GT
right at the turns, that explains the weak current->goal relative pose (Probe A/C).

Two predicted tracks are compared against GT, each Umeyama-aligned (sim3) to GT:
  * NATIVE  : original lingbot-map full-sequence NON-CAUSAL forward
              (model.forward with num_frame_per_block=N -> all frames attend
              bidirectionally). Answers "is the MODEL weak at turns?"
  * STREAM  : our causal window_forward + camera_pose per frame (the exact path
              training consumes). Answers "does our harness add turn error?"

Outputs (to --out):
  traj_viz.png  : (left) top-down xy overlay GT vs NATIVE vs STREAM, colored by
                  frame, leg-switch + goal marked;
                  (right) per-frame aligned position error vs frame, with |GT
                  yaw-rate| overlaid so turn-localized error spikes are obvious.
  traj_viz.npz  : raw centers + errors + yaw for offline re-plotting.

Run inside the apptainer overlay (same as the pose-scale probe):
  python scripts/diag_traj_viz.py --traj_idx -1 --max_frames 160
"""
import argparse
import os

import numpy as np
import torch

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset
from internnav.model.basemodel.memnav.memnav_policy import MemNavNet


def umeyama(X, Y):
    """Least-squares similarity Y ~= s R X + t. X,Y:[n,3]. Returns (s,R,t,Yhat,resid)."""
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
    return s, R, t, Yhat, resid


def w2c_to_centers(pose_enc, to_extri, inverse_se3, H, W):
    """pose_enc [N,9] (camera-FROM-world) -> camera centers [N,3] in world frame."""
    extr_w2c, _ = to_extri(pose_enc[None].float(), (H, W),
                           pose_encoding_type="absT_quaR_FoV", build_intrinsics=False)
    N = extr_w2c.shape[1]
    E = torch.eye(4, device=extr_w2c.device, dtype=extr_w2c.dtype).repeat(1, N, 1, 1)
    E[..., :3, :4] = extr_w2c
    c2w = inverse_se3(E)                       # [1,N,4,4] camera-to-world
    return c2w[0, :, :3, 3].float().cpu().numpy()


def gt_yaw(extr, frames):
    """GT camera-to-world extrinsics -> heading yaw per frame (heading axis = R col 0)."""
    R = extr[frames][:, :3, :3]
    return np.arctan2(R[:, 1, 0], R[:, 0, 0])   # atan2(R[1,0], R[0,0])


@torch.no_grad()
def native_track(net, imgs, to_extri, inverse_se3, num_scale, native_window):
    """REAL native lingbot-map streaming inference over CONSECUTIVE frames.

    Uses model.inference_streaming: phase-1 bidirectional over the `num_scale`
    scale anchors, then frame-by-frame causal streaming with the true KV-cache
    eviction (scale anchors + `native_window` sliding window in full KV, everything
    else -> 6 special tokens). This is how LingBot is actually run; a stream CANNOT
    be subsampled (that breaks the motion prior), so `imgs` MUST be consecutive.

    The model was built with kv_cache_sliding_window=our-32 (to match caches); for
    the NATIVE reference we temporarily override every attention layer to LingBot's
    own default sliding window (64), then restore."""
    model = net.lingbot.model
    N, H, W = imgs.shape[0], imgs.shape[-2], imgs.shape[-1]
    saved = [(m, m.kv_cache_sliding_window) for m in model.modules()
             if hasattr(m, "kv_cache_sliding_window")]
    for m, _ in saved:
        m.kv_cache_sliding_window = native_window
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.inference_streaming(imgs, num_scale_frames=num_scale,
                                             output_device=torch.device("cpu"))
    finally:
        for m, old in saved:
            m.kv_cache_sliding_window = old
    print(f"NATIVE stream: {N} frames, sliding_window={native_window}, scale={num_scale}")
    pe = pred["pose_enc"][0].to(net.device)      # [N,9]
    conf = None
    if "depth_conf" in pred:
        conf = pred["depth_conf"][0].float().mean(dim=(-1, -2)).cpu().numpy()   # [N]
    return w2c_to_centers(pe, to_extri, inverse_se3, H, W), conf


@torch.no_grad()
def stream_track(net, rgb_dir, cache_path, frames, W, lo, to_extri):
    """Our causal window_forward + camera_pose per frame -> (frames_ok, centers)."""
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri as to_extri2  # noqa
    dev = net.device
    cache = net._load_cache(cache_path, rgb_dir)
    ck, cv = cache["cam_k"], cache["cam_v"]
    fs, pes = [], []
    for f in frames:
        f = int(f)
        if f < lo:
            continue
        paths = [os.path.join(rgb_dir, f"{i}.jpg") for i in range(f - W + 1, f + 1)]
        if not all(os.path.isfile(p) for p in paths):
            continue
        win = net.lingbot.load_images(paths).to(dev)
        _, agg, _ = net.lingbot.window_forward(cache, win, f, return_multilayer=True)
        pes.append(net.lingbot.camera_pose(ck, cv, f, agg)[-1])
        fs.append(f)
    if not pes:
        return np.array([], dtype=int), np.zeros((0, 3))
    pe = torch.stack(pes)                        # [n,9]
    # centers via -R^T T (streaming poses share the same absT_quaR_FoV convention)
    E = to_extri(pe[None].float(), None, pose_encoding_type="absT_quaR_FoV",
                 build_intrinsics=False)[0][0]   # [n,3,4]
    R, T = E[:, :3, :3], E[:, :3, 3]
    C = -torch.einsum("nij,nj->ni", R.transpose(1, 2), T)
    return np.asarray(fs, dtype=int), C.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_idx", type=int, default=-1, help="-1 = first revisit trajectory")
    ap.add_argument("--max_frames", type=int, default=512,
                    help="cap native stream to first N CONSECUTIVE frames (stream can't be subsampled)")
    ap.add_argument("--start_frame", type=int, default=0, help="stream from this frame")
    ap.add_argument("--stream_stride", type=int, default=4,
                    help="subsample OUR window_forward eval points (overlay only; native is dense)")
    ap.add_argument("--native_window", type=int, default=64,
                    help="sliding window for the NATIVE reference stream (LingBot default 64)")
    ap.add_argument("--img_mode", default="pad", choices=["pad", "crop"],
                    help="preprocessing for the NATIVE stream images: 'pad'=our pipeline "
                         "(caches built this way), 'crop'=LingBot demo default")
    ap.add_argument("--no_stream", action="store_true", help="skip our window_forward overlay (native only)")
    ap.add_argument("--out", default="/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/traj_viz")
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]
    feat = os.environ.get("MEMNAV_FEATURE_ROOT")
    repo = os.environ["LINGBOT_REPO"]
    weights = os.environ.get("LINGBOT_WEIGHTS", os.path.join(repo, "weights/lingbot-map-long.pt"))
    W = int(os.environ.get("MEMNAV_WINDOW", 32))
    NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    MFN = int(os.environ.get("MEMNAV_MAX_FRAME_NUM", 2048))
    lo = NS + W - 1

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)
    net = MemNavNet(token_dim=384, heads=8, predict_size=24, temporal_depth=8,
                    lingbot_kwargs=dict(lingbot_repo=repo, weights=weights,
                                        window=W, num_scale=NS, max_frame_num=MFN),
                    device=dev)
    net.eval()
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri as to_extri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general as inverse_se3

    # ---- pick trajectory + markers (leg switch = align_turn, goal arrival) ----
    ti = args.traj_idx
    leg_start = goal_step = amargin = None
    if ti < 0:
        for s in ds.samples:
            if s.get("has_covis"):
                ti = int(s["traj_idx"]); leg_start = int(s.get("leg_start", -1))
                goal_step = int(s["goal_step"]); amargin = int(s.get("amargin", -1))
                break
    if ti is None or ti < 0:
        ti = int(ds.samples[0]["traj_idx"])
    print(f"traj_idx={ti} leg_start={leg_start} goal_step={goal_step} amargin={amargin}")

    _, _, extr, tlen = ds.process_data_parquet(ti)   # [T,4,4] cam->world
    T = int(min(tlen, extr.shape[0]))
    rgb_dir = ds.trajectory_rgb_dir[ti]
    cache_path = ds.trajectory_feature_path[ti]

    # CONSECUTIVE frames [start_frame, start_frame+max_frames) — a real stream cannot
    # be subsampled (that breaks LingBot's motion prior; that was the earlier confound).
    a = max(0, args.start_frame)
    b = min(T, a + args.max_frames)
    frames = np.arange(a, b)
    frames = frames[[os.path.isfile(os.path.join(rgb_dir, f"{int(f)}.jpg")) for f in frames]]
    print(f"T={T} streaming CONSECUTIVE frames [{frames[0]}..{frames[-1]}] ({len(frames)})")

    paths = [os.path.join(rgb_dir, f"{int(f)}.jpg") for f in frames]
    imgs = net.lingbot._preprocess(paths, mode=args.img_mode,
                                   image_size=net.lingbot.img_size, patch_size=net.lingbot.patch_size)
    print(f"native images: mode={args.img_mode} shape={tuple(imgs.shape)}")
    gt_c = extr[frames][:, :3, 3].astype(np.float64)
    yaw = gt_yaw(extr, frames)

    # ---- native track ----
    nat_c, conf = native_track(net, imgs, to_extri, inverse_se3, NS, args.native_window)
    s_n, R_n, t_n, nat_al, resid_n = umeyama(nat_c.astype(np.float64), gt_c)
    scene = np.sqrt(((gt_c - gt_c.mean(0)) ** 2).sum(1).mean())
    print(f"NATIVE: umeyama s={s_n:.4g} resid={resid_n:.3g}m scene={scene:.3g}m "
          f"resid/scene={resid_n/(scene+1e-9):.3f}")

    # ---- our window_forward track (subsampled overlay) ----
    str_f, str_al = np.array([], dtype=int), np.zeros((0, 3))
    resid_s = float("nan")
    if not args.no_stream:
        stride = max(1, args.stream_stride)
        sf = frames[::stride]
        str_f, str_c = stream_track(net, rgb_dir, cache_path, sf, W, lo, to_extri)
        if len(str_f) >= 4:
            gt_s = extr[str_f][:, :3, 3].astype(np.float64)
            s_s, R_s, t_s, str_al, resid_s = umeyama(str_c.astype(np.float64), gt_s)
            scene_s = np.sqrt(((gt_s - gt_s.mean(0)) ** 2).sum(1).mean())
            print(f"OUR-STREAM(win={W}): n={len(str_f)} umeyama s={s_s:.4g} resid={resid_s:.3g}m "
                  f"scene={scene_s:.3g}m resid/scene={resid_s/(scene_s+1e-9):.3f}")

    # per-frame aligned position error (native)
    err_n = np.linalg.norm(nat_al - gt_c, axis=1)
    yaw_rate = np.abs(np.concatenate([[0.0], np.diff(np.unwrap(yaw))])) * 180.0 / np.pi  # deg/step

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "traj_viz.npz"),
             ti=ti, frames=frames, gt_c=gt_c, nat_al=nat_al, err_n=err_n,
             yaw=yaw, yaw_rate=yaw_rate, conf=conf if conf is not None else [],
             str_f=str_f, str_al=str_al, leg_start=leg_start or -1, goal_step=goal_step or -1,
             resid_native=resid_n, resid_stream=resid_s)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        # left: top-down xy overlay (world ground plane = x,y for Z-up data)
        ax[0].plot(gt_c[:, 0], gt_c[:, 1], "-", color="k", lw=2.0, label="GT", zorder=1)
        sc = ax[0].scatter(nat_al[:, 0], nat_al[:, 1], c=np.arange(len(nat_al)),
                           cmap="viridis", s=14, zorder=3,
                           label=f"NATIVE stream win={args.native_window} (aligned)")
        ax[0].plot(nat_al[:, 0], nat_al[:, 1], "-", color="tab:blue", lw=0.8, alpha=0.5, zorder=2)
        if len(str_al):
            ax[0].plot(str_al[:, 0], str_al[:, 1], "-o", color="tab:red", ms=3, lw=0.8,
                       alpha=0.7, label=f"OURS window_forward win={W} (aligned)", zorder=2)
        # markers
        for fidx, lbl, col in [(leg_start, "leg switch (align_turn)", "orange"),
                               (goal_step, "goal", "magenta")]:
            if fidx is not None and fidx >= 0:
                j = int(np.argmin(np.abs(frames - fidx)))
                ax[0].scatter([gt_c[j, 0]], [gt_c[j, 1]], marker="*", s=260, color=col,
                              edgecolor="k", zorder=5, label=lbl)
        ax[0].scatter([gt_c[0, 0]], [gt_c[0, 1]], marker="s", s=80, color="lime",
                      edgecolor="k", zorder=5, label="start")
        ax[0].set_aspect("equal"); ax[0].legend(fontsize=8)
        ax[0].set_title(f"traj {ti}: top-down camera track "
                        f"(native resid/scene={resid_n/(scene+1e-9):.2f})")
        ax[0].set_xlabel("world x (m)"); ax[0].set_ylabel("world y (m)")
        fig.colorbar(sc, ax=ax[0], label="frame order")

        # right: aligned position error vs frame, |GT yaw-rate| overlaid
        ax[1].plot(frames, err_n, "-", color="tab:blue", lw=1.5, label="NATIVE aligned pos err (m)")
        ax2 = ax[1].twinx()
        ax2.plot(frames, yaw_rate, "-", color="tab:green", lw=1.0, alpha=0.7,
                 label="|GT yaw-rate| (deg/step)")
        ax2.set_ylabel("|GT yaw-rate| (deg/step)", color="tab:green")
        for fidx, col in [(leg_start, "orange"), (goal_step, "magenta")]:
            if fidx is not None and fidx >= 0:
                ax[1].axvline(fidx, color=col, ls="--", lw=1.5)
        ax[1].set_xlabel("frame"); ax[1].set_ylabel("position error (m)", color="tab:blue")
        ax[1].set_title("error vs turn rate (spike at turns => turn-induced drift)")
        ax[1].legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "traj_viz.png"), dpi=120)
        print(f"wrote {args.out}/traj_viz.png  and  traj_viz.npz")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
