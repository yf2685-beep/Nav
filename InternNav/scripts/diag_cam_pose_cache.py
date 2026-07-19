"""Compare the CACHED native streaming pose (`cam_pose_enc`, written by the
precompute during its real streaming pass) against GT camera centers — no model
rerun, just load the npz. This is the most faithful check of what training's
frozen pose signal actually is (the cache is exactly what LingBotStream replays).

For one trajectory:
  * load lingbot_cam_cache.npz -> cam_pose_enc [S,9] (absT_quaR_FoV, cam-FROM-world)
  * GT extrinsics from process_data_parquet -> centers [T,3]
  * Umeyama sim3 align, report GLOBAL + LOCAL-windowed resid/scene, plot overlay.

Local resid low + global high == accumulated streaming drift (LingBot is locally
accurate). Run inside the apptainer overlay (parquet lives in the .sqf):
  python scripts/diag_cam_pose_cache.py --traj_idx 0
"""
import argparse
import os

import numpy as np
import torch

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset


def umeyama(X, Y):
    n = len(X); mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my
    S = (Yc.T @ Xc) / n; U, D, Vt = np.linalg.svd(S); s = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: s[2, 2] = -1
    R = U @ s @ Vt; var = (Xc ** 2).sum() / n
    sc = float(np.trace(np.diag(D) @ s) / max(var, 1e-12))
    t = my - sc * R @ mx; Yh = (sc * (R @ X.T)).T + t
    resid = float(np.sqrt(((Y - Yh) ** 2).sum(1).mean()))
    scene = float(np.sqrt(((Y - Y.mean(0)) ** 2).sum(1).mean()))
    return sc, R, t, Yh, resid, scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_idx", type=int, default=0)
    ap.add_argument("--out", default="/scratch/lg154/Research/Nav/InternNav/logs/train_memnav/cam_pose_cache")
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]
    feat = os.environ.get("MEMNAV_FEATURE_ROOT")
    repo = os.environ["LINGBOT_REPO"]
    W = int(os.environ.get("MEMNAV_WINDOW", 32))
    NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))

    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)
    import sys
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri as to_extri
    from lingbot_map.utils.geometry import closed_form_inverse_se3_general as inverse_se3

    ti = args.traj_idx
    cache_path = ds.trajectory_feature_path[ti]
    cam_path = cache_path.replace("lingbot_cache.npz", "lingbot_cam_cache.npz")
    print(f"traj {ti}\n  cam cache: {cam_path}")
    cc = np.load(cam_path)
    pe = torch.from_numpy(cc["cam_pose_enc"].astype(np.float32))   # [S,9] absT_quaR_FoV
    S = pe.shape[0]

    _, _, extr, tlen = ds.process_data_parquet(ti)                # [T,4,4] cam->world
    T = int(min(tlen, extr.shape[0], S))
    pe = pe[:T]
    print(f"  cached poses S={S}, parquet T={tlen}, using T={T}")
    gt_c = extr[:T, :3, 3].astype(np.float64)

    # Two conventions for turning cam_pose_enc into a camera CENTER:
    #  (A) DIRECT: absT is already the cam-to-world position (what the VALIDATED
    #      diag_lingbot_pose_accuracy.py uses -> good ATE for this checkpoint).
    #  (B) INVERT: treat as w2c [R|T], center = -R^T T (demo.py / VGGT docstring).
    pred_direct = pe[:, :3].numpy().astype(np.float64)                       # (A)
    extr_w2c, _ = to_extri(pe[None], (518, 518), pose_encoding_type="absT_quaR_FoV",
                           build_intrinsics=False)
    E = torch.eye(4).repeat(1, T, 1, 1); E[..., :3, :4] = extr_w2c
    pred_invert = inverse_se3(E)[0, :, :3, 3].numpy().astype(np.float64)     # (B)

    results = {}
    for name, pred_c in [("DIRECT absT (validated)", pred_direct),
                         ("INVERT -R^T T (my old bug)", pred_invert)]:
        sc, R, t, pred_al, resid, scene = umeyama(pred_c, gt_c)
        loc = {}
        for w in (16, 32, 64):
            rs = [umeyama(pred_c[a:a + w], gt_c[a:a + w])[4] /
                  (umeyama(pred_c[a:a + w], gt_c[a:a + w])[5] + 1e-9)
                  for a in range(0, T - w, max(1, w // 2))]
            loc[w] = float(np.median(rs))
        print(f"\n  [{name}]  GLOBAL resid/scene={resid/(scene+1e-9):.3f} (s={sc:.4g}, resid={resid:.3g}m) "
              f"| LOCAL w16={loc[16]:.3f} w32={loc[32]:.3f} w64={loc[64]:.3f}")
        results[name] = (pred_al, resid, scene)

    # plot the DIRECT (validated) convention
    pred_al, resid, scene = results["DIRECT absT (validated)"]

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, f"cam_pose_cache_traj{ti}.npz"),
             ti=ti, pred_al=pred_al, gt_c=gt_c, resid=resid, scene=scene)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.plot(gt_c[:, 0], gt_c[:, 1], "-k", lw=2, label="GT")
        s2 = ax.scatter(pred_al[:, 0], pred_al[:, 1], c=np.arange(T), cmap="viridis",
                        s=12, label="cached cam_pose_enc (aligned)")
        ax.plot(pred_al[:, 0], pred_al[:, 1], "-", color="tab:blue", lw=0.6, alpha=0.5)
        ax.scatter([gt_c[0, 0]], [gt_c[0, 1]], marker="s", s=80, color="lime",
                   edgecolor="k", zorder=5, label="start")
        ax.set_aspect("equal"); ax.legend(fontsize=8)
        ax.set_title(f"traj {ti}: CACHED cam_pose_enc vs GT (resid/scene={resid/(scene+1e-9):.2f})")
        ax.set_xlabel("world x (m)"); ax.set_ylabel("world y (m)")
        fig.colorbar(s2, ax=ax, label="frame")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, f"cam_pose_cache_traj{ti}.png"), dpi=120)
        print(f"\nwrote {args.out}/cam_pose_cache_traj{ti}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
