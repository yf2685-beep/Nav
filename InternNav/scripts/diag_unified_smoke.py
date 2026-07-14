"""Smoke test for the unified revisit rule + decoupled retrieval loss.
Part A (CPU, masks only): dataset invariants + rank/gate loss math.
Part B (--gpu): full MemNavNet forward + exact trainer loss + backward.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    root = os.environ["MEMNAV_ROOT_DIR"]; feat = os.environ["MEMNAV_FEATURE_ROOT"]
    repo = os.environ["LINGBOT_REPO"]
    W = int(os.environ.get("MEMNAV_WINDOW", 32)); NS = int(os.environ.get("MEMNAV_NUM_SCALE", 8))
    ds = MemNav_Dataset(root, predict_size=24, image_size=518, lingbot_repo=repo,
                        feature_root=feat, window_size=W, num_scale=NS)
    t = ds.exclude_recent
    print(f"exclude_recent={t}  n_samples={len(ds.samples)}")

    # ---- Part A: label invariants over _build_label (no images) ----
    rng = np.random.default_rng(0)
    sel = rng.integers(0, len(ds.samples), size=min(args.n, len(ds.samples)))
    n_rev = n_nov = n_A = n_A_rev = 0
    for i in sel:
        s = ds.samples[int(i)]
        klo, khi = int(s["k_lo"]), int(s["k_hi"])
        k = int(rng.integers(klo, khi + 1))
        pos, neg, cand, nullp = ds._build_label(s, k)
        am = int(s["amargin"]); hi = k - t
        idx = np.arange(k + 1)
        exp_cand = (idx >= am) & (idx <= hi)
        assert pos.shape[0] == neg.shape[0] == cand.shape[0] == k + 1, "length mismatch"
        assert np.array_equal(cand, exp_cand), f"cand != [amargin..k-t] (k={k})"
        assert not (pos & neg).any(), "pos/neg overlap"
        assert (pos & ~cand).sum() == 0, "pos outside candidate region"
        assert (neg & ~cand).sum() == 0, "neg outside candidate region"
        assert nullp == (pos.sum() == 0), "null_pos != (no positive)"
        if not s["has_covis"]:                       # goalA: always novel, neg == cand
            n_A += 1
            assert pos.sum() == 0, "goalA has a positive"
            assert np.array_equal(neg, cand), "goalA neg != cand"
            if not nullp:
                n_A_rev += 1
        n_rev += int(not nullp); n_nov += int(nullp)
    assert n_A_rev == 0, "some goalA sample was revisit (should be impossible)"
    print(f"[A] invariants OK over {len(sel)} draws | revisit={n_rev} novel={n_nov} "
          f"(goalA={n_A}, all novel: {n_A_rev == 0})")

    # ---- full __getitem__ + collate on a few items (exercises cand_mask plumbing) ----
    items = [ds[int(i)] for i in sel[:3]]
    batch = memnav_collate_fn(items)
    for key in ("batch_cand_mask", "batch_pos_mask", "batch_neg_mask", "batch_is_revisit"):
        assert key in batch, f"missing {key}"
    cm, pm, nm, mm = (batch[k] for k in ("batch_cand_mask", "batch_pos_mask",
                                         "batch_neg_mask", "batch_mem_mask"))
    assert (pm & ~cm).sum() == 0 and (nm & ~cm).sum() == 0 and (cm & ~mm).sum() == 0
    print(f"[A] collate OK | cand={cm.shape} pos⊆cand⊆mem verified")

    # ---- decoupled loss math on random logits over many rows ----
    rows = []
    for i in rng.integers(0, len(ds.samples), size=128):
        s = ds.samples[int(i)]
        k = int(rng.integers(int(s["k_lo"]), int(s["k_hi"]) + 1))
        p, n, c, nu = ds._build_label(s, k)
        rows.append((p, n, nu))
    Lm = max(r[0].shape[0] for r in rows); Bn = len(rows)
    pos = torch.zeros(Bn, Lm, dtype=torch.bool); neg = torch.zeros(Bn, Lm, dtype=torch.bool)
    is_rev = torch.zeros(Bn)
    for j, (p, n, nu) in enumerate(rows):
        pos[j, : p.shape[0]] = torch.from_numpy(p); neg[j, : n.shape[0]] = torch.from_numpy(n)
        is_rev[j] = float(not nu)
    logits = torch.randn(Bn, Lm)
    gate_logit = torch.randn(Bn, requires_grad=True)
    NEG_INF = torch.finfo(logits.dtype).min
    lse_pn = logits.masked_fill(~(pos | neg), NEG_INF).logsumexp(-1)
    lse_p = logits.masked_fill(~pos, NEG_INF).logsumexp(-1)
    rank_rows = pos.any(-1) & neg.any(-1)
    rank_loss = ((lse_pn - lse_p) * rank_rows).sum() / rank_rows.sum().clamp(min=1.0)
    pw = ((1 - is_rev).sum() / is_rev.sum().clamp(min=1)).clamp(0.1, 10.0)
    gate_loss = F.binary_cross_entropy_with_logits(gate_logit, is_rev, pos_weight=pw)
    assert torch.isfinite(rank_loss) and rank_loss >= -1e-4, "bad rank loss"
    assert torch.isfinite(gate_loss), "bad gate loss"
    print(f"[A] loss math OK | rank={rank_loss.item():.4f} (rows {int(rank_rows.sum())}/{Bn}) "
          f"gate_bce={gate_loss.item():.4f} pos_weight={pw.item():.2f}")

    if not args.gpu:
        print("OK (Part A). Re-run with --gpu for the full forward+loss+backward.")
        return

    # ---- Part B: full net forward + exact trainer loss + backward ----
    from internnav.model.basemodel.memnav.memnav_policy import MemNavNet
    wts = os.environ["LINGBOT_WEIGHTS"]; MFN = int(os.environ.get("MEMNAV_MAX_FRAME_NUM", 2048))
    dev = "cuda:0"
    # revisit/novel is resolved per-k INSIDE __getitem__ (random k), so we must DRAW items
    # until they land revisit — else the batch is all-novel and the teacher-forcing path
    # (anchor->positive, aux grad) is never exercised. Build a mixed batch: >=1 revisit + 1 novel.
    covis_ids = [int(i) for i, s in enumerate(ds.samples) if s["has_covis"]]
    goalA_ids = [int(i) for i, s in enumerate(ds.samples) if not s["has_covis"]]

    def draw(idx, want_rev, tries=12):
        for _ in range(tries):
            it = ds[idx]
            if bool(it["is_revisit"]) == want_rev:
                return it
        return None

    rev_items = []
    for i in covis_ids:
        it = draw(i, want_rev=True)
        if it is not None:
            rev_items.append(it)
        if len(rev_items) >= 3:
            break
    assert rev_items, "could not draw any revisit item in Part B (check exclude_recent / covis thresholds)"
    nov_item = draw(goalA_ids[0], want_rev=False) or ds[goalA_ids[0]]   # goalA is always novel
    items = rev_items + [nov_item]
    batch = memnav_collate_fn(items)
    print(f"[B] batch n={len(items)} is_revisit={batch['batch_is_revisit'].tolist()} "
          f"cand_per_row={batch['batch_cand_mask'].sum(1).tolist()}")
    net = MemNavNet(token_dim=384, heads=8, predict_size=24, temporal_depth=8, num_diffusion_iters=10,
                    lingbot_kwargs=dict(lingbot_repo=repo, weights=wts, window=W, num_scale=NS, max_frame_num=MFN),
                    device=dev).to(dev)
    net.train()
    fwd = net(batch)
    for k in ("ret_logits", "gate_logit", "revisit_gate", "aux_pose"):
        v = fwd[k]; print(f"  {k}: {tuple(v.shape)} finite={bool(torch.isfinite(v[torch.isfinite(v)]).all())}")

    # --- teacher-forcing: train anchor must land on a GT positive for every revisit row ---
    posB = batch["batch_pos_mask"].to(dev).bool()
    revB = batch["batch_is_revisit"].to(dev).bool()
    anc = fwd["anchor_idx"]; mi = fwd["match_idx"]
    on_pos = posB.gather(1, anc[:, None]).squeeze(1)          # anchor is a positive?
    assert bool((on_pos | ~revB).all()), "train anchor off-positive on a revisit row"
    print(f"[B] TRAIN anchor: revisit rows={int(revB.sum())} anchor∈pos={int((on_pos & revB).sum())} "
          f"(match∈pos={int((posB.gather(1, mi[:,None]).squeeze(1) & revB).sum())})")
    # --- eval path must fall back to the live match_idx ---
    net.eval()
    with torch.no_grad():
        fe = net(batch)
    assert bool((fe["anchor_idx"] == fe["match_idx"]).all()), "eval anchor != match_idx"
    print("[B] EVAL anchor == match_idx (fallback OK)")
    net.train()
    # exact trainer loss
    noise = fwd["noise"]
    action_loss = 0.5 * (fwd["noise_ng"] - noise).square().mean() + 0.5 * (fwd["noise_mg"] - noise).square().mean()
    rl = fwd["ret_logits"]; gl = fwd["gate_logit"]
    pos = batch["batch_pos_mask"].to(dev).bool(); neg = batch["batch_neg_mask"].to(dev).bool()
    is_rev = batch["batch_is_revisit"].to(dev)
    NI = torch.finfo(rl.dtype).min
    rr = pos.any(-1) & neg.any(-1)
    rank = (((rl.masked_fill(~(pos | neg), NI).logsumexp(-1) - rl.masked_fill(~pos, NI).logsumexp(-1)) * rr).sum()
            / rr.sum().clamp(min=1))
    pw = ((1 - is_rev).sum() / is_rev.sum().clamp(min=1)).clamp(0.1, 10.0)
    gate = F.binary_cross_entropy_with_logits(gl, is_rev, pos_weight=pw)
    per = (fwd["aux_pose"] - batch["batch_goal_rel_pose"].to(dev)).square().mean(-1)
    aux = (per * is_rev).sum() / is_rev.sum().clamp(min=1)
    loss = action_loss + rank + gate + 0.5 * aux
    print(f"[B] loss={loss.item():.4f} act={action_loss.item():.4f} rank={rank.item():.4f} "
          f"gate={gate.item():.4f} aux={aux.item():.4f}")
    assert torch.isfinite(loss), "non-finite loss"
    loss.backward()
    for nm_ in ("gate_a", "gate_b", "log_temp"):
        g = getattr(net.retrieval, nm_).grad
        print(f"  retrieval.{nm_}.grad = {None if g is None else round(g.item(), 6)}")
        assert g is not None, f"no grad to retrieval.{nm_}"
        assert torch.isfinite(g), f"retrieval.{nm_}.grad is not finite ({g.item()})"
    # aux-pose head must receive gradient (teacher-forced anchor -> learnable aux target)
    ag = net.revisit_merge.aux_pose_head[-1].weight.grad
    gn = None if ag is None else round(ag.norm().item(), 6)
    print(f"  revisit_merge.aux_pose_head.grad_norm = {gn}")
    if bool(revB.any()):
        assert ag is not None and torch.isfinite(ag).all() and ag.abs().sum() > 0, "no/!finite aux grad"
    else:
        print("  (no revisit rows in this batch -> aux weight 0, grad check skipped)")
    print("OK (Part B): forward + decoupled loss + backward, grads reach gate + temp + aux head.")


if __name__ == "__main__":
    main()
