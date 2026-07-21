"""Quantitative demonstration of MemNav's core contribution: implicit memory =>
seen >> unseen. Uses ONLY the trained RetrievalHead on precomputed dino_cls (fast, CPU):

  for many (current step k, goal frame g):
    SEEN   : g < k  (goal was observed earlier) -> memory should FIRE (high revisit_gate)
    UNSEEN : g > k  (goal never observed)        -> memory should stay quiet (low gate, -> null)

Aggregates revisit_gate over thousands of pairs across all pilot trajectories and reports
the seen-vs-unseen separation (the memory signal) + a histogram.
"""
import argparse, glob, os, sys
sys.path.insert(0, "/home/nyuair/yuxuan/1 robot navigation/Nav/InternNav")
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from internnav.model.basemodel.memnav.memnav_policy import RetrievalHead

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/tmp/claude-1000/-home-nyuair-yuxuan/728eec84-0d5f-445c-b895-a46be90f5482/scratchpad/memnav_ckpt/memnav_step150.ckpt")
ap.add_argument("--root", default="/home/nyuair/data-001/InternData-N1/v0.1-mini/vln_n1/traj_data_navdp/matterport3d_d435i")
ap.add_argument("--gap", type=int, default=8, help="min |k-g| gap (matches training)")
ap.add_argument("--per_traj", type=int, default=40, help="seen+unseen samples per trajectory")
ap.add_argument("--lo", type=int, default=15, help="min current step (num_scale+window-1)")
ap.add_argument("--out", default="/tmp/claude-1000/-home-nyuair-yuxuan/728eec84-0d5f-445c-b895-a46be90f5482/scratchpad/seen_unseen_stats.png")
args = ap.parse_args()

# ---- load ONLY the retrieval head from the checkpoint (tiny, CPU) ----
rh = RetrievalHead(dino_dim=1024)
sd = torch.load(args.ckpt, map_location="cpu")
rsd = {k[len("retrieval."):]: v for k, v in sd.items() if k.startswith("retrieval.")}
missing, unexpected = rh.load_state_dict(rsd, strict=False)
rh.eval()
print(f"loaded RetrievalHead: {len(rsd)} tensors (missing={len(missing)} unexpected={len(unexpected)})")

caches = sorted(glob.glob(os.path.join(args.root, "*/*/videos/chunk-000/lingbot_cache.npz")))
print(f"{len(caches)} trajectories with caches")

rng = np.random.RandomState(0)
gate_seen, gate_unseen = [], []
seen_hit_real, unseen_hit_null = 0, 0   # seen: argmax is a real frame; unseen: argmax is null
n_seen = n_unseen = 0

@torch.no_grad()
def gate_of(dino_cls, k, g):
    mem = torch.tensor(dino_cls[:k + 1], dtype=torch.float32)[None]   # [1,k+1,1024]
    goal = torch.tensor(dino_cls[g], dtype=torch.float32)[None]        # [1,1024]
    mask = torch.ones(1, k + 1, dtype=torch.bool)
    match_idx, gate, logits = rh(goal, mem, mask)
    null_idx = logits.shape[1] - 1
    argmax = int(logits[0].argmax().item())
    return float(gate[0]), (argmax != null_idx)   # gate, chose_real_frame

for cpath in caches:
    with np.load(cpath) as d:
        dino_cls = d["dino_cls"].astype(np.float32)
    T = dino_cls.shape[0]
    if T < args.lo + args.gap + 2:
        continue
    for _ in range(args.per_traj // 2):
        # SEEN: k in [lo, T-1], g in [0, k-gap]
        k = rng.randint(args.lo, T)
        if k - args.gap < 0:
            continue
        g = rng.randint(0, k - args.gap + 1)
        gt, real = gate_of(dino_cls, k, g); gate_seen.append(gt); n_seen += 1; seen_hit_real += int(real)
        # UNSEEN: k in [lo, T-1-gap], g in [k+gap, T-1]
        k2 = rng.randint(args.lo, max(args.lo + 1, T - args.gap))
        if k2 + args.gap > T - 1:
            continue
        g2 = rng.randint(k2 + args.gap, T)
        gt2, real2 = gate_of(dino_cls, k2, g2); gate_unseen.append(gt2); n_unseen += 1; unseen_hit_null += int(not real2)

gs, gu = np.array(gate_seen), np.array(gate_unseen)
# separation metric: AUC of gate as a seen/unseen classifier
labels = np.r_[np.ones_like(gs), np.zeros_like(gu)]
scores = np.r_[gs, gu]
order = np.argsort(scores)
ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
n_pos, n_neg = len(gs), len(gu)
auc = (ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

print("\n================ MemNav core mechanism: seen vs unseen ================")
print(f"  SEEN   goals (in memory): n={n_seen}  mean revisit_gate = {gs.mean():.3f} ± {gs.std():.3f}")
print(f"  UNSEEN goals (future):    n={n_unseen}  mean revisit_gate = {gu.mean():.3f} ± {gu.std():.3f}")
print(f"  separation (seen-unseen mean gate) = {gs.mean()-gu.mean():+.3f}")
print(f"  gate-as-classifier AUC (seen vs unseen) = {auc:.3f}   (0.5=no memory, 1.0=perfect)")
print(f"  seen -> chose a real memory frame: {100*seen_hit_real/max(n_seen,1):.1f}%")
print(f"  unseen -> correctly chose NULL:    {100*unseen_hit_null/max(n_unseen,1):.1f}%")

plt.figure(figsize=(7, 4.5))
bins = np.linspace(0, 1, 26)
plt.hist(gu, bins=bins, alpha=0.6, label=f"UNSEEN goal (mean {gu.mean():.2f})", color="tab:red", density=True)
plt.hist(gs, bins=bins, alpha=0.6, label=f"SEEN goal (mean {gs.mean():.2f})", color="tab:green", density=True)
plt.axvline(gu.mean(), color="tab:red", ls="--"); plt.axvline(gs.mean(), color="tab:green", ls="--")
plt.xlabel("revisit_gate  (P: goal is in memory)"); plt.ylabel("density")
plt.title(f"MemNav memory: seen vs unseen goals  (AUC={auc:.3f}, sep={gs.mean()-gu.mean():+.2f})")
plt.legend(); plt.tight_layout(); plt.savefig(args.out, dpi=120)
print(f"\nsaved {args.out}")
print("RESULT: STATS DONE")
