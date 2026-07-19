"""Before/after equivalence check for removing the per-frame KV-cache .clone()
in SDPAAttention (attention.py:652-653).

Run TWICE with the same fixed batch:
  CLONE_AB_LABEL=before  (on the ORIGINAL code) -> saves outputs
  CLONE_AB_LABEL=after   (after the edit)        -> loads 'before' and compares

Outputs compared: encode_memory's current / depth_feat / cur_pose / goal_pose.
PASS = cos>0.9999 and relF<1e-3 and anchors identical (pure refactor => near-exact).
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src", "diffusion-policy"))
sys.path.insert(0, REPO)

from scripts.train.configs.memnav import memnav_exp_cfg
from internnav.model.basemodel.memnav.memnav_policy import MemNavPolicy, MemNavModelConfig
from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn


def main():
    label = os.environ["CLONE_AB_LABEL"]
    outdir = os.environ.get("CLONE_AB_DIR", "/scratch/lg154/tmp/clone_ab")
    os.makedirs(outdir, exist_ok=True)
    BS = int(os.environ.get("BATCH_SIZE", "6"))
    os.environ["MEMNAV_STREAM_GROUP"] = "1"
    cfg = memnav_exp_cfg
    il = cfg.il

    # CRITICAL: from_pretrained("") fails to load, so the TRAINABLE heads (retrieval,
    # depth output layer, action head) are RANDOMLY initialized. Seed BEFORE model
    # construction so 'before' and 'after' get identical random weights — otherwise
    # anchor/depth_feat/goal_pose differ due to random init, not the code change.
    import numpy as np
    torch.manual_seed(0); np.random.seed(0)
    torch.use_deterministic_algorithms(False)

    model_cfg = MemNavModelConfig(model_cfg=cfg.model_dump())
    model = MemNavPolicy.from_pretrained(pretrained_model_name_or_path="", config=model_cfg)
    model.to(model._device).eval()
    core = model.core

    ds = MemNav_Dataset(
        il.root_dir, predict_size=il.predict_size, image_size=il.image_size,
        lingbot_repo=il.lingbot_repo, feature_root=getattr(il, "feature_root", None),
        window_size=il.window_size, num_scale=il.num_scale,
    )
    # The dataset samples k stochastically per __getitem__ (np.random), so DON'T
    # re-sample in 'after' — persist the exact batch in 'before' and reload it,
    # guaranteeing byte-identical inputs for the equivalence check.
    batch_path = os.path.join(outdir, "batch.pt")
    if label == "before":
        import numpy as np
        torch.manual_seed(0); np.random.seed(0)
        picks = torch.randperm(len(ds))[:BS].tolist()
        batch = memnav_collate_fn([ds[i] for i in picks])
        torch.save(batch, batch_path)
    else:
        batch = torch.load(batch_path)
    print(f"label={label} BS={BS} k={[int(x) for x in batch['cur_steps']]}")

    keys = ["current", "depth_feat", "cur_pose", "goal_pose", "anchor_idx"]
    with torch.no_grad():
        out = core.encode_memory(batch)
    saved = {k: out[k].detach().float().cpu() for k in keys}

    path = os.path.join(outdir, f"{label}.pt")
    torch.save(saved, path)
    print(f"saved -> {path}")

    if label == "after":
        before = torch.load(os.path.join(outdir, "before.pt"))
        print("\n=== after vs before (clone removed) ===")
        worst_cos, worst_relF = 1.0, 0.0
        for k in keys:
            a = before[k].float(); b = saved[k].float()
            if k == "anchor_idx":
                print(f"  {k:11s} identical={torch.equal(a, b)}")
                continue
            relF = ((a - b).norm() / a.norm().clamp_min(1e-6)).item()
            af = a.reshape(a.shape[0], -1); bf = b.reshape(b.shape[0], -1)
            cos = torch.nn.functional.cosine_similarity(af, bf, dim=1).min().item()
            worst_cos = min(worst_cos, cos); worst_relF = max(worst_relF, relF)
            print(f"  {k:11s} cos={cos:.6f} relF={relF:.3e}")
        same = torch.equal(before["anchor_idx"], saved["anchor_idx"])
        ok = worst_cos > 0.9999 and worst_relF < 1e-3 and same
        print(f"\nWORST cos={worst_cos:.6f} relF={worst_relF:.3e}  anchors={same}")
        print("RESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
