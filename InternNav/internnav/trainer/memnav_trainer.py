import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
from internnav.trainer.base import BaseTrainer


class MemNavTrainer(BaseTrainer):
    """memnav: frozen LingBot front-end + trainable retrieval / novel / current_state /
    revisit / DDPM decoder. Loss = action (ε-MSE, always goal-conditioned) + retrieval-CE
    + aux-pose. No classifier-free no-goal branch (see memnav_policy.py docstring — dropped,
    never benchmarked, and label-ambiguous for our U-turn-containing episodes).
    No critic — collision is checked geometrically from the point map at eval."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.w_retr = getattr(config.il, "w_retrieval", 1.0)     # ranking InfoNCE
        self.w_gate = getattr(config.il, "w_gate", 1.0)          # revisit/novel BCE
        self.w_aux = getattr(config.il, "w_aux_pose", 0.5)
        self.model_device = (self.model.module if hasattr(self.model, "module") else self.model).device
        # Rotation-specific local-frame correction for the rotation-accuracy diagnostic
        # (compute_loss, R_rel vs batch_goal_rel_rotation). NOT RevisitMerge.aux_pose_head's
        # _R_CONV/_SCALE — empirically, translation and rotation need DIFFERENT corrections;
        # reusing translation's R_conv via conjugation (R_conv @ R_rel @ R_conv^T) was tried
        # first and gave ~180° error even on a case with ~3° translation error, i.e. it was
        # putting the rotation in the wrong plane entirely, not just the wrong angle. Fit
        # separately (Kabsch on rotation-AXIS pairs extracted from real (R_rel, GT) samples,
        # not reused from the translation fit): a clean 180° rotation about Z (fitted matrix
        # was within 0.7° of this exact diag), median residual ~1-5° depending on trajectory
        # difficulty (same VO-accuracy dependence as everything else in this pipeline).
        self._C_rot = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    # ------------------------------------------------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        fwd = model(inputs)                                       # forward(batch) moves tensors internally

        # --- diffusion action loss (always goal-conditioned) ---
        noise = fwd["noise"]
        action_loss = (fwd["noise_pred"] - noise).square().mean()

        # --- retrieval: DECOUPLED ranking (InfoNCE) + revisit gate (BCE) ---
        # A joint softmax with a null slot collapses to always-null (the easy shortcut),
        # so the two jobs are split: (a) InfoNCE ranks the true co-visible frame above the
        # other candidates on REVISIT rows; (b) an affine-on-max-cosine gate (in the head)
        # is trained by BCE to decide revisit vs novel. The candidate set E(k) already
        # excludes the recent approach window, which is what makes both signals separable.
        ret_logits = fwd["ret_logits"]                           # [B, L] cos/temp over candidates
        gate_logit = fwd["gate_logit"]                           # [B] pre-sigmoid revisit logit
        pos = inputs["batch_pos_mask"].to(dev).bool()            # [B, L]
        neg = inputs["batch_neg_mask"].to(dev).bool()            # [B, L]
        is_rev = inputs["batch_is_revisit"].to(dev)              # [B] float (1=revisit, 0=novel)
        NEG_INF = torch.finfo(ret_logits.dtype).min
        # (a) ranking: -log Σ_pos e^s / Σ_{pos∪neg} e^s, over revisit rows carrying a negative
        lse_pn = ret_logits.masked_fill(~(pos | neg), NEG_INF).logsumexp(-1)
        lse_p = ret_logits.masked_fill(~pos, NEG_INF).logsumexp(-1)
        rank_rows = pos.any(-1) & neg.any(-1)                    # [B] revisit rows w/ contrastive signal
        rank_loss = ((lse_pn - lse_p) * rank_rows).sum() / rank_rows.sum().clamp(min=1.0)
        # (b) gate BCE over ALL rows; pos_weight offsets the novel-heavy class mix
        n_rev = is_rev.sum()
        pos_weight = ((1.0 - is_rev).sum() / n_rev.clamp(min=1.0)).clamp(0.1, 10.0)
        gate_loss = F.binary_cross_entropy_with_logits(gate_logit, is_rev, pos_weight=pos_weight)

        # --- aux pose (x,y only — θ dropped, see RevisitMerge docstring): MSE on REVISIT
        # rows only (relocalization branch). aux_pose_head is FROZEN (pre-calibrated, not
        # trainable — cur_pose/goal_pose come from the frozen camera head under no_grad,
        # so this contributes zero gradient today); kept in the loss sum as a no-op so a
        # future LoRA fine-tune of the frozen branch can unfreeze aux_pose_head and have
        # this term start training with no other code changes.
        gt_pose = inputs["batch_goal_rel_pose"][..., :2].to(dev)  # [B,2]
        revisit = is_rev                                         # 1 = goal is in memory
        per = (fwd["aux_pose"] - gt_pose).square().mean(-1)      # [B]
        aux_loss = (per * revisit).sum() / revisit.sum().clamp(min=1.0)
        R_rel = fwd["R_rel"]                                      # [B,3,3] predicted relative rotation

        loss = (action_loss + self.w_retr * rank_loss
                + self.w_gate * gate_loss + self.w_aux * aux_loss)

        with torch.no_grad():
            gate_prob = torch.sigmoid(gate_logit)                # [B] P(revisit)
            ns = revisit.sum().clamp(min=1.0)
            nu = (1.0 - revisit).sum().clamp(min=1.0)
            gate_seen = (gate_prob * revisit).sum() / ns          # → 1 (visited)
            gate_unseen = (gate_prob * (1.0 - revisit)).sum() / nu  # → 0 (novel)
            gate_sep = gate_seen - gate_unseen                    # → large + (the separation)
            gate_acc = ((gate_prob > 0.5).float() == revisit).float().mean()
            pred = ret_logits.argmax(-1)                          # [B] best candidate frame
            hit = pos.gather(1, pred[:, None]).squeeze(1).float()
            seen_match = (hit * revisit).sum() / ns               # retrieved the true frame (revisit)

            # --- position diagnostics, plain-units companions to aux_loss (MSE): pos_err_m is
            # the actual metric distance (not squared) between the calibrated aux_pose and GT
            # (x,y); pos_dir_err_deg is the BEARING error only (angle between the predicted and
            # GT relative-position vectors, scale-invariant) — separates "direction is right but
            # scale/magnitude is off" from "direction itself is wrong". Neither is comparable to
            # rot_err_deg below: these are about WHERE the goal is, rot_err_deg is about the
            # camera's relative ORIENTATION — a sample can have one right and the other wrong. ---
            pos_err = torch.linalg.norm(fwd["aux_pose"] - gt_pose, dim=-1)          # [B] meters
            pos_dir_cos = ((fwd["aux_pose"] * gt_pose).sum(-1)
                          / (torch.linalg.norm(fwd["aux_pose"], dim=-1)
                             * torch.linalg.norm(gt_pose, dim=-1) + 1e-9)).clamp(-1, 1)
            pos_dir_err_deg = torch.rad2deg(torch.arccos(pos_dir_cos))              # [B]
            pos_err_m = (pos_err * revisit).sum() / ns
            pos_dir_err = (pos_dir_err_deg * revisit).sum() / ns

            # --- rotation-accuracy diagnostic (pure logging, no loss/gradient — R_rel comes
            # from the frozen camera head under no_grad same as t_rel; see RevisitMerge). Not
            # comparable to GT theta (path-tangent, unrelated) — this compares the actual
            # relative CAMERA rotation against real GT extrinsics/render orientation
            # (batch_goal_rel_rotation). Conjugate, not left-multiply, to change basis for a
            # rotation matrix rather than a translation vector. ---
            C_rot = self._C_rot.to(dev, R_rel.dtype)
            R_rel_conv = C_rot @ R_rel @ C_rot.transpose(-1, -2)      # [B,3,3]
            gt_rot = inputs["batch_goal_rel_rotation"].to(dev)          # [B,3,3]
            cos_ang = (((R_rel_conv * gt_rot).sum(dim=(-2, -1)) - 1) / 2).clamp(-1, 1)
            rot_err_deg = torch.rad2deg(torch.arccos(cos_ang))          # [B]
            rot_err = (rot_err_deg * revisit).sum() / ns                # masked mean, revisit rows only
        outputs = dict(loss=loss, action_loss=action_loss,
                       retrieval_loss=rank_loss, gate_loss=gate_loss, aux_loss=aux_loss,
                       gate_seen=gate_seen, gate_unseen=gate_unseen, gate_sep=gate_sep,
                       gate_acc=gate_acc, seen_match_acc=seen_match, rot_err_deg=rot_err,
                       pos_err_m=pos_err_m, pos_dir_err_deg=pos_dir_err)
        if (dist.get_rank() if dist.is_initialized() else 0) == 0:
            print(f"[Step {self.state.global_step}] loss={loss.item():.4f} act={action_loss.item():.4f} "
                  f"rank={rank_loss.item():.4f} gate={gate_loss.item():.4f} aux={aux_loss.item():.4f} | "
                  f"gate seen={gate_seen.item():.2f} unseen={gate_unseen.item():.2f} sep={gate_sep.item():+.2f} "
                  f"acc={gate_acc.item():.2f} | match seen={seen_match.item():.2f} | "
                  f"pos_err={pos_err_m.item():.2f}m dir_err={pos_dir_err.item():.1f}deg "
                  f"rot_err={rot_err.item():.1f}deg")

        # Per-component metrics → wandb/tb. self.log is rank-0-only inside HF Trainer;
        # gate by logging_steps to match train/loss cadence and avoid extra .item() syncs.
        if self.state.global_step % self.args.logging_steps == 0:
            log_payload = {
                'train/action_loss': action_loss.item(),
                'train/retrieval_loss': rank_loss.item(),
                'train/gate_loss': gate_loss.item(),
                'train/aux_loss': aux_loss.item(),
                'train/gate_acc': gate_acc.item(),
                'train/gate_seen': gate_seen.item(),
                'train/gate_unseen': gate_unseen.item(),
                'train/gate_sep': gate_sep.item(),
                'train/seen_match_acc': seen_match.item(),
                'train/rot_err_deg': rot_err.item(),
                'train/pos_err_m': pos_err_m.item(),
                'train/pos_dir_err_deg': pos_dir_err.item(),
            }
            if dev.type == 'cuda':
                # Peak since previous logging_step, in GiB. Reset right after so the
                # next window measures its own peak — otherwise max_ stays monotone.
                alloc = torch.cuda.max_memory_allocated(dev) / 2**30
                reserved = torch.cuda.max_memory_reserved(dev) / 2**30
                log_payload['train/mem_alloc_gb'] = alloc
                log_payload['train/mem_reserved_gb'] = reserved
                if (dist.get_rank() if dist.is_initialized() else 0) == 0:
                    print(f"[Step {self.state.global_step}] mem peak "
                          f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB")
                torch.cuda.reset_peak_memory_stats(dev)
            self.log(log_payload)

        return (loss, outputs) if return_outputs else loss

    # ------------------------------------------------------------------ #
    def create_optimizer(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        lr = getattr(self.config.il, "lr", 1e-4)
        m = self.model.module if hasattr(self.model, "module") else self.model
        params = [p for p in m.parameters() if p.requires_grad]       # frozen LingBot excluded
        self.optimizer = torch.optim.Adam(params, lr=lr)
        if rank == 0:
            n = sum(p.numel() for p in params)
            print(f"[Rank 0] Adam lr={lr}; trainable params: {n:,} ({len(params)} tensors)")
        return self.optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000)
        return self.lr_scheduler

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(self.optimizer, num_training_steps)
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1234)
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator or memnav_collate_fn,
        )

    def save_model(self, output_dir, state_dict=None, **kwargs):
        """Save only the trainable heads (skip the frozen LingBot — reloaded separately at eval)."""
        m = self.model.module if hasattr(self.model, "module") else self.model
        sd = {k: v for k, v in m.state_dict().items() if "lingbot." not in k}
        os.makedirs(output_dir, exist_ok=True)
        torch.save(sd, os.path.join(output_dir, "memnav.ckpt"))
        print(f"Saved {len(sd)} trainable tensors to {output_dir}/memnav.ckpt")
