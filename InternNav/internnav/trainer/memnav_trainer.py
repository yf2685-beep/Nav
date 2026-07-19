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

        # --- wandb panel organization ---------------------------------- #
        # HF's WandbCallback runs every logged key through rewrite_logs(), which
        # force-prefixes it with "train/". That collapses all ~19 of our panels
        # into a single "train" section. We keep the callback for run setup/finish
        # but silence its on_log, then re-emit metrics ourselves in log() under the
        # four sections below. TensorBoard/console/log_history are untouched (they
        # still flow through super().log()).
        self._wb = None
        try:
            from transformers.integrations import WandbCallback
            for cb in self.callback_handler.callbacks:
                if isinstance(cb, WandbCallback):
                    cb.on_log = lambda *a, **k: None      # no-op: we log to wandb ourselves
                    self._wb = cb._wandb                  # the wandb module handle
                    break
        except Exception as e:
            print(f"[MemNavTrainer] wandb re-sectioning disabled: {e}")

    # Map bare metric name -> "<section>/<panel>". Four sections:
    #   retrieval  — ranking + revisit-gate metrics
    #   pose       — camera-pose (relocalization) errors
    #   action     — diffusion action loss (overall + per goal category) + total loss
    #   config     — optimizer/schedule/system knobs
    # Anything unmapped falls back to "misc/<name>" so nothing is silently dropped.
    _WB_TARGET = {
        # (1) retrieval
        'retrieval_loss': 'retrieval/retrieval_loss',
        'gate_loss': 'retrieval/gate_loss',
        'gate_acc': 'retrieval/gate_acc',
        'gate_seen': 'retrieval/gate_seen',
        'gate_unseen': 'retrieval/gate_unseen',
        'gate_sep': 'retrieval/gate_sep',
        'seen_match_acc': 'retrieval/seen_match_acc',
        # (2) camera pose
        'aux_loss': 'pose/aux_loss',
        'aux_loss_shallow': 'pose/aux_loss_shallow',
        'aux_loss_deep': 'pose/aux_loss_deep',
        'rot_err_deg': 'pose/rot_err_deg',
        'rot_err_deg_shallow': 'pose/rot_err_deg_shallow',
        'rot_err_deg_deep': 'pose/rot_err_deg_deep',
        'pos_err_m': 'pose/pos_err_m',
        'pos_err_m_shallow': 'pose/pos_err_m_shallow',
        'pos_err_m_deep': 'pose/pos_err_m_deep',
        'pos_dir_err_deg': 'pose/pos_dir_err_deg',
        'pos_dir_err_deg_shallow': 'pose/pos_dir_err_deg_shallow',
        'pos_dir_err_deg_deep': 'pose/pos_dir_err_deg_deep',
        # (3) action + total loss
        'loss': 'action/total_loss',
        'action_loss': 'action/action_loss',
        'action_loss_novel': 'action/action_loss_novel',
        'action_loss_leg2': 'action/action_loss_leg2',
        'action_loss_leg3': 'action/action_loss_leg3',
        # (4) training config / system
        'learning_rate': 'config/learning_rate',
        'grad_norm': 'config/grad_norm',
        'epoch': 'config/epoch',
        'mem_alloc_gb': 'config/mem_alloc_gb',
        'mem_reserved_gb': 'config/mem_reserved_gb',
    }

    def log(self, logs, *args, **kwargs):
        # Re-emit to wandb under our four sections (bypassing rewrite_logs' flat
        # "train/" prefix), then hand the original dict to the parent for
        # console/TensorBoard/log_history. Guarded on an active run so nothing
        # breaks under report_to='tensorboard'/'none'.
        if (self.is_world_process_zero() and self._wb is not None
                and getattr(self._wb, 'run', None) is not None):
            sectioned = {}
            for k, v in logs.items():
                base = k.split('/')[-1]
                sectioned[self._WB_TARGET.get(base, f'misc/{base}')] = v
            self._wb.log(sectioned, step=self.state.global_step)
        return super().log(logs, *args, **kwargs)

    # ------------------------------------------------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        fwd = model(inputs)                                       # forward(batch) moves tensors internally

        # --- diffusion action loss (always goal-conditioned). Kept per-sample first
        # (mean over predict_size/action-dim only) so it can be bucketed by
        # novel/leg2/leg3 below; action_loss itself is unchanged (mean of per-sample
        # means == the old flat .mean() since every sample has the same [predict_size,3]
        # shape, no padding). ---
        noise = fwd["noise"]
        per_action_loss = (fwd["noise_pred"] - noise).square().mean(dim=(-2, -1))  # [B]
        action_loss = per_action_loss.mean()

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
            # --- action-loss split by goal category: overall (already `action_loss`),
            # novel (revisit==0 — goal A + any covis goal B/C that landed novel for this
            # sample's k), leg2 (revisit goal B, goal_j==0), leg3+ (revisit goal C or
            # deeper, goal_j>=1). goal_j is the dataset's fixed which-goal label, NOT
            # the same axis as the shallow/deep recall_gap split above (a leg2 sample
            # can still be a "deep" recall_gap and vice versa) — this answers "does the
            # action head specifically do worse navigating leg 3" as asked, independent
            # of pose-head accuracy.
            goal_j_t = torch.tensor(inputs["goal_js"], device=dev, dtype=torch.float32)
            is_novel_row = 1.0 - revisit
            is_leg2 = revisit * (goal_j_t == 0).float()
            is_leg3 = revisit * (goal_j_t >= 1).float()
            n_novel_raw, n_leg2_raw, n_leg3_raw = is_novel_row.sum(), is_leg2.sum(), is_leg3.sum()
            has_novel = bool(n_novel_raw.item() > 0.5)
            has_leg2 = bool(n_leg2_raw.item() > 0.5)
            has_leg3 = bool(n_leg3_raw.item() > 0.5)
            n_novel, n_leg2, n_leg3 = (n_novel_raw.clamp(min=1.0), n_leg2_raw.clamp(min=1.0),
                                       n_leg3_raw.clamp(min=1.0))
            action_loss_novel = (per_action_loss * is_novel_row).sum() / n_novel
            action_loss_leg2 = (per_action_loss * is_leg2).sum() / n_leg2
            action_loss_leg3 = (per_action_loss * is_leg3).sum() / n_leg3
            gate_prob = torch.sigmoid(gate_logit)                # [B] P(revisit)
            n_revisit_raw = revisit.sum()                         # UNclamped — 0 iff this batch has no revisit rows
            has_revisit = bool(n_revisit_raw.item() > 0.5)
            ns = n_revisit_raw.clamp(min=1.0)
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

            # --- shallow/deep split of the same revisit-only diagnostics, bucketed by
            # recall_gap = cur_step - goal_anchor_idx (frames between "now" and the matched
            # revisit frame). Answers "is the error coming from deep (long-gap) revisits" --
            # recall_gap, not leg-count, is the actual mechanism the offline sweep pinned
            # down (149-frame gap -> ~3 deg dir error; 291-frame gap -> up to 114 deg), and
            # it's confound-free (a 3-leg goal B can have a short gap; bucketing by leg
            # would blur that). goal_anchor_idx is the POST-CLAMP anchor actually fed to
            # goal_append_warm/camera_pose (memnav_policy.py encode_memory) -- not the raw
            # teacher-forced anchor_idx, which can differ when anchor < lo.
            DEEP_GAP = 200.0
            cur_step_t = torch.tensor(inputs["cur_steps"], device=dev, dtype=torch.float32)
            recall_gap = cur_step_t - fwd["goal_anchor_idx"].float()
            is_deep = revisit * (recall_gap >= DEEP_GAP).float()
            is_shallow = revisit * (recall_gap < DEEP_GAP).float()
            n_deep_raw, n_shallow_raw = is_deep.sum(), is_shallow.sum()
            has_deep = bool(n_deep_raw.item() > 0.5)
            has_shallow = bool(n_shallow_raw.item() > 0.5)
            n_deep, n_shallow = n_deep_raw.clamp(min=1.0), n_shallow_raw.clamp(min=1.0)
            pos_err_m_deep = (pos_err * is_deep).sum() / n_deep
            pos_err_m_shallow = (pos_err * is_shallow).sum() / n_shallow
            pos_dir_err_deep = (pos_dir_err_deg * is_deep).sum() / n_deep
            pos_dir_err_shallow = (pos_dir_err_deg * is_shallow).sum() / n_shallow
            rot_err_deep = (rot_err_deg * is_deep).sum() / n_deep
            rot_err_shallow = (rot_err_deg * is_shallow).sum() / n_shallow
            aux_loss_deep = (per * is_deep).sum() / n_deep
            aux_loss_shallow = (per * is_shallow).sum() / n_shallow
        outputs = dict(loss=loss, action_loss=action_loss,
                       retrieval_loss=rank_loss, gate_loss=gate_loss, aux_loss=aux_loss,
                       gate_seen=gate_seen, gate_unseen=gate_unseen, gate_sep=gate_sep,
                       gate_acc=gate_acc, seen_match_acc=seen_match, rot_err_deg=rot_err,
                       pos_err_m=pos_err_m, pos_dir_err_deg=pos_dir_err,
                       action_loss_novel=action_loss_novel, action_loss_leg2=action_loss_leg2,
                       action_loss_leg3=action_loss_leg3)
        if (dist.get_rank() if dist.is_initialized() else 0) == 0:
            revisit_part = (f"pos_err={pos_err_m.item():.2f}m dir_err={pos_dir_err.item():.1f}deg "
                            f"rot_err={rot_err.item():.1f}deg aux={aux_loss.item():.4f} "
                            f"match={seen_match.item():.2f}" if has_revisit else
                            "no revisit rows this batch (pos/rot/aux/match skipped)")
            depth_part = ' '.join(filter(None, [
                f"shallow(n={int(n_shallow_raw.item())}): pos={pos_err_m_shallow.item():.2f}m "
                f"rot={rot_err_shallow.item():.1f}deg" if has_shallow else '',
                f"deep(n={int(n_deep_raw.item())}): pos={pos_err_m_deep.item():.2f}m "
                f"rot={rot_err_deep.item():.1f}deg" if has_deep else '',
            ]))
            action_part = ' '.join(filter(None, [
                f"novel(n={int(n_novel_raw.item())})={action_loss_novel.item():.4f}" if has_novel else '',
                f"leg2(n={int(n_leg2_raw.item())})={action_loss_leg2.item():.4f}" if has_leg2 else '',
                f"leg3(n={int(n_leg3_raw.item())})={action_loss_leg3.item():.4f}" if has_leg3 else '',
            ]))
            print(f"[Step {self.state.global_step}] loss={loss.item():.4f} act={action_loss.item():.4f} "
                  f"rank={rank_loss.item():.4f} gate={gate_loss.item():.4f} | "
                  f"gate seen={gate_seen.item():.2f} unseen={gate_unseen.item():.2f} sep={gate_sep.item():+.2f} "
                  f"acc={gate_acc.item():.2f} | {revisit_part}"
                  + (f" | {depth_part}" if depth_part else "")
                  + (f" | act: {action_part}" if action_part else ""))

        # Per-component metrics → wandb/tb. self.log is rank-0-only inside HF Trainer;
        # gate by logging_steps to match train/loss cadence and avoid extra .item() syncs.
        if self.state.global_step % self.args.logging_steps == 0:
            # Bare metric names; log() re-sections them for wandb (see _WB_TARGET).
            log_payload = {
                'action_loss': action_loss.item(),
                'retrieval_loss': rank_loss.item(),
                'gate_loss': gate_loss.item(),
                'gate_acc': gate_acc.item(),
                'gate_seen': gate_seen.item(),
                'gate_unseen': gate_unseen.item(),
                'gate_sep': gate_sep.item(),
            }
            # revisit-only metrics (aux_loss, rot_err_deg, pos_err_m, pos_dir_err_deg,
            # seen_match_acc): only logged when this batch actually contains revisit rows
            # -- otherwise the masked-mean formula silently reports 0 (0/clamp(0,min=1)),
            # and logging that 0 as a real data point would dilute/distort the wandb curve
            # instead of giving a clean signal of revisit-case accuracy specifically.
            if has_revisit:
                log_payload.update({
                    'aux_loss': aux_loss.item(),
                    'seen_match_acc': seen_match.item(),
                    'rot_err_deg': rot_err.item(),
                    'pos_err_m': pos_err_m.item(),
                    'pos_dir_err_deg': pos_dir_err.item(),
                })
            # same revisit-only diagnostics, split by recall_gap (see above) -- each half
            # gated independently since a bucket can be empty even when has_revisit is True
            # (a batch with one shallow revisit row and zero deep ones, or vice versa).
            if has_shallow:
                log_payload.update({
                    'aux_loss_shallow': aux_loss_shallow.item(),
                    'rot_err_deg_shallow': rot_err_shallow.item(),
                    'pos_err_m_shallow': pos_err_m_shallow.item(),
                    'pos_dir_err_deg_shallow': pos_dir_err_shallow.item(),
                })
            if has_deep:
                log_payload.update({
                    'aux_loss_deep': aux_loss_deep.item(),
                    'rot_err_deg_deep': rot_err_deep.item(),
                    'pos_err_m_deep': pos_err_m_deep.item(),
                    'pos_dir_err_deg_deep': pos_dir_err_deep.item(),
                })
            # action loss split by goal category (see is_novel_row/is_leg2/is_leg3 above) --
            # goal_j-based (which goal), NOT the same axis as the shallow/deep recall_gap
            # split. Each gated independently, same 0-dilution guard as everything above.
            if has_novel:
                log_payload['action_loss_novel'] = action_loss_novel.item()
            if has_leg2:
                log_payload['action_loss_leg2'] = action_loss_leg2.item()
            if has_leg3:
                log_payload['action_loss_leg3'] = action_loss_leg3.item()
            if dev.type == 'cuda':
                # Peak since previous logging_step, in GiB. Reset right after so the
                # next window measures its own peak — otherwise max_ stays monotone.
                alloc = torch.cuda.max_memory_allocated(dev) / 2**30
                reserved = torch.cuda.max_memory_reserved(dev) / 2**30
                log_payload['mem_alloc_gb'] = alloc
                log_payload['mem_reserved_gb'] = reserved
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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Resume-time weight load. save_model writes only the trainable heads to
        memnav.ckpt (no standard HF weight file), so HF's default loader can't find
        one. Load memnav.ckpt non-strictly — the missing keys are exactly the frozen
        LingBot backbone, which was already loaded at model construction. HF still
        restores optimizer/scheduler/global_step/RNG from the checkpoint separately."""
        if model is None:
            model = self.model
        m = model.module if hasattr(model, "module") else model
        ckpt = os.path.join(resume_from_checkpoint, "memnav.ckpt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"resume checkpoint has no memnav.ckpt: {ckpt}")
        sd = torch.load(ckpt, map_location="cpu")
        inc = m.load_state_dict(sd, strict=False)
        unexpected = list(inc.unexpected_keys)
        frozen_missing = [k for k in inc.missing_keys if "lingbot." not in k]
        print(f"[resume] loaded memnav.ckpt: {len(sd)} trainable tensors; "
              f"unexpected={len(unexpected)} non-lingbot-missing={len(frozen_missing)}")
        if unexpected or frozen_missing:
            print(f"[resume] WARN unexpected={unexpected[:5]} non_lingbot_missing={frozen_missing[:5]}")
