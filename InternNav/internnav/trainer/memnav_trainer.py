import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from internnav.dataset.memnav_dataset_lerobot import memnav_collate_fn
from internnav.trainer.base import BaseTrainer


class MemNavTrainer(BaseTrainer):
    """memnav: frozen LingBot front-end + trainable retrieval / novel / current_state /
    revisit / DDPM decoder. Loss = 0.5·ng + 0.5·mg (ε-MSE) + retrieval-CE + aux-pose.
    No critic — collision is checked geometrically from the point map at eval."""

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.w_retr = getattr(config.il, "w_retrieval", 1.0)
        self.w_aux = getattr(config.il, "w_aux_pose", 0.5)
        self.model_device = (self.model.module if hasattr(self.model, "module") else self.model).device
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Model device: {self.model_device}")

    # ------------------------------------------------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        dev = next(model.parameters()).device
        fwd = model(inputs)                                       # forward(batch) moves tensors internally

        # --- diffusion action loss (classifier-free ng + mg) ---
        noise = fwd["noise"]
        ng_loss = (fwd["noise_ng"] - noise).square().mean()
        mg_loss = (fwd["noise_mg"] - noise).square().mean()
        action_loss = 0.5 * ng_loss + 0.5 * mg_loss

        # --- retrieval loss: windowed soft-label CE (seen) + hard null (unseen) + recall bias ---
        # Adjacent trajectory frames are near-identical, so a hard exact-index CE is nearly
        # unlearnable (exact-match sat at 0.00) and pushes the head toward null. Instead we credit
        # a neighbourhood ±W of the true goal frame (triangular soft target), and add a recall
        # penalty that pushes SEEN samples off the null so the memory branch actually fires
        # ("更容易采取之前的图片"). Unseen goals keep a hard null target.
        logits = fwd["ret_logits"]                               # [B, L+1] (last = null)
        is_seen = inputs["batch_is_seen"].to(dev).bool()         # [B]
        mem_mask = inputs["batch_mem_mask"].to(dev).bool()       # [B, L] valid history frames
        Lp1 = logits.shape[1]
        L = Lp1 - 1
        null_idx = L
        # hard target kept for the diagnostics below (ret_acc / seen_match)
        ret_target = inputs["batch_retrieval_target"].to(dev).clone()
        ret_target = torch.where(is_seen, ret_target, ret_target.new_full((), null_idx))
        # windowed soft target over the real frames
        Wm = getattr(self.config.il, "retrieval_window", 2)
        tgt_idx = inputs["batch_retrieval_target"].to(dev).clamp(min=0)         # k_goal (seen)
        ar = torch.arange(L, device=dev).unsqueeze(0)                          # [1, L]
        d = (ar - tgt_idx.unsqueeze(1)).abs().float()                          # [B, L]
        win = (d <= Wm).float() * mem_mask.float() * (1.0 - d / (Wm + 1))      # in-window, valid, triangular
        win = win / win.sum(1, keepdim=True).clamp(min=1e-6)                   # per-sample distribution
        soft = torch.zeros_like(logits)
        soft[:, :L] = win
        soft[~is_seen] = 0.0
        soft[~is_seen, null_idx] = 1.0                                         # unseen -> null (hard)
        logp = F.log_softmax(logits, dim=1)
        # padded frames have logit -inf -> logp -inf; soft is 0 there, and 0*-inf = NaN.
        # only accumulate where soft>0 (valid window / null) to keep the CE finite.
        term = torch.where(soft > 0, soft * logp, torch.zeros_like(logp))
        retrieval_loss = -term.sum(1).mean()
        w_recall = getattr(self.config.il, "w_retrieval_recall", 0.5)
        if is_seen.any():
            retrieval_loss = retrieval_loss + w_recall * logits.softmax(1)[is_seen, null_idx].mean()

        # --- aux pose (x,y,θ): MSE on SEEN samples only (revisit is the active branch) ---
        gt_pose = inputs["batch_goal_rel_pose"].to(dev)          # [B,3]
        seen_f = is_seen.float()
        per = (fwd["aux_pose"] - gt_pose).square().mean(-1)      # [B]
        aux_loss = (per * seen_f).sum() / seen_f.sum().clamp(min=1.0)

        loss = action_loss + self.w_retr * retrieval_loss + self.w_aux * aux_loss

        with torch.no_grad():
            ret_acc = (logits.argmax(-1) == ret_target).float().mean()
            # --- gate seen/unseen separation + seen-only retrieval match acc (key diagnostics) ---
            gate = fwd["revisit_gate"]                            # [B] P(some real match): want HIGH seen / LOW unseen
            ns = seen_f.sum().clamp(min=1.0)
            nu = (1.0 - seen_f).sum().clamp(min=1.0)
            gate_seen = (gate * seen_f).sum() / ns                # → 1 (visited)
            gate_unseen = (gate * (1.0 - seen_f)).sum() / nu      # → 0 (unseen)
            gate_sep = gate_seen - gate_unseen                    # → large +  (the separation)
            correct = (logits.argmax(-1) == ret_target).float()  # exact index match (used for unseen->null)
            # seen "similar" match: adjacent trajectory frames are near-identical, so requiring the
            # argmax to hit the EXACT goal index is overly strict (that metric sat at 0.00). Instead,
            # count a seen retrieval as correct when the argmax lands on a REAL frame (not null) within
            # ±match_window of the true goal frame. match_window is configurable (default ±2).
            match_w = getattr(self.config.il, "match_window", 2)
            pred = logits.argmax(-1)
            seen_hit = ((pred != null_idx) & ((pred - ret_target).abs() <= match_w)).float()
            seen_match = (seen_hit * seen_f).sum() / ns           # found a frame within ±W of the goal (seen)
            unseen_null = (correct * (1.0 - seen_f)).sum() / nu   # correctly chose null (unseen)
        outputs = dict(loss=loss, action_loss=action_loss, ng_loss=ng_loss, mg_loss=mg_loss,
                       retrieval_loss=retrieval_loss, aux_loss=aux_loss, ret_acc=ret_acc,
                       gate_seen=gate_seen, gate_unseen=gate_unseen, gate_sep=gate_sep,
                       seen_match_acc=seen_match, unseen_null_acc=unseen_null)
        if (dist.get_rank() if dist.is_initialized() else 0) == 0:
            print(f"[Step {self.state.global_step}] loss={loss.item():.4f} act={action_loss.item():.4f} "
                  f"retr={retrieval_loss.item():.4f}(acc {ret_acc.item():.2f}) aux={aux_loss.item():.4f} | "
                  f"gate seen={gate_seen.item():.2f} unseen={gate_unseen.item():.2f} sep={gate_sep.item():+.2f} | "
                  f"match seen(±{match_w})={seen_match.item():.2f} unseen_null={unseen_null.item():.2f}")
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
