import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from internnav.trainer.base import BaseTrainer


class LoGoPlannerTrainer(BaseTrainer):
    """Trainer for LoGoPlanner (Peng et al., arxiv 2512.19629).

    Paper Sec IV.B / V.A specifies:
      - Loss terms: local points (eq 2), camera pose (eq 4), world points (eq 6),
        diffusion action (eq 11). Paper also mentions Goal (sub-pointgoal) and
        implicitly retains NavDP's critic head — both are included here.
      - Two-stage training: stage 1 fine-tunes the geometry decoder + task-specific
        heads (bs=12, 24h); stage 2 trains the diffusion head with geometry backbone
        frozen (bs=32, 3 days). Stage selection is done by setting freeze flags on
        the model and by zeroing individual loss weights in config.

    Loss weights (paper gives no numeric values). Exposed via config.il.loss with
    getattr defaults:
      w_diffusion (default 1.0), w_critic (1.0), w_pose (1.0),
      w_local (0.5), w_world (0.5), w_subgoal (0.1).

    Expected batch keys from the dataset / collate_fn:
      batch_pg              [B, 3]         goal in ego frame (x, y, yaw)
      batch_memory_rgb      [B, M, H, W, 3]
      batch_memory_depth    [B, H, W, 1]   (last-frame depth; matches LoGoPlanner
                                           inference which uses memory_rgbd[:,-1,:,:,3:4])
      batch_context_rgb     [B, N, H, W, 3]  N = context_size (12 in paper)
      batch_context_depth   [B, N, H, W, 1]
      batch_labels          [B, T, 3]      GT action waypoints (Δx, Δy, Δθ), T = 24
      batch_augments        [B, T, 3]      augmented (negative) actions for critic
      batch_label_critic    [B]            GT critic value for labels
      batch_augment_critic  [B]            GT critic value for augments
      batch_gt_camera_poses [B, N, P]      GT camera pose per context frame (P=5 matches
                                           ExtrinctHead.fc_pose; dataset must encode
                                           [x, y, z, sinθ, cosθ] or agreed equivalent)
      batch_gt_local_points [B, N, H, W, 3]  GT local points = D·K⁻¹·[u v 1]ᵀ
      batch_gt_world_points [B, N, H, W, 3]  GT world points = T_cw · local
      batch_gt_subgoal      [B, 3]         GT sub-pointgoal used by pg_pred_mlp
    """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.writer = None
        self.start_time = time.time()
        if hasattr(self.model, 'module'):
            self.model_device = self.model.module.device
        else:
            self.model_device = self.model.device
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[Rank {rank}] Model device: {self.model_device}")

        # Per-component loss bookkeeping for wandb/tensorboard. compute_loss runs
        # every micro-step; log() fires every logging_steps. We sum here and emit
        # the window-average in log() so each logged point is representative.
        self._loss_comp_sums: dict[str, float] = {}
        self._loss_comp_count: int = 0
        self._wandb_cfg_logged: bool = False

    def _accumulate_loss_components(self, comp: dict):
        """Add this micro-step's component values into the running window sum."""
        for k, v in comp.items():
            val = v.item() if hasattr(v, 'item') else float(v)
            self._loss_comp_sums[k] = self._loss_comp_sums.get(k, 0.0) + val
        self._loss_comp_count += 1

    def _maybe_log_lambdas_to_wandb(self):
        """One-time push of the active lambda weights into wandb.config so each
        run is self-describing. No-op if wandb isn't the active reporter."""
        if self._wandb_cfg_logged:
            return
        self._wandb_cfg_logged = True
        try:
            import wandb
            if wandb.run is not None:
                wandb.config.update(
                    {f'lambda_{k}': v for k, v in self._loss_weights().items()},
                    allow_val_change=True,
                )
        except Exception:
            pass

    def log(self, logs, *args, **kwargs):
        """Inject window-averaged per-component losses alongside the metrics
        the HF Trainer already emits (train/total loss, learning_rate, grad_norm).
        Everything in `logs` is dispatched to whatever report_to backends are
        active (wandb / tensorboard)."""
        if self._loss_comp_count > 0:
            for k, total in self._loss_comp_sums.items():
                logs[k] = total / self._loss_comp_count
            self._loss_comp_sums = {}
            self._loss_comp_count = 0
        self._maybe_log_lambdas_to_wandb()
        return super().log(logs, *args, **kwargs)

    def _loss_weights(self):
        w = self.config.il.loss
        return {
            'diffusion': getattr(w, 'w_diffusion', 1.0),
            'critic': getattr(w, 'w_critic', 1.0),
            'pose': getattr(w, 'w_pose', 1.0),
            'local': getattr(w, 'w_local', 0.5),
            'world': getattr(w, 'w_world', 0.5),
            'subgoal': getattr(w, 'w_subgoal', 0.1),
            'safety': getattr(w, 'w_safety', 0.0),  # Stage 7: collision penalty (default off)
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model_device = next(model.parameters()).device

        input_keys = [
            'batch_pg', 'batch_memory_rgb', 'batch_memory_depth',
            'batch_context_rgb', 'batch_context_depth',
            'batch_labels', 'batch_augments',
            'batch_label_critic', 'batch_augment_critic',
            'batch_gt_camera_poses', 'batch_gt_local_points',
            'batch_gt_world_points', 'batch_gt_subgoal',
        ]
        if 'batch_goal_image' in inputs:
            input_keys.append('batch_goal_image')
        inp = {k: inputs[k].to(model_device, non_blocking=True) for k in input_keys}
        torch.cuda.synchronize(model_device)

        _goal_image = inp.get('batch_goal_image', None)
        out = model(
            inp['batch_pg'],
            inp['batch_memory_rgb'],
            inp['batch_memory_depth'],
            inp['batch_context_rgb'],
            inp['batch_context_depth'],
            inp['batch_labels'],
            inp['batch_augments'],
            batch_goal_image=_goal_image,
        )
        # out: dict returned by LoGoPlannerNet.forward (train mode), see contract above.

        ng_action_loss = (out['noise_pred_ng'] - out['ng_noise']).square().mean()
        mg_action_loss = (out['noise_pred_mg'] - out['mg_noise']).square().mean()
        action_loss = 0.5 * ng_action_loss + 0.5 * mg_action_loss

        critic_loss = (
            (out['label_critic_pred'] - inp['batch_label_critic']).square().mean()
            + (out['augment_critic_pred'] - inp['batch_augment_critic']).square().mean()
        )

        # Geometry supervision. In streaming mode the backbone window has N frames
        # (anchor + trajectory + window) while the dense geometry GT is only
        # provided for the `context_size` reference frames, so shapes differ —
        # skip the term (the frozen, pretrained LingBot backbone already carries
        # geometry; per-window GT can be added later to re-enable Stage-1 here).
        def _geo_mse(pred_key, gt_key):
            pred, gt = out[pred_key], inp[gt_key]
            if pred.shape != gt.shape:
                return torch.zeros((), device=model_device)
            return (pred - gt).square().mean()
        pose_loss = _geo_mse('camera_poses_pred', 'batch_gt_camera_poses')
        local_loss = _geo_mse('local_points_pred', 'batch_gt_local_points')
        world_loss = _geo_mse('world_points_pred', 'batch_gt_world_points')

        subgoal_loss = (out['subgoal_pred'] - inp['batch_gt_subgoal']).square().mean()

        w = self._loss_weights()
        loss = (
            w['diffusion'] * action_loss
            + w['critic'] * critic_loss
            + w['pose'] * pose_loss
            + w['local'] * local_loss
            + w['world'] * world_loss
            + w['subgoal'] * subgoal_loss
        )

        # Stage 7: collision safety penalty on the policy's predicted trajectory.
        # Only added when w_safety > 0 (default off → pure imitation + inference
        # reranking). The penalty is geometric + differentiable; gradients update
        # only the policy (the obstacle cloud is detached). Monitor safety_loss vs
        # action_loss so safety never dominates — fall back to w_safety=0 if unstable.
        safety_loss = torch.zeros((), device=model_device)
        if w['safety'] > 0 and 'pred_traj_mg' in out:
            try:
                from collision_critic import differentiable_safety_loss
                lc = self.config.il.loss
                foot = getattr(lc, 'safety_footprint', 0.3)
                margin = getattr(lc, 'safety_margin', 0.3)
                max_pts = int(getattr(lc, 'safety_max_points', 1024))
                # obstacle cloud = current (last context) frame's world points, ground-plane xy
                wp = inp['batch_gt_world_points'][:, -1]               # (B, H, W, 3)
                B = wp.shape[0]
                pts = wp.reshape(B, -1, 3)                            # (B, H*W, 3)
                if pts.shape[1] > max_pts:                            # random subsample
                    idx = torch.randint(0, pts.shape[1], (max_pts,), device=pts.device)
                    pts = pts[:, idx, :]
                valid = pts.norm(dim=-1) > 1e-4                       # (B, P) drop zero/invalid
                obstacle_xy = pts[..., :2].detach()                  # treat cloud as constant
                pred_xy = out['pred_traj_mg'][..., :2]               # (B, T, 2) differentiable
                safety_loss = differentiable_safety_loss(
                    pred_xy, obstacle_xy, obstacle_mask=valid,
                    footprint_radius=foot, safety_margin=margin,
                )
                loss = loss + w['safety'] * safety_loss
            except Exception as e:  # never let the safety term crash training
                print(f"[stage7] safety loss skipped: {e}")

        # Phase α-Fix++: image-goal distillation from start_encoder (teacher).
        # Stashed by the model's training forward when IMAGEGOAL_MODE=1.
        _p = model.module.policy if hasattr(model, 'module') else model.policy
        _img_distill = getattr(_p, '_last_image_distill_loss', None)
        if _img_distill is not None:
            loss = loss + 1.0 * _img_distill

        outputs = {
            'loss': loss,
            'ng_action_loss': ng_action_loss,
            'mg_action_loss': mg_action_loss,
            'action_loss': action_loss,
            'critic_loss': critic_loss,
            'pose_loss': pose_loss,
            'local_loss': local_loss,
            'world_loss': world_loss,
            'subgoal_loss': subgoal_loss,
            'safety_loss': safety_loss,
        }

        # --- per-component metrics for wandb/tensorboard ---
        # raw      = unweighted loss term (use this to judge whether a head is
        #            actually learning, independent of its lambda).
        # weighted = lambda * raw, i.e. the term's real contribution to total.
        # Compare raw curves across runs; weighted curves explain total_loss.
        # Guarded on model.training so eval batches never pollute train/ metrics.
        if getattr(model, 'training', True):
            self._accumulate_loss_components({
                'train/total_loss':              loss,
                'train/loss_diffusion_raw':      action_loss,
                'train/loss_critic_raw':         critic_loss,
                'train/loss_pose_raw':           pose_loss,
                'train/loss_local_raw':          local_loss,
                'train/loss_world_raw':          world_loss,
                'train/loss_subgoal_raw':        subgoal_loss,
                'train/loss_safety_raw':         safety_loss,
                'train/loss_diffusion_weighted': w['diffusion'] * action_loss,
                'train/loss_critic_weighted':    w['critic'] * critic_loss,
                'train/loss_pose_weighted':      w['pose'] * pose_loss,
                'train/loss_local_weighted':     w['local'] * local_loss,
                'train/loss_world_weighted':     w['world'] * world_loss,
                'train/loss_subgoal_weighted':   w['subgoal'] * subgoal_loss,
                'train/loss_safety_weighted':    w['safety'] * safety_loss,
            })
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        try:
            lr = self.config.il.lr
            if rank == 0:
                print(f"[Rank 0] Using learning rate: {lr}")
        except AttributeError:
            lr = 1e-4
            if rank == 0:
                print(f"[Rank 0] Warning: Using default learning rate: {lr}")

        model_for_optim = self.model.module if hasattr(self.model, 'module') else self.model

        trainable = [p for p in model_for_optim.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        if rank == 0:
            total_params = sum(p.numel() for p in trainable)
            all_params = sum(p.numel() for p in model_for_optim.parameters())
            print(f"[Rank 0] Optimizer created with {len(optimizer.param_groups)} param groups")
            print(f"[Rank 0] Trainable parameters: {total_params:,} / {all_params:,}")

        return optimizer

    def create_scheduler(self, optimizer, num_training_steps: int):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.5, total_iters=10000
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        print("\n=== create optimizer and scheduler ===")
        self.optimizer = self.create_optimizer()
        self.lr_scheduler = self.create_scheduler(self.optimizer, num_training_steps)
        return self.optimizer, self.lr_scheduler

    def get_train_dataloader(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # Stage 2: sequential mode streams episodes in temporal order via a
        # lane-major batch sampler so the model can carry a per-episode KV cache.
        if getattr(self.train_dataset, 'sequential', False):
            from internnav.dataset.logoplanner_sequential import StreamingEpisodeBatchSampler
            if world_size > 1:
                raise NotImplementedError(
                    'sequential streaming sampler is single-process for now; '
                    'launch with one rank (the fixed-window train path supports DDP).'
                )
            batch_sampler = StreamingEpisodeBatchSampler(
                self.train_dataset.episodes, self.config.il.batch_size, drop_ragged_tail=True
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.config.il.num_workers,
                pin_memory=True,
                collate_fn=self.data_collator,
            )
        sampler = DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=1234
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.il.batch_size,
            sampler=sampler,
            num_workers=self.config.il.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=self.data_collator,
        )

    def save_model(self, output_dir, state_dict=None, **kwargs):
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model_to_save.state_dict(), output_dir + "logoplanner.ckpt")
        print(f"Saving model to {output_dir} (is DDP: {hasattr(self.model, 'module')})")
