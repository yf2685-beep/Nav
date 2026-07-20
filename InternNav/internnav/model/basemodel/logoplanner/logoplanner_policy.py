"""HuggingFace wrapper for LoGoPlanner training in InternNav.

The underlying model (``LoGoPlanner_Policy``) lives in
``NavDP/baselines/logoplanner/policy_network.py`` and is NOT duplicated here.
This module imports it via ``sys.path`` and adds:

  - ``LoGoPlannerModelConfig``: mirrors ``NavDPModelConfig`` so the trainer's
    ``from_pretrained(... config=config)`` path works.
  - ``LoGoPlannerNet(PreTrainedModel)``: thin wrapper that owns a single
    ``LoGoPlanner_Policy`` instance as ``self.policy``.
  - A training ``forward()`` that reuses ``self.policy``'s submodules
    (rgbd_encoder, state_encoder, start_encoder, state_decoder, pg_pred_mlp,
    input_embed, decoder, action_head, critic_head, time_emb, noise_scheduler,
    cond_pos_embed, out_pos_embed, layernorm, tgt_mask, cond_critic_mask)
    and returns a dict keyed exactly as ``LoGoPlannerTrainer`` expects.

Note on Pi3 weights: ``GeometryModel`` extends ``Pi3`` which instantiates
``dinov2_vitl14_reg(pretrained=False)`` — the network is structurally fine
with random weights. For a smoke test we do not need the released checkpoint.
"""

import os
import sys

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from internnav.configs.model.base_encoders import ModelCfg
from internnav.configs.trainer.exp import ExpCfg


# --- Make NavDP/baselines/logoplanner importable -------------------------
# Repo layout: <ROOT>/InternNav/ and <ROOT>/NavDP/ are siblings.
_THIS = os.path.dirname(os.path.abspath(__file__))
_INTERNNAV_ROOT = os.path.abspath(os.path.join(_THIS, '../../../..'))
_ROOT = os.path.dirname(_INTERNNAV_ROOT)
_LOGO_DIR = os.path.join(_ROOT, 'NavDP', 'baselines', 'logoplanner')
if _LOGO_DIR not in sys.path:
    sys.path.insert(0, _LOGO_DIR)

from policy_network import LoGoPlanner_Policy  # noqa: E402


class LoGoPlannerModelConfig(PretrainedConfig):
    model_type = 'logoplanner'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)

    @classmethod
    def from_dict(cls, config_dict):
        if 'model_cfg' in config_dict:
            config_dict['model_cfg'] = ExpCfg(**config_dict['model_cfg'])
        return super().from_dict(config_dict)


class LoGoPlannerNet(PreTrainedModel):
    config_class = LoGoPlannerModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config, 'model_dump'):
            config = cls.config_class(model_cfg=config)

        model = cls(config)
        model.to(model._device)

        if pretrained_model_name_or_path is None or len(pretrained_model_name_or_path) == 0:
            pass
        elif os.path.isdir(pretrained_model_name_or_path):
            incompatible_keys, _ = model.load_state_dict(
                torch.load(os.path.join(pretrained_model_name_or_path, 'pytorch_model.bin'))
            )
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')
        else:
            ckpt = torch.load(pretrained_model_name_or_path, map_location='cpu')
            state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
            # strict=False ignores missing/unexpected keys but STILL raises on
            # shape mismatches — so drop those explicitly. Streaming resizes
            # cond_pos_embed (36->12) and adds gct_assembler; those reinit fresh
            # while the backbone / DA-S / diffusion layers still warm-start.
            msd = model.state_dict()
            dropped = [k for k, v in state.items()
                       if k in msd and hasattr(v, 'shape') and tuple(v.shape) != tuple(msd[k].shape)]
            for k in dropped:
                state.pop(k)
            if dropped:
                print(f'Dropping {len(dropped)} shape-mismatched ckpt keys: {dropped[:5]}')
            incompatible_keys, _ = model.load_state_dict(state, strict=False)
            if len(incompatible_keys) > 0:
                print(f'Incompatible keys: {incompatible_keys}')

        return model

    def __init__(self, config: LoGoPlannerModelConfig):
        super().__init__(config)
        if isinstance(config, LoGoPlannerModelConfig):
            self.model_config = ModelCfg(**config.model_cfg['model'])
        else:
            self.model_config = config

        il = self.config.model_cfg['il']
        self._device = torch.device(f"cuda:{config.model_cfg['local_rank']}")
        self.image_size = il['image_size']
        self.memory_size = il['memory_size']
        self.predict_size = il['predict_size']
        self.temporal_depth = il['temporal_depth']
        self.attention_heads = il['heads']
        self.input_channels = il['channels']
        self.token_dim = il['token_dim']
        self.context_size = il.get('context_size', 12)
        # Stage 1: RGB-only trajectory backbone. Default True keeps legacy
        # (depth-on) behaviour for configs that predate this flag.
        self.use_depth = il.get('use_depth', True)

        self.policy = LoGoPlanner_Policy(
            image_size=self.image_size,
            memory_size=self.memory_size,
            context_size=self.context_size,
            predict_size=self.predict_size,
            temporal_depth=self.temporal_depth,
            heads=self.attention_heads,
            token_dim=self.token_dim,
            channels=self.input_channels,
            use_depth=self.use_depth,
            device=self._device,
        )

        # Apply paper-style two-stage freezing if `loss.stage` is set.
        #   stage 1: freeze the geometry ViT-L encoder (Pi3 dinov2_vitl14_reg);
        #            geometry decoder + all task-specific heads stay trainable.
        #   stage 2: also freeze the geometry decoder + register_token; only
        #            task-specific heads + the diffusion-policy decoder train.
        loss_cfg = il.get('loss') or {}
        stage = loss_cfg.get('stage', 0) if isinstance(loss_cfg, dict) else getattr(loss_cfg, 'stage', 0)
        self._apply_stage_freeze(int(stage or 0))

    def _apply_stage_freeze(self, stage: int):
        """Freeze parameters according to LoGoPlanner paper's training stage.

        See paper §V.A. stage=0 is the single-stage baseline (no freezing).
        """
        if stage == 0:
            print('[stage-freeze] stage=0 (single-stage), no freezing applied')
            return
        if stage not in (1, 2):
            raise ValueError(f'unsupported stage {stage}')

        ge = self.policy.state_encoder  # GeometryModel

        frozen_count = 0
        if stage >= 1:
            for p in ge.encoder.parameters():
                p.requires_grad = False
                frozen_count += p.numel()

        if stage >= 2:
            for p in ge.decoder.parameters():
                p.requires_grad = False
                frozen_count += p.numel()
            # Pi3 register_token sits beside the decoder; freeze together.
            ge.register_token.requires_grad = False
            frozen_count += ge.register_token.numel()

        all_count = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f'[stage-freeze] stage={stage}: '
            f'frozen {frozen_count/1e6:.1f}M params, '
            f'trainable {trainable/1e6:.1f}M / {all_count/1e6:.1f}M total '
            f'({100*trainable/all_count:.1f}%)'
        )

    # Keep ``policy.device`` / tgt_mask / cond_critic_mask consistent with HF's
    # .to() — ``LoGoPlanner_Policy`` stores .device as a plain attribute.
    def to(self, device, *args, **kwargs):
        self = super().to(device, *args, **kwargs)
        self._device = device
        self.policy.device = device
        self.policy.tgt_mask = self.policy.tgt_mask.to(device)
        self.policy.cond_critic_mask = self.policy.cond_critic_mask.to(device)
        return self

    # --------------------------------------------------------------------
    # Training forward
    #
    # Matches the call made by ``LoGoPlannerTrainer.compute_loss``:
    #     out = model(batch_pg, batch_memory_rgb, batch_memory_depth,
    #                 batch_context_rgb, batch_context_depth,
    #                 batch_labels, batch_augments)
    # and returns a dict keyed:
    #     noise_pred_ng, noise_pred_mg, ng_noise, mg_noise,
    #     label_critic_pred, augment_critic_pred,
    #     camera_poses_pred, local_points_pred, world_points_pred,
    #     subgoal_pred
    # --------------------------------------------------------------------
    def _sample_noise(self, action):
        device = action.device
        p = self.policy
        noise = torch.randn(action.shape, device=device)
        timesteps = torch.randint(
            0, p.noise_scheduler.config.num_train_timesteps, (action.shape[0],), device=device
        ).long()
        time_embeds = p.time_emb(timesteps).unsqueeze(1)
        noisy_action = p.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action_embed = p.input_embed(noisy_action)
        # also return raw noisy_action + timesteps so forward() can reconstruct the
        # predicted clean trajectory (x0) for the Stage-7 safety loss.
        return noise, time_embeds, noisy_action_embed, noisy_action, timesteps

    def forward(
        self,
        batch_pg,
        batch_memory_rgb,
        batch_memory_depth,
        batch_context_rgb,
        batch_context_depth,
        batch_labels,
        batch_augments,
        batch_goal_image=None,
    ):
        # batch_goal_image is supplied by the trainer for image-goal (Phase α) mode.
        # The point-goal / multi-stop path conditions on batch_pg (the subgoal point),
        # so we accept and ignore it here unless image-goal mode is wired in.
        _ = batch_goal_image
        p = self.policy
        device = next(self.parameters()).device

        if getattr(p, '_streaming', False):
            return self._forward_stream(
                batch_pg, batch_context_rgb, batch_context_depth,
                batch_labels, batch_augments, device,
            )

        pg = batch_pg.to(device, dtype=torch.float32)
        mem_rgb = batch_memory_rgb.to(device, dtype=torch.float32)
        mem_depth = batch_memory_depth.to(device, dtype=torch.float32)
        ctx_rgb = batch_context_rgb.to(device, dtype=torch.float32)
        ctx_depth = batch_context_depth.to(device, dtype=torch.float32)
        labels = batch_labels.to(device, dtype=torch.float32)
        augments = batch_augments.to(device, dtype=torch.float32)

        B = pg.shape[0]
        assert mem_rgb.shape[1] == self.memory_size, (
            f"memory_size mismatch: got {mem_rgb.shape[1]}, expected {self.memory_size}"
        )
        assert ctx_rgb.shape[1] == self.context_size, (
            f"context_size mismatch: got {ctx_rgb.shape[1]}, expected {self.context_size}"
        )

        # --- encode memory + context (real forward paths of LoGoPlanner_Policy)
        # Stage 1: depth is dropped from the trajectory backbone when use_depth
        # is False; mem_depth stays in the batch (collision critic uses it).
        rgbd_embed = p.rgbd_encoder(mem_rgb, mem_depth if self.use_depth else None)  # (B, M, D)
        (_, state_token, scene_token), (camera_poses_pred, local_points_pred, world_points_pred) = (
            p.state_encoder(ctx_rgb, ctx_depth)
        )
        unify_token = torch.cat([state_token, scene_token], dim=1)  # (B, 2N, D)

        startgoal_embed = p.start_encoder(pg).unsqueeze(1)  # (B, 1, D)

        # === v3: FeatureCompressor + cosine hard retrieval + concat-or-null ===
        # Goal image is a single frame (not tiled). Compressor operates on
        # mean-pooled backbone features (2048-d) already stored by state_encoder.
        past_compact = None
        goal_compact = None
        retrieved_token = None
        if batch_goal_image is not None:
            gi = batch_goal_image.to(device, dtype=torch.float32)        # (B, H, W, 3)
            goal_feat = p.state_encoder.encode_single_frame(gi)          # (B, 2048)
            past_feats = p.state_encoder._last_frame_feats               # (B, T, 2048)
            past_compact = p.feature_compressor(past_feats)              # (B, T, 128)
            goal_compact = p.feature_compressor(goal_feat)               # (B, 128)
            sim = torch.bmm(past_compact,
                            goal_compact.unsqueeze(-1)).squeeze(-1)      # (B, T)
            max_sim, argmax_t = sim.max(dim=-1)                          # (B,)
            gather_idx = argmax_t.view(B, 1, 1).expand(-1, 1, state_token.size(-1))
            retrieved_real = torch.gather(state_token, 1, gather_idx)   # (B, 1, D)
            thr = torch.sigmoid(p.retrieval_threshold_logit)
            gate = (max_sim > thr).float().view(B, 1, 1)
            null = p.null_retrieval.expand(B, -1, -1)
            retrieved_token = gate * retrieved_real + (1.0 - gate) * null

        state_embed = p.state_decoder(torch.cat([state_token, startgoal_embed], dim=1))  # (B, 1, D)
        if retrieved_token is not None:
            state_embed = p.retrieval_concat_proj(
                torch.cat([state_embed, retrieved_token], dim=-1)        # (B, 1, 2D) → (B, 1, D)
            )
        subgoal_pred = p.pg_pred_mlp(state_embed).squeeze(1)  # (B, 3)

        # --- diffusion: sample noise for ng and mg branches
        ng_noise, ng_time_embed, ng_noisy_action_embed, _, _ = self._sample_noise(labels)
        mg_noise, mg_time_embed, mg_noisy_action_embed, mg_noisy_action, mg_timesteps = self._sample_noise(labels)

        nogoal_embed = torch.zeros_like(startgoal_embed)  # (B, 1, D)

        # --- Build conditioning sequences -----
        def build_cond(time_embed, goal_slots):
            # goal_slots: list of three (B, 1, D) tensors
            cond = torch.cat([time_embed, *goal_slots, rgbd_embed, unify_token], dim=1)
            return cond + p.cond_pos_embed(cond)

        # no-goal branch: goal slots are zero
        ng_cond = build_cond(ng_time_embed, [nogoal_embed, nogoal_embed, nogoal_embed])
        # multi-goal branch: use the sub-pointgoal state_embed in all three slots
        mg_cond = build_cond(mg_time_embed, [state_embed, state_embed, state_embed])

        out_pos_embed_nx = p.out_pos_embed(ng_noisy_action_embed)
        ng_act_in = ng_noisy_action_embed + out_pos_embed_nx
        mg_act_in = mg_noisy_action_embed + out_pos_embed_nx

        ng_out = p.decoder(tgt=ng_act_in, memory=ng_cond, tgt_mask=p.tgt_mask)
        ng_out = p.layernorm(ng_out)
        noise_pred_ng = p.action_head(ng_out)

        mg_out = p.decoder(tgt=mg_act_in, memory=mg_cond, tgt_mask=p.tgt_mask)
        mg_out = p.layernorm(mg_out)
        noise_pred_mg = p.action_head(mg_out)

        # --- Stage 7: reconstruct the predicted CLEAN trajectory (x0) from the mg
        # epsilon prediction, one-step (differentiable wrt the policy). The action
        # head predicts ε; x0 = (x_t - sqrt(1-ᾱ_t)·ε) / sqrt(ᾱ_t). Waypoint xy is
        # the cumulative sum of the per-step deltas (÷4, matching inference).
        abar = p.noise_scheduler.alphas_cumprod.to(device)[mg_timesteps].view(-1, 1, 1)
        x0_pred_mg = (mg_noisy_action - (1.0 - abar).sqrt() * noise_pred_mg) / abar.sqrt().clamp_min(1e-6)
        pred_traj_mg = torch.cumsum(x0_pred_mg / 4.0, dim=1)  # (B, T, 3) robot-frame waypoints

        # --- critic on GT labels and augments (no-goal cond, masked per cond_critic_mask)
        label_embed = p.input_embed(labels).detach()
        augment_embed = p.input_embed(augments).detach()
        label_act_in = label_embed + out_pos_embed_nx
        augment_act_in = augment_embed + out_pos_embed_nx

        cr_label_out = p.decoder(tgt=label_act_in, memory=ng_cond, memory_mask=p.cond_critic_mask)
        cr_label_out = p.layernorm(cr_label_out)
        label_critic_pred = p.critic_head(cr_label_out.mean(dim=1))[:, 0]

        cr_aug_out = p.decoder(tgt=augment_act_in, memory=ng_cond, memory_mask=p.cond_critic_mask)
        cr_aug_out = p.layernorm(cr_aug_out)
        augment_critic_pred = p.critic_head(cr_aug_out.mean(dim=1))[:, 0]

        return {
            'noise_pred_ng': noise_pred_ng,
            'noise_pred_mg': noise_pred_mg,
            'ng_noise': ng_noise,
            'mg_noise': mg_noise,
            'label_critic_pred': label_critic_pred,
            'augment_critic_pred': augment_critic_pred,
            'camera_poses_pred': camera_poses_pred,
            'local_points_pred': local_points_pred,
            'world_points_pred': world_points_pred,
            'subgoal_pred': subgoal_pred,
            'pred_traj_mg': pred_traj_mg,  # Stage 7: predicted clean waypoints (B, T, 3)
        }

    # --------------------------------------------------------------------
    # Streaming GCT training forward (LOGO_STREAMING=1)
    #
    # batch_context_rgb/depth carry the bounded per-episode WINDOW
    #   [anchor (n_anchor) | trajectory keyframes | recent window (n_window)]
    # with the decision frame last. One parallel causal backbone forward
    # (anchors bidirectional) -> per-frame state/scene tokens -> GCT summary ->
    # diffusion head conditioned on [time, goal*3, summary]. No NavDP memory.
    # --------------------------------------------------------------------
    def _forward_stream(self, batch_pg, batch_context_rgb, batch_context_depth,
                        batch_labels, batch_augments, device):
        from streaming_gct import partition_window_tokens
        p = self.policy
        pg = batch_pg.to(device, dtype=torch.float32)
        ctx_rgb = batch_context_rgb.to(device, dtype=torch.float32)   # (B, N, H, W, 3)
        ctx_depth = batch_context_depth.to(device, dtype=torch.float32)
        labels = batch_labels.to(device, dtype=torch.float32)
        augments = batch_augments.to(device, dtype=torch.float32)

        # Encode the bounded window into per-frame tokens.
        #   Stage-2 (frozen backbone, geometry not supervised): stream frame-by-
        #   frame through the KV cache (encode_window_streaming) so attention
        #   memory stays bounded by the sliding window — the parallel forward's
        #   O(N^2) global attention over N=anchor+traj+window frames is the
        #   memory hog. Detach (no grad into the frozen backbone).
        #   Stage-1 (LOGO_STREAM_BACKBONE_GRAD=1): parallel forward WITH grad so
        #   geometry heads / DA-S fusion train (needs a large GPU).
        import contextlib
        nograd = getattr(p, '_stream_backbone_nograd', True)
        use_parallel = getattr(p, '_stream_train_parallel', False) or not hasattr(p.state_encoder, 'encode_window_streaming')
        _ctx = torch.no_grad() if nograd else contextlib.nullcontext()
        with _ctx:
            if use_parallel:
                # One batched causal forward over the window (fast; bigger memory).
                (_, state_token, scene_token), (camera_poses_pred, local_points_pred, world_points_pred) = (
                    p.state_encoder(ctx_rgb, ctx_depth, reset_cache=True, assert_context=False)
                )
            else:
                # Frame-by-frame streaming encode (bounded memory; slow — N forwards).
                (_, state_token, scene_token), (camera_poses_pred, local_points_pred, world_points_pred) = (
                    p.state_encoder.encode_window_streaming(ctx_rgb, ctx_depth)
                )
        if nograd:
            state_token = state_token.detach()
            scene_token = scene_token.detach()

        parts = partition_window_tokens(state_token, scene_token, p.n_anchor, p.n_window)
        summary = p.gct_assembler(parts)            # (B, 8, D)
        cur_state = state_token[:, -1:]             # (B, 1, D) current/decision frame

        # sub-pointgoal head (trained against batch_gt_subgoal)
        startgoal_embed = p.start_encoder(pg).unsqueeze(1)                      # (B,1,D)
        state_embed = p.state_decoder(torch.cat([cur_state, startgoal_embed], dim=1))  # (B,1,D)
        subgoal_pred = p.pg_pred_mlp(state_embed).squeeze(1)                    # (B,3)

        # diffusion: ng (no-goal) + mg (goal-conditioned) branches
        ng_noise, ng_time_embed, ng_noisy_action_embed, _, _ = self._sample_noise(labels)
        mg_noise, mg_time_embed, mg_noisy_action_embed, mg_noisy_action, mg_timesteps = self._sample_noise(labels)
        nogoal_embed = torch.zeros_like(startgoal_embed)

        def build_cond(time_embed, goal_slots):
            cond = torch.cat([time_embed, *goal_slots, summary], dim=1)         # (B, 12, D)
            return cond + p.cond_pos_embed(cond)

        ng_cond = build_cond(ng_time_embed, [nogoal_embed, nogoal_embed, nogoal_embed])
        mg_cond = build_cond(mg_time_embed, [state_embed, state_embed, state_embed])

        out_pe = p.out_pos_embed(ng_noisy_action_embed)
        ng_out = p.action_head(p.layernorm(
            p.decoder(tgt=ng_noisy_action_embed + out_pe, memory=ng_cond, tgt_mask=p.tgt_mask)))
        mg_out = p.action_head(p.layernorm(
            p.decoder(tgt=mg_noisy_action_embed + out_pe, memory=mg_cond, tgt_mask=p.tgt_mask)))

        # Stage 7: reconstruct predicted clean trajectory (x0) from the mg epsilon.
        abar = p.noise_scheduler.alphas_cumprod.to(device)[mg_timesteps].view(-1, 1, 1)
        x0_pred_mg = (mg_noisy_action - (1.0 - abar).sqrt() * mg_out) / abar.sqrt().clamp_min(1e-6)
        pred_traj_mg = torch.cumsum(x0_pred_mg / 4.0, dim=1)

        # critic on GT labels + augments (no-goal cond, masked per cond_critic_mask)
        label_act_in = p.input_embed(labels).detach() + out_pe
        augment_act_in = p.input_embed(augments).detach() + out_pe
        label_critic_pred = p.critic_head(p.layernorm(
            p.decoder(tgt=label_act_in, memory=ng_cond, memory_mask=p.cond_critic_mask)).mean(dim=1))[:, 0]
        augment_critic_pred = p.critic_head(p.layernorm(
            p.decoder(tgt=augment_act_in, memory=ng_cond, memory_mask=p.cond_critic_mask)).mean(dim=1))[:, 0]

        return {
            'noise_pred_ng': ng_out,
            'noise_pred_mg': mg_out,
            'ng_noise': ng_noise,
            'mg_noise': mg_noise,
            'label_critic_pred': label_critic_pred,
            'augment_critic_pred': augment_critic_pred,
            'camera_poses_pred': camera_poses_pred,
            'local_points_pred': local_points_pred,
            'world_points_pred': world_points_pred,
            'subgoal_pred': subgoal_pred,
            'pred_traj_mg': pred_traj_mg,
        }
