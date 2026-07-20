"""Plan C: Hybrid Pi3 + LingBot policy network.

Goal: keep Pi3 for navigation-friendly current-frame perception while adding
LingBot's streaming long-memory branch + camera-pose-as-action-prior.

Architecture
============

    short memory (8 frames) ─→ rgbd_encoder (NavDP_RGBD_Backbone)
    current ctx (12 frames) ─→ state_encoder (Pi3, FROZEN after Stage 1)
                                      │
                                      ▼
                              state_token, scene_token

    long memory (48 frames) ─→ long_memory_encoder (LingBot Aggregator)
                                      │
                                      ▼
                              long_state_token, long_scene_token,
                              long_camera_poses  ◄── action prior!

    diffusion decoder cross-attends BOTH Pi3 tokens AND LingBot long tokens.
    Initial noise = scaled LingBot pose-delta prior + gaussian (warm-start).

Stage separation
================
- Stage 1: train Pi3 + heads with geometry losses (same as original LoGoPlanner).
           LingBot path is dormant / frozen at its pretrained weights.
- Stage 2: freeze Pi3.  Train LingBot path's adapters + diffusion + critic +
           subgoal. LingBot's GCA backbone can stay frozen (it already encodes
           good memory features) OR fine-tune with low LR.

Why this satisfies the project goal
===================================
The two stages train DIFFERENT subsets of parameters:
- Stage 1: state_encoder (Pi3) + Pi3 heads
- Stage 2: long_memory_encoder adapters + diffusion + critic + subgoal
There is NO overlap and NO need to freeze-then-unfreeze the same module.

Implementation status:  SKELETON ONLY — marked TODOs throughout.
Not wired into trainer / server yet.  F1 single-node training is unaffected.
"""

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy_backbone import *
from geometry_model import GeometryModel              # Pi3
from geometry_model_lingbot import GeometryModel_LingBot


class LoGoPlanner_Policy_Hybrid(nn.Module):
    """Hybrid Pi3 + LingBot diffusion policy.

    Args:
        image_size, memory_size, context_size, predict_size: same as original.
        long_memory_size: number of frames in LingBot long-memory branch (e.g. 48).
        use_pose_prior: if True, diffusion init noise is biased by LingBot's
            predicted camera-pose delta (acts as behavior-cloning warm-start).
        pose_align_loss: if True, training adds an aux loss aligning the first
            few action steps with LingBot's short-horizon pose-delta prediction.
    """

    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 context_size=12,
                 long_memory_size=48,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 use_pose_prior=True,
                 pose_align_loss=True,
                 lingbot_freeze=True,         # Stage-2-default: freeze LingBot GCA
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.long_memory_size = long_memory_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        self.use_pose_prior = use_pose_prior
        self.pose_align_loss = pose_align_loss

        # ─── input encoders ─────────────────────────────────────────────────

        # 1. Short-horizon RGB-D memory (8 frames)
        self.rgbd_encoder = NavDP_RGBD_Backbone(
            image_size, token_dim, memory_size=memory_size, device=device,
        )

        # 2. Current context: Pi3 (navigation-friendly)
        # Stage 1 trains this with geometry losses.
        self.state_encoder = GeometryModel(context_size=context_size, device=device)

        # 3. NEW: Long-horizon memory: LingBot streaming aggregator.
        # context_size here is LONG (48) — this is where LingBot's
        # paged KV cache pays off.
        self.long_memory_encoder = GeometryModel_LingBot(
            context_size=long_memory_size, device=device,
        )
        if lingbot_freeze:
            for p in self.long_memory_encoder.parameters():
                p.requires_grad = False

        # ─── adapters into policy embedding space ───────────────────────────

        self.point_encoder = nn.Linear(3, self.token_dim)
        self.start_encoder = nn.Linear(3, self.token_dim)

        # Pi3 outputs (B, T, 384) — already in token_dim.
        # LingBot also outputs (B, T, 384) — same.
        # We add a small adapter to align distributions before cross-attention.
        # TODO: confirm long_memory_encoder.state_layer output dim is 384.
        self.long_state_adapter = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
        )
        self.long_scene_adapter = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, self.token_dim),
        )

        # ─── pose prior projection ──────────────────────────────────────────
        # LingBot camera_pose: (B, T, 5) = (x, y, z, sin theta, cos theta).
        # Take the *delta* between consecutive frames as a per-step action prior.
        # Project (5,) → token_dim to inject as initial-noise bias.
        # NOTE: at inference time we only get poses for the *observed* frames;
        # we extrapolate the most-recent delta as a constant velocity prior.
        if self.use_pose_prior:
            self.pose_prior_proj = nn.Linear(5, 3)  # → action space (x, y, theta)

        # ─── decoder + heads ────────────────────────────────────────────────
        self.state_decoder = TokenCompressor(
            embed_dim=token_dim, num_heads=heads, target_length=1,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=self.temporal_depth,
        )

        self.input_embed = nn.Linear(3, token_dim)
        self.pg_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2), nn.ReLU(),
            nn.Linear(token_dim // 2, token_dim // 4), nn.ReLU(),
            nn.Linear(token_dim // 4, 3),
        )

        self.cond_pos_embed = LearnablePositionalEncoding(
            token_dim,
            # cond seq = time(1) + goal(3) + rgbd(M) + ctx_tokens(2T) + long_tokens(2T_long)
            1 + 3 + memory_size + context_size * 2 + long_memory_size * 2,
        )
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

        # ─── DDP compat (freeze unused; same trick as patched policy_network.py)
        _unused = ('cs_pred_mlp', 'point_encoder.', 'mask_token')
        for n, p in self.named_parameters():
            if any(s in n for s in _unused):
                p.requires_grad = False

        # ─── attention masks ────────────────────────────────────────────────
        self.tgt_mask = torch.triu(
            torch.ones(predict_size, predict_size)
        ).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(
            self.tgt_mask == 0, float('-inf'),
        ).masked_fill(self.tgt_mask == 1, 0.0)

        # ─── stage freeze helpers (so logoplanner_policy._apply_stage_freeze works)
        # Stage 1 should freeze NOTHING new (we want Pi3 to train).
        # Stage 2 should freeze Pi3 (state_encoder).
        # We expose Pi3 as .encoder/.decoder aliases just like Pi3 itself.

    # ────────────────────────────────────────────────────────────────────────
    # FORWARD (inference)
    # ────────────────────────────────────────────────────────────────────────

    def predict_noise(self, last_actions, timestep,
                      goal_embed, rgbd_embed, ctx_unify, long_unify):
        """Diffusion noise prediction conditioned on ALL feature sources."""
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile(
            (last_actions.shape[0], 1, 1),
        )
        cond_seq = torch.cat([
            time_embeds,
            goal_embed, goal_embed, goal_embed,
            rgbd_embed,
            ctx_unify,        # Pi3 current scene
            long_unify,       # LingBot long memory
        ], dim=1)
        cond_embedding = cond_seq + self.cond_pos_embed(cond_seq)
        input_embedding = (
            action_embeds + self.out_pos_embed(action_embeds) + goal_embed
        )
        output = self.decoder(
            tgt=input_embedding,
            memory=cond_embedding,
            tgt_mask=self.tgt_mask.to(self.device),
        )
        output = self.layernorm(output)
        return self.action_head(output)

    def _compute_pose_prior(self, lb_camera_poses):
        """Convert LingBot camera-pose sequence into a per-step action prior.

        lb_camera_poses: (B, T_long, 5) = (x, y, z, sin θ, cos θ)
        Returns: (B, predict_size, 3) — predicted (Δx, Δy, Δθ) per step,
            extrapolated forward from the last observed delta (constant-velocity).
        """
        if not self.use_pose_prior:
            return None
        # delta between consecutive observed poses
        dxy = lb_camera_poses[:, 1:, :2] - lb_camera_poses[:, :-1, :2]  # (B,T-1,2)
        sin_t = lb_camera_poses[..., 3]
        cos_t = lb_camera_poses[..., 4]
        theta = torch.atan2(sin_t, cos_t)
        dtheta = theta[:, 1:] - theta[:, :-1]                          # (B,T-1)
        # Constant-velocity extrapolation: take mean of last few deltas
        k = min(4, dxy.shape[1])
        last_dxy = dxy[:, -k:].mean(dim=1, keepdim=True)               # (B,1,2)
        last_dt  = dtheta[:, -k:].mean(dim=1, keepdim=True).unsqueeze(-1)  # (B,1,1)
        prior = torch.cat([last_dxy, last_dt], dim=-1)                 # (B,1,3)
        prior = prior.expand(-1, self.predict_size, -1)                # (B,P,3)
        return prior

    @torch.no_grad()
    def predict_pointgoal_action(self,
                                 start_goal,
                                 memory_rgbd,
                                 context_rgbd,
                                 long_memory_rgbd,
                                 sample_num=16):
        """Hybrid inference.

        Args:
            start_goal: (B, 3) goal in robot frame.
            memory_rgbd: (B, 8, H, W, 4) short-horizon RGBD memory.
            context_rgbd: (B, 12, H, W, 4) current context for Pi3.
            long_memory_rgbd: (B, 48, H, W, 4) long memory for LingBot.
        """
        B = start_goal.shape[0]
        tensor_start_goal = torch.as_tensor(
            start_goal[0:1], dtype=torch.float32, device=self.device,
        )
        startgoal_embed = self.start_encoder(tensor_start_goal).unsqueeze(1)

        # Pi3 short branches (same as original)
        rgbd_embed = self.rgbd_encoder(
            memory_rgbd[0:1][..., :3], memory_rgbd[0:1, -1][..., 3:4],
        )
        [_, state_tok_pi3, scene_tok_pi3], _ = self.state_encoder(
            context_rgbd[0:1][..., :3], context_rgbd[0:1][..., 3:4],
        )
        ctx_unify = torch.cat([state_tok_pi3, scene_tok_pi3], dim=1)

        # LingBot long branch
        [_, state_tok_lb, scene_tok_lb], [lb_camera_poses, _, _] = \
            self.long_memory_encoder(
                long_memory_rgbd[0:1][..., :3],
                long_memory_rgbd[0:1][..., 3:4],
            )
        state_tok_lb = self.long_state_adapter(state_tok_lb)
        scene_tok_lb = self.long_scene_adapter(scene_tok_lb)
        long_unify = torch.cat([state_tok_lb, scene_tok_lb], dim=1)

        # subgoal head uses Pi3 state (unchanged from baseline)
        state_embed = self.state_decoder(
            torch.cat([state_tok_pi3, startgoal_embed], dim=1),
        )
        sub_pointgoal_pd = self.pg_pred_mlp(state_embed).squeeze(1)

        # tile for sample_num diffusion candidates
        rgbd_embed   = torch.repeat_interleave(rgbd_embed,   sample_num, dim=0)
        state_embed  = torch.repeat_interleave(state_embed,  sample_num, dim=0)
        ctx_unify    = torch.repeat_interleave(ctx_unify,    sample_num, dim=0)
        long_unify   = torch.repeat_interleave(long_unify,   sample_num, dim=0)

        # ─── initial noise (pose-prior biased) ──────────────────────────────
        pose_prior = self._compute_pose_prior(lb_camera_poses)        # (B,P,3)
        if pose_prior is not None:
            pose_prior_tiled = torch.repeat_interleave(
                pose_prior, sample_num, dim=0,
            )
            gauss = torch.randn(
                (sample_num * B, self.predict_size, 3),
                device=self.device,
            )
            noisy_action = 0.5 * pose_prior_tiled + 0.5 * gauss
        else:
            noisy_action = torch.randn(
                (sample_num * B, self.predict_size, 3), device=self.device,
            )

        # diffusion loop
        self.noise_scheduler.set_timesteps(
            self.noise_scheduler.config.num_train_timesteps,
        )
        naction = noisy_action
        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.predict_noise(
                naction, k.unsqueeze(0),
                state_embed, rgbd_embed, ctx_unify, long_unify,
            )
            naction = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=naction,
            ).prev_sample

        # critic (unchanged)
        critic_values = self.predict_critic(
            naction, rgbd_embed, ctx_unify, long_unify,
        )
        critic_values = critic_values.reshape(B, sample_num)

        all_trajectory = torch.cumsum(naction / 4.0, dim=1)
        all_trajectory = all_trajectory.reshape(
            B, sample_num, self.predict_size, 3,
        )
        # ... ranking + top-k same as original ...

        return all_trajectory, critic_values, lb_camera_poses

    def predict_critic(self, predict_trajectory, rgbd_embed,
                       ctx_unify, long_unify):
        # TODO: mirror predict_noise's cond_seq layout. Original critic masked
        # the goal positions. Keep same masking idea.
        raise NotImplementedError("port from policy_network.py with extra long_unify input")


# ────────────────────────────────────────────────────────────────────────────
# TODO list for full implementation:
# ────────────────────────────────────────────────────────────────────────────
# 1. Verify GeometryModel_LingBot output shapes:
#    - state_layer outputs (B, T, 384)?  scene_layer outputs (B, T, 384)?
# 2. Dataset (logoplanner_dataset_lerobot.py):
#    - Currently provides memory_rgbd (8) + context_rgbd (12).
#    - Need to add long_memory_rgbd (48 frames).
#    - Confirm InternData-N1 episodes have >=48 prior frames available.
#    - Fallback if not enough: pad with first frame or replicate context_rgbd.
# 3. Trainer (logoplanner_trainer.py):
#    - Add long_memory_rgbd to batch.
#    - Add aux loss: ||action[:5] - pose_prior_extrapolated[:5]||² with weight 0.1
# 4. Stage freeze (logoplanner_policy.py:_apply_stage_freeze):
#    - For hybrid model: stage=1 freezes nothing new (Pi3 trains).
#    - Stage=2 freezes state_encoder (Pi3 ge.encoder + ge.decoder).
#    - LINGBOT_FREEZE_AGG still respected for long_memory_encoder.
# 5. Server (logoplanner_server.py):
#    - Accept long_memory_rgbd HTTP field (could just be larger memory_rgbd).
#    - May need bigger buffer in client (client_utils.py).
# 6. Sanity check: long_memory_size=48 GPU memory with LingBot ~ 24 GB needed?
#    Run smoke test first.
