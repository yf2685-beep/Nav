import os
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from policy_backbone import *
from geometry_model import GeometryModel

# Backbone selector. Both phase-1 LingBot methods are kept side by side here; they are
# alternative geometry backends chosen at runtime, never active together.
#   LOGO_BACKBONE=pi3         (default) original Pi3 path, unchanged
#   LOGO_BACKBONE=lingbot_map Method 1 — frozen LingBot-Map (GCTStream) + Adapter
#   LOGO_BACKBONE=lingbot_v2  Method 2 — AggregatorStream (frame attention + GCA)
#                             replacing Pi3's bidirectional decoder, DA-S depth-prior
#                             fusion preserved
_LOGO_BACKBONE = os.environ.get('LOGO_BACKBONE', 'pi3').lower()
# Stage selector, Method 1 only:
#   LOGO_STAGE=1 → build geometric heads on Adapter output, return real preds
#                  (pair with w_pose/w_local/w_world > 0, w_diffusion/critic/subgoal = 0)
#   LOGO_STAGE=2 (default) → no geometric heads, return dummy zeros (B 方案 / 论文 stage 2)
_LOGO_STAGE = int(os.environ.get('LOGO_STAGE', '2'))
# Streaming GCT navigation mode (set LOGO_STREAMING=1). When on, the policy drops
# the 8-frame NavDP memory backbone and conditions the diffusion head on
# GCT-summary tokens (anchor / trajectory-memory / pose-reference-window /
# current) assembled from the LingBot streaming backbone over the FULL episode,
# instead of 12 evenly-subsampled context frames. Requires LOGO_BACKBONE=lingbot_v2.
_LOGO_STREAMING = os.environ.get('LOGO_STREAMING', '0') == '1'
_LOGO_N_ANCHOR = int(os.environ.get('LOGO_N_ANCHOR', '8'))
_LOGO_N_WINDOW = int(os.environ.get('LOGO_N_WINDOW', '64'))
if _LOGO_BACKBONE == 'lingbot_map':
    from lingbot_map_geometry import LingBotMapGeometryModel
elif _LOGO_BACKBONE == 'lingbot_v2':
    from geometry_model_lingbot import GeometryModel_LingBot
if _LOGO_STREAMING:
    from streaming_gct import GCTSummaryAssembler, StreamingContextBuffer, partition_window_tokens

class FeatureCompressor(nn.Module):
    """Project backbone features (2048-d) → compact retrieval space (128-d, L2-normed).

    Trained via InfoNCE: positive = own-episode best-matching past frame,
    negatives = other-trajectory frames + top-K hard mining.
    Output is L2-normalized so dot product == cosine similarity.
    """

    def __init__(self, in_dim: int = 2048, out_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class LoGoPlanner_Policy(nn.Module):
    def __init__(self,
                 image_size=224,
                 memory_size=8,
                 context_size=12,
                 predict_size=24,
                 temporal_depth=8,
                 heads=8,
                 token_dim=384,
                 channels=3,
                 use_depth=True,
                 device='cuda:0'):
        super().__init__()
        self.device = device
        self.image_size = image_size
        self.memory_size = memory_size
        self.context_size = context_size
        self.predict_size = predict_size
        self.temporal_depth = temporal_depth
        self.attention_heads = heads
        self.input_channels = channels
        self.token_dim = token_dim
        # Stage 1: RGB-only trajectory backbone when False; state_encoder
        # (Pi3/LingBot geometry) keeps its own depth prior independently.
        self.use_depth = use_depth

        # --- Streaming GCT navigation mode --------------------------------
        self._streaming = _LOGO_STREAMING
        self.n_anchor = _LOGO_N_ANCHOR
        self.n_window = _LOGO_N_WINDOW
        if self._streaming:
            assert _LOGO_BACKBONE == 'lingbot_v2', \
                'LOGO_STREAMING=1 requires LOGO_BACKBONE=lingbot_v2'

        # input encoders
        # Streaming mode drops the NavDP 8-frame memory backbone entirely — the
        # GCT pose-reference window already carries recent-frame context.
        if not self._streaming:
            self.rgbd_encoder = NavDP_RGBD_Backbone(image_size,token_dim,memory_size=memory_size,use_depth=use_depth,device=device)
        if _LOGO_BACKBONE == 'lingbot_map':
            # Method 1: frozen LingBot-Map (GCTStream) + Adapter.
            self.state_encoder = LingBotMapGeometryModel(
                context_size=context_size,
                device=device,
                stage1_heads=(_LOGO_STAGE == 1),
            )
        elif _LOGO_BACKBONE == 'lingbot_v2':
            # Method 2: AggregatorStream + DA-S depth-prior fusion preserved.
            self.state_encoder = GeometryModel_LingBot(
                context_size=context_size, device=device, n_anchor=self.n_anchor,
            )
        else:
            self.state_encoder = GeometryModel(context_size=context_size,device=device)

        if self._streaming:
            # Trainable assembler: per-frame state/scene tokens -> 8 summary tokens.
            self.gct_assembler = GCTSummaryAssembler(dim=token_dim, heads=heads)
            # Per-environment inference buffers (created lazily on reset).
            self.stream_buffers = None
            # Run the (frozen, pretrained) geometry backbone under no_grad during
            # streaming training. Correct for Stage-2 (geometry not supervised);
            # cuts activation memory ~10x so the N=anchor+traj+window window fits.
            # Set LOGO_STREAM_BACKBONE_GRAD=1 to keep grad (Stage-1 geometry
            # supervision over the window).
            self._stream_backbone_nograd = os.environ.get('LOGO_STREAM_BACKBONE_GRAD', '0') != '1'
            # Training-time window encode: 'parallel' (one batched causal forward
            # over the whole window — fast, needs more GPU memory) vs the default
            # frame-by-frame streaming encode (bounded memory, but N sequential
            # forwards/sample → slow). Use parallel on big-memory GPUs (e.g. GB10).
            self._stream_train_parallel = os.environ.get('LOGO_STREAM_TRAIN_PARALLEL', '0') == '1'
        self.point_encoder = nn.Linear(3,self.token_dim)

        self.start_encoder = nn.Linear(3,self.token_dim)

        # === v3: FeatureCompressor + cosine hard retrieval + concat-or-null ===
        # Compressor maps frozen backbone per-frame features (2048-d mean-pooled)
        # to a compact L2-normed space (128-d) for cosine retrieval.
        # Training signal: InfoNCE (own-episode best match positive, cross-traj negatives).
        # retrieved state_token is concat'd to state_embed channel-wise → linear back to D.
        self.feature_compressor = FeatureCompressor(in_dim=2048, out_dim=128)
        self.retrieval_threshold_logit = nn.Parameter(torch.tensor(0.0))   # sigmoid(0)=0.5
        self.null_retrieval = nn.Parameter(torch.zeros(1, 1, self.token_dim))
        nn.init.normal_(self.null_retrieval, std=0.02)
        self.retrieval_concat_proj = nn.Linear(2 * self.token_dim, self.token_dim)

        # === Phase α-Fix++: image-goal via frozen ResNet18 + distill from start_encoder ===
        # Previous attempts:
        #   - random CNN  → goal-blind (138325)
        #   - tile×12 through LingBot state_encoder → still goal-blind (138328)
        # Root cause: LingBot's backbone is trained for VIDEO sequences with
        # camera motion. Static-image tile is OOD; output is near-constant.
        # Fix: use ResNet18 (ImageNet pretrained, frozen) as a true semantic
        # image encoder. Plus add distillation: train goal_image_proj to
        # mimic start_encoder(point_goal) — mirrors Plan D's Pi3 distillation.
        self._imagegoal_mode = os.environ.get('IMAGEGOAL_MODE', '0') == '1'
        if self._imagegoal_mode:
            import torchvision.models as tvm
            _resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
            _resnet.fc = nn.Identity()  # output (B, 512)
            for _p in _resnet.parameters():
                _p.requires_grad = False
            self.goal_image_backbone = _resnet
            self.goal_image_proj = nn.Linear(512, self.token_dim)
            # Freeze start_encoder so it stays a stable distillation teacher.
            for _p in self.start_encoder.parameters():
                _p.requires_grad = False
            # ImageNet normalize stats — registered as buffers so they move with model.
            self.register_buffer('_imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('_imagenet_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        # === Stage A: explicit metric tokens ===
        # distance to goal (scalar) and min obstacle distance in front (scalar).
        # Toggled via env STAGE_A_EXPLICIT_METRIC=1; when off, behaves identically
        # to the original LoGoPlanner.
        self._stage_a_metric = os.environ.get('STAGE_A_EXPLICIT_METRIC', '0') == '1'
        if self._stage_a_metric:
            self.dist_encoder = nn.Linear(1, self.token_dim)
            self.obstacle_encoder = nn.Linear(1, self.token_dim)
        self.state_decoder = TokenCompressor(embed_dim=token_dim,
                                              num_heads=heads,
                                              target_length=1)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = token_dim,
                                                        nhead = heads,
                                                        dim_feedforward = 4 * token_dim,
                                                        activation = 'gelu',
                                                        batch_first = True,
                                                        norm_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer = self.decoder_layer,
                                             num_layers = self.temporal_depth)
        
        self.input_embed = nn.Linear(3,token_dim)
        self.pg_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim//2),
            nn.ReLU(),
            nn.Linear(token_dim//2, token_dim//4),
            nn.ReLU(),
            nn.Linear(token_dim//4, 3)
        )
        self.cs_pred_mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim//2),
            nn.ReLU(),
            nn.Linear(token_dim//2, token_dim//4),
            nn.ReLU(),
            nn.Linear(token_dim//4, 3)
        )
        
        # cond length = 1 (time) + 3 (goal slots) + M (rgbd) + 2C (state+scene)
        # + (2 if Stage A else 0) for [distance, obstacle] appended at the END.
        _cond_extra = 2 if self._stage_a_metric else 0
        if self._streaming:
            # cond = time(1) + goal(3) + GCT summary(8). No rgbd, no per-frame unify.
            self._cond_len = 4 + GCTSummaryAssembler.N_SUMMARY
        else:
            self._cond_len = memory_size + context_size*2 + 4 + _cond_extra
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, self._cond_len)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)

        self.action_head = nn.Linear(token_dim, 3)
        self.critic_head = nn.Linear(token_dim, 1)
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=10,
                                       beta_schedule='squaredcos_cap_v2',
                                       clip_sample=True,
                                       prediction_type='epsilon')

        self.tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = self.tgt_mask.float().masked_fill(self.tgt_mask == 0, float('-inf')).masked_fill(self.tgt_mask == 1, float(0.0))
        self.cond_critic_mask = torch.zeros((predict_size, self._cond_len))
        # mask positions 0..3 (time + 3 goal slots = the goal-conditioned channels)
        # dist/obstacle slots at the END are NOT masked: critic should see them.
        self.cond_critic_mask[:,0:4] = float('-inf')
    
    def predict_noise(self,last_actions,timestep,goal_embed,rgbd_embed,unify_token,
                      dist_embed=None,obstacle_embed=None,summary=None):
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(self.device)).unsqueeze(1).tile((last_actions.shape[0],1,1))
        if self._streaming:
            # cond = time(1) + goal(3) + GCT summary(8); no rgbd/unify in streaming.
            assert summary is not None, 'streaming predict_noise needs summary'
            cond_cat = torch.cat([time_embeds, goal_embed, goal_embed, goal_embed, summary], dim=1)
            cond_embedding = cond_cat + self.cond_pos_embed(cond_cat)
            input_embedding = action_embeds + self.out_pos_embed(action_embeds)
            output = self.decoder(tgt=input_embedding, memory=cond_embedding, tgt_mask=self.tgt_mask.to(self.device))
            return self.action_head(self.layernorm(output))
        cond_parts = [time_embeds, goal_embed, goal_embed, goal_embed, rgbd_embed, unify_token]
        if self._stage_a_metric:
            assert dist_embed is not None and obstacle_embed is not None, "Stage A on but dist/obstacle missing"
            cond_parts += [dist_embed, obstacle_embed]
        cond_cat = torch.cat(cond_parts, dim=1)
        cond_embedding = cond_cat + self.cond_pos_embed(cond_cat)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        output = self.decoder(tgt = input_embedding,memory = cond_embedding, tgt_mask = self.tgt_mask.to(self.device))
        output = self.layernorm(output)
        output = self.action_head(output)
        return output

    def predict_critic(self,predict_trajectory,rgbd_embed,unify_token,
                       dist_embed=None,obstacle_embed=None,summary=None):
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        if self._streaming:
            assert summary is not None, 'streaming predict_critic needs summary'
            nogoal_embed = torch.zeros_like(summary[:, 0:1])
            cond_cat = torch.cat([nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, summary], dim=1)
            cond_embeddings = cond_cat + self.cond_pos_embed(cond_cat)
            critic_output = self.decoder(tgt=action_embeddings, memory=cond_embeddings, memory_mask=self.cond_critic_mask.to(self.device))
            critic_output = self.layernorm(critic_output)
            return self.critic_head(critic_output.mean(dim=1))[:, 0]
        nogoal_embed = torch.zeros_like(rgbd_embed[:,0:1])
        cond_parts = [nogoal_embed, nogoal_embed, nogoal_embed, nogoal_embed, rgbd_embed, unify_token]
        if self._stage_a_metric:
            assert dist_embed is not None and obstacle_embed is not None, "Stage A on but dist/obstacle missing"
            cond_parts += [dist_embed, obstacle_embed]
        cond_cat = torch.cat(cond_parts, dim=1)
        cond_embeddings = cond_cat + self.cond_pos_embed(cond_cat)
        critic_output = self.decoder(tgt = action_embeddings, memory = cond_embeddings, memory_mask = self.cond_critic_mask.to(self.device))
        critic_output = self.layernorm(critic_output)
        critic_output = self.critic_head(critic_output.mean(dim=1))[:,0]
        return critic_output
    
    def predict_imagegoal_action(self, goal_image, memory_rgbd, context_rgbd, sample_num=16):
        """Image-goal inference path (Phase α).

        goal_image: (1, H, W, 3) float in [0, 255] or [0, 1]  — the target view.
        memory_rgbd, context_rgbd: same as predict_pointgoal_action.
        Returns the same 5-tuple shape as predict_pointgoal_action so the agent /
        server stays uniform.
        """
        assert self._imagegoal_mode, "predict_imagegoal_action called but IMAGEGOAL_MODE=0"
        with torch.no_grad():
            gi = torch.as_tensor(goal_image[0:1], dtype=torch.float32, device=self.device)
            if gi.max() > 1.5:
                gi = gi / 255.0
            # Frozen ResNet18 semantic features + Linear adapter.
            gi_BCHW = gi.permute(0, 3, 1, 2).contiguous()  # (1, 3, H, W)
            gi_norm = (gi_BCHW - self._imagenet_mean) / self._imagenet_std
            feat = self.goal_image_backbone(gi_norm)        # (1, 512)
            # Inference-time norm scaling: student goal_token learned to be
            # goal-distinct but with norm ~9 vs teacher start_encoder ~34.
            # Multiply by GOAL_SCALE so diffusion (trained on teacher-norm cond)
            # actually reads the goal signal. Env-tunable.
            _gscale = float(os.environ.get('GOAL_TOKEN_SCALE', '4.0'))
            startgoal_embed = self.goal_image_proj(feat).unsqueeze(1) * _gscale  # (1, 1, D)

            _mem_depth = memory_rgbd[0:1, -1][..., 3:4] if self.use_depth else None
            rgbd_embed = self.rgbd_encoder(memory_rgbd[0:1][..., :3], _mem_depth)
            [hidden, state_token, scene_token], _ = self.state_encoder(
                context_rgbd[0:1][..., :3], context_rgbd[0:1][..., 3:4]
            )
            unify_token = torch.cat([state_token, scene_token], dim=1)
            state_embed = self.state_decoder(torch.cat([state_token, startgoal_embed], dim=1))
            sub_pointgoal_pd = self.pg_pred_mlp(state_embed).squeeze(1)

            # Stage A metric tokens — distance is unknown in imagegoal mode (we
            # don't have a metric goal), use a placeholder = 0.0 so the same
            # cond shape holds. Obstacle is still depth-derived.
            dist_embed = obstacle_embed = None
            if self._stage_a_metric:
                _zero = torch.zeros((1, 1), device=self.device, dtype=torch.float32)
                dist_embed = self.dist_encoder(_zero).unsqueeze(1)
                _last_depth = memory_rgbd[0:1, -1, ..., 3:4]
                _last_depth_t = torch.as_tensor(_last_depth, dtype=torch.float32, device=self.device)
                _flat = _last_depth_t.reshape(_last_depth_t.shape[0], -1)
                _mask = _flat > 1e-4
                _flat = torch.where(_mask, _flat, torch.full_like(_flat, 1e6))
                _min_obs = _flat.min(dim=-1, keepdim=True).values.clamp(max=10.0)
                obstacle_embed = self.obstacle_encoder(_min_obs).unsqueeze(1)

            rgbd_embed = torch.repeat_interleave(rgbd_embed, sample_num, dim=0)
            state_embed = torch.repeat_interleave(state_embed, sample_num, dim=0)
            unify_token = torch.repeat_interleave(unify_token, sample_num, dim=0)
            if self._stage_a_metric:
                dist_embed = torch.repeat_interleave(dist_embed, sample_num, dim=0)
                obstacle_embed = torch.repeat_interleave(obstacle_embed, sample_num, dim=0)

            B = 1
            noisy_action = torch.randn((sample_num * B, self.predict_size, 3), device=self.device)
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction, k.unsqueeze(0), state_embed,
                                                rgbd_embed, unify_token,
                                                dist_embed=dist_embed, obstacle_embed=obstacle_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample
            critic_values = self.predict_critic(naction, rgbd_embed, unify_token,
                                                dist_embed=dist_embed, obstacle_embed=obstacle_embed)
            critic_values = critic_values.reshape(B, sample_num)

            all_trajectory = torch.cumsum(naction / 4.0, dim=1).reshape(B, sample_num, self.predict_size, 3)
            traj_len = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
            all_trajectory[traj_len < 0.5] = all_trajectory[traj_len < 0.5] * torch.tensor([[[0, 0, 1.0]]], device=all_trajectory.device)
            sorted_idx = (-critic_values).argsort(dim=1)
            top2 = sorted_idx[:, 0:2]
            bi = torch.arange(B).unsqueeze(1).expand(-1, 2)
            pos_traj = all_trajectory[bi, top2]
            neg_traj = all_trajectory[bi, sorted_idx[:, -2:]]

            return (all_trajectory.cpu().numpy(),
                    critic_values.cpu().numpy(),
                    pos_traj.cpu().numpy(),
                    neg_traj.cpu().numpy(),
                    sub_pointgoal_pd.cpu().numpy())

    # ===================================================================
    # Streaming GCT inference (LOGO_STREAMING=1)
    # ===================================================================
    def reset_streaming(self, batch_size):
        """Start fresh episodes: clear backbone KV cache + per-env token buffers.

        NOTE: the LingBot KV cache is shared across the batch, so all envs must
        reset together (lock-step episodes). Per-env episode boundaries within a
        shared batch are not supported (same constraint as the base backbone).
        """
        assert self._streaming, 'reset_streaming called but LOGO_STREAMING=0'
        self.state_encoder.reset_kv_cache()
        self.stream_buffers = [
            StreamingContextBuffer(n_anchor=self.n_anchor, n_window=self.n_window)
            for _ in range(batch_size)
        ]

    def _build_summary_stream(self, image, depth, episode_start):
        """Stream ONE new frame per env through the backbone + update buffers.

        Args:
            image: (B, H, W, 3) or (B, 1, H, W, 3) current RGB.
            depth: (B, H, W, 1) or (B, 1, H, W, 1) current depth.
            episode_start: bool — clears KV cache + buffers before this frame.
        Returns:
            summary: (B, 8, D)
            cur_state_token: (B, 1, D) — current-frame state token.
        """
        B = image.shape[0]
        if episode_start or self.stream_buffers is None:
            self.reset_streaming(B)
        # Backbone single-frame streaming (persistent KV cache, no reset here).
        state_token, scene_token = self.state_encoder.step_streaming(
            image, depth, episode_start=episode_start,
        )  # each (B, 1, D)
        summaries = []
        for b in range(B):
            self.stream_buffers[b].step(state_token[b, 0], scene_token[b, 0])
            parts = self.stream_buffers[b].build_parts()
            summaries.append(self.gct_assembler(parts))  # (1, 8, D)
        summary = torch.cat(summaries, dim=0)  # (B, 8, D)
        return summary, state_token

    @torch.no_grad()
    def predict_pointgoal_action_stream(self, start_goal, image, depth,
                                        episode_start=False, sample_num=16):
        """Streaming point-goal inference. One observation per call.

        Mirrors :meth:`predict_pointgoal_action` (same 5-tuple return) but the
        geometric conditioning is the GCT summary over the full episode so far,
        not 12 subsampled frames.
        """
        B = start_goal.shape[0]
        summary, cur_state = self._build_summary_stream(image, depth, episode_start)

        tensor_start_goal = torch.as_tensor(start_goal, dtype=torch.float32, device=self.device)
        startgoal_embed = self.start_encoder(tensor_start_goal).unsqueeze(1)  # (B,1,D)
        state_embed = self.state_decoder(torch.cat([cur_state, startgoal_embed], dim=1))  # (B,1,D)
        sub_pointgoal_pd = self.pg_pred_mlp(state_embed).squeeze(1)  # (B,3)

        summary = torch.repeat_interleave(summary, sample_num, dim=0)
        state_embed = torch.repeat_interleave(state_embed, sample_num, dim=0)

        noisy_action = torch.randn((sample_num * B, self.predict_size, 3), device=self.device)
        naction = noisy_action
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        for k in self.noise_scheduler.timesteps[:]:
            noise_pred = self.predict_noise(naction, k.unsqueeze(0), state_embed,
                                            None, None, summary=summary)
            naction = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=naction).prev_sample

        critic_values = self.predict_critic(naction, None, None, summary=summary)
        critic_values = critic_values.reshape(B, sample_num)

        all_trajectory = torch.cumsum(naction / 4.0, dim=1).reshape(B, sample_num, self.predict_size, 3)
        trajectory_length = all_trajectory[:, :, -1, 0:2].norm(dim=-1)
        all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0, 0, 1.0]]], device=all_trajectory.device)

        sorted_indices = (-critic_values).argsort(dim=1)
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, 2)
        positive_trajectory = all_trajectory[batch_indices, sorted_indices[:, 0:2]]
        negative_trajectory = all_trajectory[batch_indices, (critic_values).argsort(dim=1)[:, 0:2]]

        # .float() guards against bf16 (autocast) which numpy() cannot convert.
        return (all_trajectory.float().cpu().numpy(), critic_values.float().cpu().numpy(),
                positive_trajectory.float().cpu().numpy(), negative_trajectory.float().cpu().numpy(),
                sub_pointgoal_pd.float().cpu().numpy())

    def predict_pointgoal_action(self,start_goal,memory_rgbd,context_rgbd,sample_num=16):
        with torch.no_grad():
            tensor_start_goal = torch.as_tensor(start_goal[0:1],dtype=torch.float32,device=self.device)
            startgoal_embed = self.start_encoder(tensor_start_goal).unsqueeze(1)
            _mem_depth = memory_rgbd[0:1, -1][..., 3:4] if self.use_depth else None
            rgbd_embed = self.rgbd_encoder(memory_rgbd[0:1][..., :3], _mem_depth)
            [hidden, state_token, scene_token], [camera_poses, local_points, world_points] = self.state_encoder(context_rgbd[0:1][..., :3], context_rgbd[0:1][..., 3:4]) # (B, 16*T, D)
            unify_token = torch.cat([state_token, scene_token], dim=1) # (B, T*3, D)
            
            state_embed = self.state_decoder(torch.cat([state_token, startgoal_embed], dim=1)) # (B, 1, D)
            sub_pointgoal_pd = self.pg_pred_mlp(state_embed).squeeze(1) # (B, 3)

            # === Stage A: explicit metric tokens ===
            dist_embed = obstacle_embed = None
            if self._stage_a_metric:
                # distance to goal = norm of xy components (start_goal in robot frame)
                distance = tensor_start_goal[:, :2].norm(dim=-1, keepdim=True)  # (B, 1)
                dist_embed = self.dist_encoder(distance).unsqueeze(1)  # (B, 1, D)
                # min obstacle distance: global min on flattened last memory frame depth
                # (robust to layout) with zeros masked as invalid and clamped to 10m.
                _last_depth = memory_rgbd[0:1, -1, ..., 3:4]  # (1, H, W, 1) at this call site
                _last_depth_t = torch.as_tensor(_last_depth, dtype=torch.float32, device=self.device)
                _flat = _last_depth_t.reshape(_last_depth_t.shape[0], -1)
                _mask = _flat > 1e-4
                _flat = torch.where(_mask, _flat, torch.full_like(_flat, 1e6))
                _min_obs = _flat.min(dim=-1, keepdim=True).values.clamp(max=10.0)  # (1, 1)
                obstacle_embed = self.obstacle_encoder(_min_obs).unsqueeze(1)  # (B, 1, D)

            rgbd_embed = torch.repeat_interleave(rgbd_embed,sample_num,dim=0) # Tiles cond tensor sample_num times along batch dim so all 16 diffusion samples can be denoised in one forward pass.
            state_embed = torch.repeat_interleave(state_embed,sample_num,dim=0)
            unify_token = torch.repeat_interleave(unify_token,sample_num,dim=0)
            if self._stage_a_metric:
                dist_embed = torch.repeat_interleave(dist_embed, sample_num, dim=0)
                obstacle_embed = torch.repeat_interleave(obstacle_embed, sample_num, dim=0)

            noisy_action = torch.randn((sample_num * start_goal.shape[0], self.predict_size, 3), device=self.device) # Gaussian noise of shape (16, 24, 3) — 16 candidate traj, 24 waypoints each
            naction = noisy_action
            self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
            for k in self.noise_scheduler.timesteps[:]:
                noise_pred = self.predict_noise(naction,k.unsqueeze(0),state_embed,rgbd_embed,unify_token,
                                                dist_embed=dist_embed, obstacle_embed=obstacle_embed)
                naction = self.noise_scheduler.step(model_output=noise_pred,timestep=k,sample=naction).prev_sample

            critic_values = self.predict_critic(naction,rgbd_embed,unify_token,
                                                dist_embed=dist_embed, obstacle_embed=obstacle_embed)
            critic_values = critic_values.reshape(start_goal.shape[0],sample_num)
            
            all_trajectory = torch.cumsum(naction / 4.0, dim=1)
            all_trajectory = all_trajectory.reshape(start_goal.shape[0],sample_num,self.predict_size,3)
            trajectory_length = all_trajectory[:,:,-1,0:2].norm(dim=-1)
            all_trajectory[trajectory_length < 0.5] = all_trajectory[trajectory_length < 0.5] * torch.tensor([[[0,0,1.0]]],device=all_trajectory.device)
            
            sorted_indices = (-critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(start_goal.shape[0]).unsqueeze(1).expand(-1, 2)
            positive_trajectory = all_trajectory[batch_indices, topk_indices]
            
            sorted_indices = (critic_values).argsort(dim=1)
            topk_indices = sorted_indices[:,0:2]
            batch_indices = torch.arange(start_goal.shape[0]).unsqueeze(1).expand(-1, 2)
            negative_trajectory = all_trajectory[batch_indices, topk_indices]
            
            return all_trajectory.cpu().numpy(), critic_values.cpu().numpy(), positive_trajectory.cpu().numpy(), negative_trajectory.cpu().numpy(), sub_pointgoal_pd.cpu().numpy()
    
if __name__ == "__main__":
    policy = LoGoPlanner_Policy()
    policy = policy.to("cuda:0")
    memory_rgbd = torch.rand(1,8,168,308,4).to("cuda:0")
    context_rgbd = torch.rand(1,12,168,308,4).to("cuda:0")
    start_goal = torch.zeros((1,3), device="cuda:0")  # Example start goal (x, y, theta)
    
    all_trajectory, critic_values, positive_trajectory, negative_trajectory, sub_pointgoal_pd = policy.predict_pointgoal_action(start_goal, memory_rgbd, context_rgbd)
    
    print("All Trajectory Shape:", all_trajectory.shape)
    print("Critic Values Shape:", critic_values.shape)
    print("Positive Trajectory Shape:", positive_trajectory.shape)
    print("Negative Trajectory Shape:", negative_trajectory.shape)