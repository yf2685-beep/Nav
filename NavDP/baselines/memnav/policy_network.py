"""MemNav policy network.

`MemNav_Policy` = trainable cross-view retrieval head + NavDP/LoGoPlanner DDPM diffusion decoder
+ goal-agnostic critic. It is LoGoPlanner stripped of the Pi3 `GeometryModel` and the pointgoal
`start_encoder`, with the RGBD backbone replaced by LingBot memory tokens + a goal-grounded
context produced by the retrieval head.

Conditioning fed to the shared TransformerDecoder (matches NavDP layout):

    cond = [ time_embed(1), goal_context(G), memory_tokens(M), unify(?) ]

- `time_embed`  — diffusion timestep token
- `goal_context`— G tokens from the retrieval head (classifier-free: zeroed in the no-goal branch)
- `memory_tokens`— compressed LingBot history (anchors for old frames, window patches pooled by
  the head); variable length with a padding mask so we can extend context beyond NavDP's fixed 8
- the critic masks out time+goal slots via `cond_critic_mask` (goal-agnostic), as in NavDP.

This file is mirrored in the training repo
(`InternNav/internnav/model/basemodel/memnav/memnav_policy.py`); keep them in sync.

NOTE: NavDP's `internnav/.../navdp_policy.py` carries an active `pdb.set_trace()` in `forward`.
Do NOT copy it here.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from policy_backbone import (
    LearnablePositionalEncoding,
    SinusoidalPosEmb,
    TokenCompressor,
)


# --------------------------------------------------------------------------- #
# Cross-space retrieval head (the core trainable contribution)
# --------------------------------------------------------------------------- #
class CrossViewRetrievalHead(nn.Module):
    """Find "where the goal-looking place is" in memory and read out its geometry.

    Coarse-to-fine, decoupling the two concerns:
      1. Frame selection (coarse, all frames): goal `dino_desc` vs history `dino_desc`
         → attention logits over the context window. Supervised by the GT seen-frame index
         (cross-entropy, masked to seen samples). Yields top-k frames + soft weights.
      2. Dense cross-view (fine, top-k frames): goal `dino_patch` × selected frames' `dino_patch`
         → patch correspondence; read out the matched frames' `anchor`/geometry as values.

    Output: G goal-grounded context tokens (token_dim) for the diffusion decoder, plus the
    retrieval logits used by the auxiliary loss.

    Dims (frozen LingBot, img 518/14): dino D'=1024, patch P=1369, anchor 6×2048.

    TODO (decide with training):
      - projection dims q/k/v, number of cross-view heads, top-k.
      - G (context slot count): start G=1.
      - value space: `anchor` (frame-level, start here) vs `agg_patch` (patch-level).
      - learnable null key for the no-match / unseen case.
    """

    def __init__(self, token_dim=384, dino_dim=1024, anchor_dim=2048, n_goal_ctx=1, topk=4):
        super().__init__()
        self.token_dim = token_dim
        self.n_goal_ctx = n_goal_ctx
        self.topk = topk
        # TODO: proj_q/proj_k over dino space, proj_v over anchor space, dense cross-view attn,
        #       output projection to token_dim. Left unparameterized until training fixes shapes.

    def forward(self, goal_feat, mem_dino, mem_anchor, mem_mask, mem_patch=None, goal_patch=None):
        """
        goal_feat : {dino_desc [B,D'], dino_patch [B,P,D']}
        mem_dino  : [B, M, D']     history match keys (pooled)
        mem_anchor: [B, M, 6, 2C]  history geometry values
        mem_mask  : [B, M] bool    valid history slots
        mem_patch : [B, Wp, P, D'] optional full patches for window frames (dense cross-view)
        returns   : goal_context [B, G, token_dim], retrieval_logits [B, M]
        """
        raise NotImplementedError("retrieval head: coarse frame select + dense cross-view")


# --------------------------------------------------------------------------- #
# MemNav diffusion policy
# --------------------------------------------------------------------------- #
class MemNav_Policy(nn.Module):
    def __init__(
        self,
        memory_size=64,        # max history tokens conditioned on (variable, masked)
        context_goal=1,        # G goal-context slots
        predict_size=24,
        temporal_depth=12,
        heads=8,
        token_dim=384,
        dino_dim=1024,
        anchor_dim=2048,
        device="cuda:0",
    ):
        super().__init__()
        self.device = device
        self.memory_size = memory_size
        self.context_goal = context_goal
        self.predict_size = predict_size
        self.token_dim = token_dim

        # --- memory / goal encoders ---
        self.retrieval_head = CrossViewRetrievalHead(
            token_dim=token_dim, dino_dim=dino_dim, anchor_dim=anchor_dim, n_goal_ctx=context_goal
        )
        # Project a frame's 6 anchor tokens (6*2C) → one memory token of token_dim.
        self.mem_proj = nn.Linear(6 * anchor_dim, token_dim)

        # --- shared diffusion/critic decoder (NavDP layout) ---
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=4 * token_dim,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=temporal_depth)
        self.input_embed = nn.Linear(3, token_dim)          # noisy waypoints → tokens
        self.cond_len = 1 + context_goal + memory_size      # time + goal_ctx + memory
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, self.cond_len)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)          # ε-prediction
        self.critic_head = nn.Linear(token_dim, 1)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=10, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon",
        )
        # causal tgt mask over the 24 waypoint tokens
        tgt_mask = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float("-inf")).masked_fill(tgt_mask == 1, 0.0)
        # critic masks out time + goal slots (goal-agnostic), like NavDP cond_critic_mask
        self.cond_critic_mask = torch.zeros((predict_size, self.cond_len))
        self.cond_critic_mask[:, 0 : 1 + context_goal] = float("-inf")

    # ----- memory assembly -------------------------------------------------- #
    def encode_memory(self, mem_anchor, mem_mask):
        """[B,M,6,2C] anchors → [B,M,token_dim] memory tokens (+ [B,M] key_padding_mask)."""
        B, M = mem_anchor.shape[:2]
        tokens = self.mem_proj(mem_anchor.reshape(B, M, -1))
        key_padding = ~mem_mask.bool()
        return tokens, key_padding

    # ----- diffusion ε / critic (templated on LoGoPlanner_Policy) ----------- #
    def predict_noise(self, last_actions, timestep, goal_context, mem_tokens, mem_key_padding):
        raise NotImplementedError("ε-prediction over [time, goal_context, mem_tokens] cond")

    def predict_critic(self, predict_trajectory, mem_tokens, mem_key_padding):
        raise NotImplementedError("goal-agnostic critic over masked cond")

    @torch.no_grad()
    def predict_imagegoal_action(self, goal_feat, mem_dino, mem_anchor, mem_mask,
                                 mem_patch=None, goal_patch=None, sample_num=16):
        """Inference: retrieval → DDPM sampling (sample_num candidates) → critic ranking.

        Returns (all_trajectory, critic_values, positive_trajectory, negative_trajectory) like
        `LoGoPlanner_Policy.predict_pointgoal_action`, minus the sub-pointgoal. Trajectory is
        `cumsum(naction / 4.0)`.
        """
        raise NotImplementedError("retrieval + 10-step DDPM sampling + critic top-k ranking")

    # ----- training forward (ng/mg branches + critic), used by the trainer -- #
    def forward(self, goal_feat, mem_dino, mem_anchor, mem_mask,
                output_actions, augment_actions, retrieval_target=None,
                mem_patch=None, goal_patch=None):
        """Returns (noise_pred_ng, noise_pred_mg, cr_label, cr_augment, retrieval_logits, [noises]).

        - ng branch: goal_context zeroed (classifier-free) → drives the no-goal / unseen prior.
        - mg branch: goal_context from the retrieval head.
        - action_loss = 0.5*ng + 0.5*mg ε-MSE; critic on label+augment; retrieval CE on seen.
        Mirrors `NavDPNet.forward` (without its stray pdb.set_trace).
        """
        raise NotImplementedError("ng/mg ε branches + critic decode + retrieval logits")


if __name__ == "__main__":
    # structural smoke test (shapes only; bodies are TODO)
    policy = MemNav_Policy(device="cpu")
    print("MemNav_Policy built. cond_len =", policy.cond_len,
          "| params =", sum(p.numel() for p in policy.parameters()))
