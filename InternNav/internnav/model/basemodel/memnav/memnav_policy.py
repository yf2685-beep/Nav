"""MemNav policy — trainable head over the frozen LingBotStream front-end.

Three goal pathways (see GL.md / memnav-project memory):
  (1) backbone current state      — frozen GCT (LingBotStream.window_forward)
  (2) revisit goal→history        — frozen GCT (LingBotStream.goal_append), visited goals
  (3) novel current→goal (DINO)   — TRAINABLE cross-attention, unseen goals
Retrieval confidence biases the decoder cross-attention toward (2) vs (3) (no multiply,
no goal_cls). NavDP DDPM decoder on top; NO critic (collision is geometric at eval).
Always goal-conditioned — no classifier-free "no-goal" branch (dropped: our goal-directed
two-leg episodes can require a genuine U-turn, so masking the goal out of the label gave
the unconditional branch contradictory supervision — same visual context, opposite action,
depending on whether that episode happened to reverse — worst exactly at the turn where a
CFG contrast would matter most; CFG guidance scale was never benchmarked for this model).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import PretrainedConfig, PreTrainedModel

from internnav.model.basemodel.memnav.cache_schema import validate_cache_pair
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream, ground_scale_from_h_est
from internnav.model.encoder.navdp_backbone import (
    LearnablePositionalEncoding,
    NavDP_ImageGoal_Backbone,
    SinusoidalPosEmb,
    TokenCompressor,
)


# --------------------------------------------------------------------------- #
# (2.retrieval) Target-image retrieval over dino_cls — trainable, supervised
# --------------------------------------------------------------------------- #
class RetrievalHead(nn.Module):
    """goal_cls vs mem_cls (history CLS) over the revisit CANDIDATE set → decoupled
    ranking logits + a separate revisit/novel gate.

    Two jobs, DECOUPLED (a joint softmax with a null slot collapses to always-null):
      * RANKING  : cosine(goal, mem)/temp over candidate frames -> ret_logits [B,L].
                   Multi-positive InfoNCE (revisit rows) is applied on these.
      * GATE     : the revisit/novel decision is an AFFINE readout of the single most
                   similar candidate: g = a·max_i cos_i + b -> BCE(sigmoid(g), revisit).
                   (Probe: absolute top-1 cosine over the candidate region separates
                   revisit vs novel at AUC≈0.91; within-scene contrast / peak-sharpness
                   do NOT — so gate on the max, not a shape statistic.)
    The candidate set (mem_mask here = cand_mask from the loader) already excludes the
    recent approach window + <anchor_margin, which is what makes max-cos discriminative.

      - match_idx : argmax candidate frame (drives LingBotStream.goal_append)
      - gate_logit: [B] pre-sigmoid revisit logit (BCE target)
      - ret_logits: [B, L] cosine/temp over candidates (-inf elsewhere) for InfoNCE
    """

    def __init__(self, dino_dim=1024, proj_dim=256, temp_init=0.07):
        super().__init__()
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))
        # affine gate on the max candidate cosine: sigmoid(a·max_cos + b) = P(revisit)
        self.gate_a = nn.Parameter(torch.tensor(10.0))
        self.gate_b = nn.Parameter(torch.tensor(-8.0))

    def forward(self, goal_cls, mem_cls, cand_mask):
        """goal_cls [B,D'], mem_cls [B,L,D'], cand_mask [B,L] bool (revisit candidates)."""
        gq = F.normalize(self.proj_goal(goal_cls), dim=-1)        # [B,d]
        mk = F.normalize(self.proj_mem(mem_cls), dim=-1)          # [B,L,d]
        temp = self.log_temp.exp().clamp(0.01, 1.0)

        cos = (gq.unsqueeze(1) * mk).sum(-1)                      # [B,L] raw cosine (finite)
        NEG_INF = torch.finfo(cos.dtype).min
        # mask AFTER dividing by temp with a FINITE floor: putting -inf through /temp
        # makes 0*inf = nan flow into log_temp on backward (masked_fill zeros the upstream
        # grad, but the local d(-inf/temp)/dtemp = inf).
        ret_logits = (cos / temp).masked_fill(~cand_mask, NEG_INF)  # ranking logits over candidates

        # gate feature = max candidate cosine (finite floor for all-masked rows)
        has_cand = cand_mask.any(-1)                             # [B]
        max_cos = cos.masked_fill(~cand_mask, -1.0).max(-1).values  # [B] in [-1,1]
        max_cos = torch.where(has_cand, max_cos, max_cos.new_full((), -1.0))
        gate_logit = self.gate_a * max_cos + self.gate_b        # [B]

        match_idx = ret_logits.argmax(-1)                       # best candidate frame
        return match_idx, gate_logit, ret_logits


# --------------------------------------------------------------------------- #
# (3.novel) current DINO  →  goal DINO  cross-attention — trainable
# --------------------------------------------------------------------------- #
class NovelBranch(nn.Module):
    """Early-fusion goal↔current (NavDP_ImageGoal_Backbone design): 6-ch `concat(current, goal)`
    is **jointly** encoded by a trainable DINOv2-S (the 6-ch `patch_embed.proj` mixes the two
    images from layer 0 — true early fusion, the optical-flow-friendly inductive bias), → patch
    tokens → TokenCompressor → m_novel tokens. For unseen/overlapping goals; the diffusion reads
    the heading toward goal-matching content. (skips NavDP's mean-pool to keep spatial info.)
    """

    def __init__(self, dim=384, heads=8, out_tokens=4, image_size=224, device="cuda"):
        super().__init__()
        self.backbone = NavDP_ImageGoal_Backbone(image_size=image_size, embed_size=dim, device=device)
        self.backbone.project_layer = nn.Identity()              # unused (we skip NavDP's mean-pool)
        self.image_size = image_size
        self.proj = nn.Linear(384, dim)                          # DINOv2-S patch dim -> token_dim
        self.compress = TokenCompressor(dim, heads, out_tokens)

    def forward(self, cur_img, goal_img):
        """cur_img, goal_img: [B, 3, H, W] in [0,1] -> readout [B, out_tokens, dim]."""
        sz = (self.image_size, self.image_size)
        cur = F.interpolate(cur_img, size=sz, mode="bilinear", align_corners=False)
        goal = F.interpolate(goal_img, size=sz, mode="bilinear", align_corners=False)
        six = torch.cat([cur, goal], dim=1)                      # [B, 6, H, W]  early fusion
        patch = self.backbone.imagegoal_encoder.get_intermediate_layers(six)[0]  # [B, N, 384] (no pool)
        return self.compress(self.proj(patch))                   # [B, out_tokens, dim]


# --------------------------------------------------------------------------- #
# (2.merge) Revisit: analytic relative pose -> decoder tokens + calibrated (x,y)
# --------------------------------------------------------------------------- #
class RevisitMerge(nn.Module):
    """Turns the **current** and **goal** absolute camera poses (frozen camera head, map
    frame) into the goal's relative pose, analytically — NOT via independently-embedded
    absolute-pose tokens merged by attention. T_cur^-1 T_goal is BILINEAR in the two
    absolute poses (t_rel = R_cur^T(t_goal - t_cur) is a product of a rotation derived
    from cur_pose and a translation difference derived from both); a linear embed of each
    pose + attention-merge can only produce affine combinations of the two, and can never
    synthesize that cross term. So it's computed here in closed form (`_relative_pose`),
    same reasoning as VGGT/Pi3 supervising relative pose directly.

      - revisit_head  → revisit_readout (the diffusion goal slot). TRAINABLE: a plain
        Linear on [t_rel, R_rel.flatten()] (12-d) — no attention needed for a single
        input feature vector (TokenCompressor would degenerate to per-slot linear reads
        of it anyway).
      - aux_pose_head → (x, y) ONLY, not θ. θ (net heading change along the path from
        departure to arrival) is NOT a function of the two endpoint poses — it depends on
        the geodesic route's shape between them (obstacle layout), which two poses don't
        encode; that's the diffusion decoder's job (it sees current_state's depth/visual
        context), not RevisitMerge's. And the goal image's own rendered orientation is
        independent of the real arrival heading by construction of the data generator
        (MemNavData/generate_twoleg.py: "NO terminal orientation alignment... arrival
        heading is the natural approach heading"; goal_yaw = anchor's OWN heading +
        random jitter) — so there is no θ signal in (cur_pose, goal_pose) to extract even
        in principle.
        aux_pose_head's calibration mode is `aux_pose_calibration` ('empirical' | 'trainable'):
        cur_pose/goal_pose come from the frozen camera head under no_grad, so t_rel itself
        carries no gradient either way — but aux_pose_head is a plain nn.Linear with its
        OWN weight/bias, which DOES receive a local gradient from w_aux*aux_loss
        (MemNavTrainer.compute_loss) regardless of the no_grad upstream of it; 'trainable'
        just stops blocking that local gradient. Per-video LingBot scale ambiguity (see
        below) means neither mode can be exact per-sample, only a pooled compromise.
        'empirical': frozen at the fitted constant below (pure diagnostic, matches the
        original design). 'trainable': same init, but allowed to adapt its own weight/bias
        during training — since t_rel/R_rel carry no gradient, this can only re-fit the
        SAME kind of global affine map a precomputed constant would, so it is not expected
        to beat 'empirical' by much; it exists so both can be compared directly.

        Axis convention + scale were refit 2026-07-20 against the corrected GT (axis/mount,
        angle-wrap, and goal-endpoint fixes — see MEMNAV_AXIS_AND_ANGLE_WRAP.md) using an
        origin-anchored 2D Procrustes fit (rotation + scale, no reflection) of real t_rel
        vs real goal_rel_pose, over 14 clean mp3d_2leg validate_gated samples (3-leg
        excluded: noisier, deeper revisits). The fit converged to a rotation within ~5 deg
        of the clean axis-permutation `[t_rel_z, -t_rel_x]` already used independently by
        revisit_pose.py's GaugeInvariantRevisitPose and predicted from first principles —
        NOT the old _R_CONV, which was fit against pre-fix GT contaminated by the identity
        camera_extrinsic bug (axis fix item 1) and is stale.
        Scale is NOT shared across videos: the same 14 samples' per-sample scale ratio
        (|goal_rel_pose| / |t_rel_xz|) ranges 0.96-5.51x (std/mean=39%, i.e. LingBot's
        monocular scale is a per-trajectory quantity, not a global constant — expected,
        since t_rel alone gives the head no way to know which video it's in). _SCALE below
        is the pooled least-squares compromise across those samples, not a precise
        per-trajectory calibration.
        Ground-anchored per-trajectory scale (`scale_mode='ground'`, the default):
        instead of the pooled constant, t_rel is multiplied by a PER-TRAJECTORY metric
        scale recovered from LingBot's own geometry — VGP-Nav (arXiv:2606.09268) sec
        III-E "Ground-Anchored Scale Recovery": the FULL frozen depth head is run on
        the trajectory's num_scale scale-frames (the block that defines the map frame),
        the points are unprojected into the map frame, and the dominant peak of the
        height histogram below the cameras is the floor; scale = camera_height_m /
        estimated_camera_to_floor — computed once per trajectory and cached
        (LingBotStream.get_metric_scale, per-trajectory persistent by rgb_dir key).
        In this mode aux_pose_head is initialized to the PURE axis conversion (R_conv,
        no _SCALE), since t_rel is already metric when it arrives; samples whose floor
        recovery failed (no confident floor points in the scale block) fall back to
        the pooled _SCALE as their per-sample multiplier. `scale_mode='pooled'`
        preserves the old single-constant behavior for comparison.
    """

    # See docstring above for how these were derived and their (known, per-video) limits.
    _R_CONV = ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0))
    _SCALE = 2.564

    def __init__(self, dim=384, n_out=4, aux_pose_calibration='empirical', scale_mode='ground'):
        super().__init__()
        self.revisit_head = nn.Linear(12, n_out * dim)       # [t_rel(3), R_rel.flatten(9)] -> n_out tokens
        self.n_out, self.dim = n_out, dim
        assert aux_pose_calibration in ('empirical', 'trainable'), aux_pose_calibration
        assert scale_mode in ('ground', 'pooled'), scale_mode
        self.aux_pose_calibration = aux_pose_calibration
        self.scale_mode = scale_mode
        # calibrated (x,y) readout, frozen or trainable per aux_pose_calibration — see class docstring
        self.aux_pose_head = nn.Linear(3, 2)
        R_conv = torch.tensor(self._R_CONV)
        with torch.no_grad():
            # 'ground': t_rel arrives already metric (per-trajectory scale applied in
            # forward), so the head is the pure axis conversion; 'pooled': old behavior.
            w = R_conv[:2] if scale_mode == 'ground' else self._SCALE * R_conv[:2]
            self.aux_pose_head.weight.copy_(w)
            self.aux_pose_head.bias.zero_()
        self.aux_pose_head.requires_grad_(aux_pose_calibration == 'trainable')

    @staticmethod
    def _split_pose9(pose9):
        """9-d (absT[3], quaR[4] xyzw cam->world, FoV[2]) -> (t [...,3], unit-quat [...,4]).
        Drops FoV (constant intrinsic); normalizes the quaternion (head emits raw
        non-unit quat; magnitude is decoded away)."""
        return pose9[..., :3], F.normalize(pose9[..., 3:7], dim=-1)

    @staticmethod
    def _relative_pose(cur_pose9, goal_pose9):
        """Analytic T_cur^-1 @ T_goal, split (not recombined into a quaternion — nothing
        downstream needs the compact 4-d form, and mat_to_quat's branch-selection has
        known numerical rough edges near 180-deg rotations that a plain flattened
        rotation matrix avoids).
        quaR is cam->world (p_world = R @ p_cam), so T_cur^-1 expresses goal in cur's own
        local frame: t_rel = R_cur^T(t_goal - t_cur), R_rel = R_cur^T R_goal — the
        bilinear cross term a linear head can't reconstruct from (cur_pose, goal_pose)
        embedded independently. Lazy import: needs lingbot_repo on sys.path, which
        LingBotStream.__init__ guarantees has already run by the time this is called.
        """
        from lingbot_map.utils.rotation import quat_to_mat
        t_cur, q_cur = RevisitMerge._split_pose9(cur_pose9)
        t_goal, q_goal = RevisitMerge._split_pose9(goal_pose9)
        R_cur = quat_to_mat(q_cur)                                    # [B,3,3]
        R_goal = quat_to_mat(q_goal)                                  # [B,3,3]
        R_cur_T = R_cur.transpose(-1, -2)
        t_rel = (R_cur_T @ (t_goal - t_cur).unsqueeze(-1)).squeeze(-1)   # R_cur^T (t_goal - t_cur)
        R_rel = R_cur_T @ R_goal                                         # R_cur^T R_goal
        return t_rel, R_rel

    def forward(self, cur_pose, goal_pose, metric_scale=None):
        """cur_pose, goal_pose: [B, 9] absolute camera poses (map frame).
        metric_scale: [B] per-trajectory lingbot-units -> meters multiplier
        (ground-anchored; required when scale_mode='ground', ignored otherwise)."""
        t_rel, R_rel = self._relative_pose(cur_pose, goal_pose)          # [B,3], [B,3,3]
        if self.scale_mode == 'ground':
            t_rel = t_rel * metric_scale.to(t_rel).unsqueeze(-1)         # metric t_rel
        aux_pose = self.aux_pose_head(t_rel)                             # [B,2]  (x,y) only
        rel_feat = torch.cat([t_rel, R_rel.flatten(-2)], dim=-1)         # [B,12]
        revisit_readout = self.revisit_head(rel_feat).view(-1, self.n_out, self.dim)
        # R_rel returned too — not for any loss (no head/calibration needed for it, it's a
        # raw feature into revisit_head), just so the trainer can log a rotation-accuracy
        # diagnostic against GT (batch_goal_rel_rotation), same treatment as the gate/match
        # diagnostics already logged under no_grad in MemNavTrainer.compute_loss.
        return revisit_readout, aux_pose, R_rel


# --------------------------------------------------------------------------- #
# MemNavNet — full policy: frozen encode loop + (trainable) gate/compress/decoder
# --------------------------------------------------------------------------- #
class MemNavNet(nn.Module):
    def __init__(self, lingbot_kwargs=None, dino_dim=1024, lingbot_dim=2048, depth_feat_dim=256,
                 token_dim=384, heads=8, m_rgbd=4, m_depth=4, m_revisit=4, m_novel=4,
                 predict_size=24, temporal_depth=8, num_diffusion_iters=10, goal_warm=64,
                 aux_pose_calibration='empirical', scale_mode='ground',
                 require_versioned_cache=False, device="cuda"):
        super().__init__()
        self.lingbot = LingBotStream(device=device, **(lingbot_kwargs or {}))
        self.window = self.lingbot.window
        self.num_scale = self.lingbot.num_scale
        self.device = device
        self.heads = heads
        self.predict_size = predict_size
        # goal_append_warm's live-recompute depth before streaming the goal — deeper than
        # `window` on purpose (see LingBotStream.goal_append_warm); validated against a
        # continuous-stream oracle in scripts/diag_lingbot_pose_accuracy.py.
        self.goal_warm = goal_warm
        # Sparse-cache training must fail closed on a legacy/mixed cache tree
        # (silently shifted temporal indices otherwise); dense-cache training
        # keeps the permissive default.
        self.require_versioned_cache = bool(require_versioned_cache)

        # trainable heads
        self.retrieval = RetrievalHead(dino_dim=dino_dim)
        self.novel = NovelBranch(dim=token_dim, heads=heads, out_tokens=m_novel, device=device)

        # current_state = two Perceiver branches (LoGoPlanner-style: perception + geometry)
        #   RGBD branch  : post-GCT window tokens (2C)        -> m_rgbd tokens
        #   depth branch : feature-only depth head (geometry) -> m_depth tokens
        self.proj_current = nn.Linear(lingbot_dim, token_dim)
        self.proj_depth = nn.Linear(depth_feat_dim, token_dim)
        self.compress_rgbd = TokenCompressor(token_dim, heads, m_rgbd)
        self.compress_depth = TokenCompressor(token_dim, heads, m_depth)
        # revisit: analytic relative pose from current + goal absolute camera poses (+ aux pose head)
        self.revisit_merge = RevisitMerge(token_dim, m_revisit, aux_pose_calibration=aux_pose_calibration,
                                          scale_mode=scale_mode)
        self.scale_mode = scale_mode

        # --- NavDP DDPM decoder (no critic) ---
        # memory layout: [ time(1) | current_state(n_cs) | revisit(n_rev) | novel(n_nov) ]
        self.n_cs, self.n_rev, self.n_nov = m_rgbd + m_depth, m_revisit, m_novel
        self.mem_len = 1 + self.n_cs + self.n_rev + self.n_nov
        self.input_embed = nn.Linear(3, token_dim)            # noisy waypoints -> tokens
        self.time_emb = SinusoidalPosEmb(token_dim)
        self.cond_pos_embed = LearnablePositionalEncoding(token_dim, self.mem_len)
        self.out_pos_embed = LearnablePositionalEncoding(token_dim, predict_size)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=heads, dim_feedforward=4 * token_dim,
            activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=temporal_depth)
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, 3)
        # (no critic — collision is checked geometrically from LingBot's point map at eval)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters, beta_schedule="squaredcos_cap_v2",
            clip_sample=True, prediction_type="epsilon")
        tgt = (torch.triu(torch.ones(predict_size, predict_size)) == 1).transpose(0, 1)
        self.register_buffer("tgt_mask",
                             tgt.float().masked_fill(tgt == 0, float("-inf")).masked_fill(tgt == 1, 0.0))

        # global prior on revisit vs novel, ADDED to the per-sample gate bias in the decoder
        # cross-attention. [0]=revisit, [1]=novel; only the difference matters (softmax).
        # Learnable by default (the model tunes the global balance); to force/ablate a weighting
        # set `net.branch_bias.data = torch.tensor([r, n])` and `net.branch_bias.requires_grad_(False)`.
        self.branch_bias = nn.Parameter(torch.zeros(2))

        self.to(device)   # move trainable heads to device (lingbot.model already there)

    def build_current_state(self, current, depth_feat):
        """current [B,P,2C] (post-GCT), depth_feat [B,Pf,Cd] -> current_state [B, m_rgbd+m_depth, token_dim]."""
        rgbd = self.compress_rgbd(self.proj_current(current))    # [B, m_rgbd, token_dim]
        geom = self.compress_depth(self.proj_depth(depth_feat))  # [B, m_depth, token_dim]
        return torch.cat([rgbd, geom], dim=1)

    def build_revisit(self, cur_pose, goal_pose, metric_scale=None):
        """cur_pose/goal_pose [B, 9] absolute camera poses (current frame + goal_append_warm),
        metric_scale [B] per-trajectory ground-anchored scale (scale_mode='ground')
        -> (revisit_readout [B,m_revisit,token_dim], aux_pose [B,2] (x,y) only, R_rel [B,3,3])."""
        return self.revisit_merge(cur_pose, goal_pose, metric_scale)

    # ----- DDPM decoder ------------------------------------------------ #
    def _memory(self, current_state, revisit, novel, timestep):
        """[B, mem_len, D] = [time | current_state | revisit | novel] + pos embed."""
        B = current_state.shape[0]
        time_emb = self.time_emb(timestep.to(self.device)).unsqueeze(1).expand(B, 1, -1)
        mem = torch.cat([time_emb, current_state, revisit, novel], dim=1)
        return mem + self.cond_pos_embed(mem)

    def _gate_mask(self, gate):
        """Per-sample cross-attention bias [B*heads, predict_size, mem_len] — directs
        attention without scaling the readouts.
          revisit cols += log(gate), novel cols += log(1-gate)"""
        B = gate.shape[0]
        bias = gate.new_zeros(B, self.mem_len)
        rs, re = 1 + self.n_cs, 1 + self.n_cs + self.n_rev
        ns, ne = re, re + self.n_nov
        g = gate.clamp(1e-4, 1 - 1e-4)
        bias[:, rs:re] = torch.log(g).unsqueeze(1) + self.branch_bias[0]      # revisit
        bias[:, ns:ne] = torch.log(1 - g).unsqueeze(1) + self.branch_bias[1]  # novel
        bias = bias[:, None, None, :].expand(B, self.heads, self.predict_size, self.mem_len)
        return bias.reshape(B * self.heads, self.predict_size, self.mem_len)

    def predict_noise(self, noisy, timestep, current_state, revisit, novel, gate):
        a = self.input_embed(noisy)
        a = a + self.out_pos_embed(a)
        mem = self._memory(current_state, revisit, novel, timestep)
        out = self.decoder(tgt=a, memory=mem, tgt_mask=self.tgt_mask,
                           memory_mask=self._gate_mask(gate))
        return self.action_head(self.layernorm(out))

    def forward(self, batch):
        dev = self.device
        enc = self.encode_memory(batch)
        current_state = self.build_current_state(enc["current"], enc["depth_feat"])
        revisit, aux_pose, R_rel = self.build_revisit(enc["cur_pose"], enc["goal_pose"],
                                                      enc["metric_scale"])
        novel = self.novel(batch["batch_window_images"][:, -1].to(dev),   # current frame [B,3,H,W]
                           batch["batch_goal_image"].to(dev))             # goal frame
        gate = enc["revisit_gate"]

        labels = batch["batch_labels"].to(dev)          # [B, predict_size, 3]
        B = labels.shape[0]
        noise = torch.randn_like(labels)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=dev)
        noisy = self.noise_scheduler.add_noise(labels, noise, timesteps)

        noise_pred = self.predict_noise(noisy, timesteps, current_state, revisit, novel, gate)
        return dict(
            noise_pred=noise_pred, noise=noise,
            aux_pose=aux_pose, R_rel=R_rel, ret_logits=enc["ret_logits"], revisit_gate=gate,
            gate_logit=enc["gate_logit"], match_idx=enc["match_idx"], anchor_idx=enc["anchor_idx"],
            goal_anchor_idx=enc["goal_anchor_idx"],
        )

    @torch.no_grad()
    def _load_cache(self, path, rgb_dir):
        """Assemble the KV cache dict from disk. If the npz lacks
        ``scale_k/scale_v`` (--skip_scale precompute mode), compute it on the
        fly from the first ``num_scale`` RGB frames of ``rgb_dir`` — bf16 output,
        LRU-cached per trajectory inside LingBotStream."""
        with np.load(path) as c_file, np.load(
            path.replace("lingbot_cache.npz", "lingbot_cam_cache.npz")
        ) as cc_file:
            # Materialize once: validation and GPU conversion otherwise cause
            # np.load to reread the large ZIP_STORED KV arrays independently.
            c = {name: c_file[name] for name in c_file.files}
            cc = {name: cc_file[name] for name in cc_file.files}
        # Two-timeline layout check: which raw frames the sparse KV rows cover
        # (legacy dense caches validate as interval-1 with full coverage).
        layout = validate_cache_pair(
            c, cc,
            expected_num_scale_frames=self.num_scale,
            require_versioned=self.require_versioned_cache,
        )
        keys = set(c)
        if "scale_k" in keys and "scale_v" in keys:
            sk, sv, ak, av = LingBotStream._cache_to_layered(
                c["scale_k"], c["scale_v"], c["anchor_k"], c["anchor_v"], self.device)
        else:
            sk, sv = self.lingbot.get_scale_kv(rgb_dir)
            ak = torch.as_tensor(c["anchor_k"], device=self.device, dtype=torch.bfloat16)\
                .permute(1, 2, 0, 3, 4).contiguous()
            av = torch.as_tensor(c["anchor_v"], device=self.device, dtype=torch.bfloat16)\
                .permute(1, 2, 0, 3, 4).contiguous()
        ck, cv = LingBotStream._cam_to_device(cc["cam_k"], cc["cam_v"], self.device)
        # cam_pose_enc [S,9]: the frozen camera head's own pose for every REAL trajectory
        # frame, captured during precompute's genuinely continuous stream (extract_trajectory)
        # — used directly for cur_pose (see encode_memory) instead of re-deriving it from a
        # window_forward recompute, which cold-starts at k-W+1 with no real predecessors and
        # is measurably worse (diag_lingbot_pose_accuracy.py: ATE 3.35m vs 0.04m on a 2-leg
        # smoke episode). goal_pose still needs a live camera_pose() call — the goal image is
        # newly inserted, not a frame this array has an entry for.
        cam_pose_enc = torch.as_tensor(cc["cam_pose_enc"], device=self.device, dtype=torch.float32)
        # whole-episode ground-anchored floor estimate stored by precompute
        # (--skip_ground_scale omits it; NaN = stream saw no confident floor).
        # None -> encode_memory falls back to the on-the-fly 64-frame estimate.
        ghe = float(cc["ground_h_est"]) if "ground_h_est" in cc else float("nan")
        out = dict(scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av, cam_k=ck, cam_v=cv,
                   cam_pose_enc=cam_pose_enc,
                   ground_h_est=ghe if np.isfinite(ghe) else None)
        if not layout.legacy_dense:
            # Sparse keyframe caches: raw-frame indices of the surviving KV rows.
            # LingBotStream's injection uses these to translate "history up to raw
            # frame k" into row counts + compressed aggregator time; the camera
            # head keeps raw times. Legacy dense caches omit them and take the
            # original row==frame arithmetic unchanged.
            out["anchor_frame_indices"] = torch.as_tensor(
                layout.anchor_frame_indices, dtype=torch.long)
            out["cam_frame_indices"] = torch.as_tensor(
                layout.cam_frame_indices, dtype=torch.long)
        return out

    def encode_memory(self, batch):
        """Frozen front-end orchestration. Retrieval (trainable, batched) picks the
        match index; a per-sample loop runs the frozen LingBot ops. Returns the
        readouts the trainable head consumes.
        """
        dev = self.device
        # goal_cls: real goal images (goal_{j}.jpg) have no cached CLS, so compute it
        # from the goal image via the frozen context-free DINO trunk (same space as the
        # cached per-frame dino_cls). Fall back to a provided batch_goal_cls (old path /
        # smoke tests where the goal is a trajectory frame).
        if batch.get("batch_goal_cls") is not None:
            goal_cls = batch["batch_goal_cls"].to(dev)
        else:
            goal_cls = self.lingbot.dino(batch["batch_goal_image"].to(dev))["cls"]  # [B, D']
        mem_cls = batch["batch_mem_cls"].to(dev)
        cand_mask = batch["batch_cand_mask"].to(dev)   # revisit candidates E(k) = [amargin..k-t]
        # (trainable) retrieval — match index + gate logit + ranking logits (over candidates)
        match_idx, gate_logit, ret_logits = self.retrieval(goal_cls, mem_cls, cand_mask)
        revisit_gate = torch.sigmoid(gate_logit)       # P(revisit) for the decoder soft-gate

        # goal_append anchor: at TRAIN time teacher-force it to a GT co-visible frame so the
        # goal_pose (-> aux + revisit token) is well-anchored from step 1, decoupling those
        # heads from retrieval convergence. At EVAL (no pos_mask / self.eval()) fall back to
        # the live match_idx — the same anchor a converged retrieval produces. Novel rows have
        # no positive -> keep match_idx (aux weight is 0 for them anyway).
        pos_mask = batch.get("batch_pos_mask")
        if self.training and pos_mask is not None:
            pos_mask = pos_mask.to(dev).bool()
            NEG_INF = torch.finfo(ret_logits.dtype).min
            tf_idx = ret_logits.masked_fill(~pos_mask, NEG_INF).argmax(-1)   # best-scoring positive
            anchor = torch.where(pos_mask.any(-1), tf_idx, match_idx)
        else:
            anchor = match_idx

        B = len(batch["cache_paths"])
        lo = self.num_scale + self.window - 1
        # MEMNAV_STREAM_GROUP: how many trajectories' streams to run on the batch dim
        # at once. 1 (default) = the original one-at-a-time loop (byte-identical path).
        # >1 collapses the per-sample window_forward into the aggregator's batch axis
        # for a ~G-fold throughput gain, at the cost of ~G resident KV caches — the
        # loop still processes the batch in chunks of G so peak memory is bounded to G.
        G = max(1, int(os.environ.get("MEMNAV_STREAM_GROUP", "1")))
        ks = [int(batch["cur_steps"][b]) for b in range(B)]
        cur_t = [None] * B; dfeat_t = [None] * B
        curp = [None] * B; goalp = [None] * B; goal_m = [None] * B
        mscale = [RevisitMerge._SCALE] * B   # per-sample scale; pooled constant = fallback
        for s in range(0, B, G):
            idx = list(range(s, min(s + G, B)))
            with torch.no_grad():
                caches = [self._load_cache(batch["cache_paths"][b], batch["rgb_dirs"][b]) for b in idx]
                wig = batch["batch_window_images"][idx].to(dev)                     # [g, W, 3, H, W]
                # (1) current state: post-GCT tokens + depth-head geometry.
                if len(idx) == 1:
                    # singleton (G==1, or the odd tail group): original scalar path.
                    b = idx[0]
                    wt, cur_agg, psi = self.lingbot.window_forward(caches[0], wig[0], ks[b], return_multilayer=True)
                    cur_t[b] = wt[-1]                                               # [P, 2C]
                    dfeat_t[b] = self.lingbot.depth_feature(cur_agg, wig[0][-1:][None], psi)  # [Pf, Cd]
                else:
                    # batched: G streams stacked on the batch dim (one pass streams all).
                    kg = [ks[b] for b in idx]
                    curb, cur_aggb, psi = self.lingbot.window_forward_batched(caches, wig, kg, return_multilayer=True)
                    dfeatb = self.lingbot.depth_feature_batched(cur_aggb, wig[:, -1:], psi)   # [g, Pf, Cd]
                    for jj, b in enumerate(idx):
                        cur_t[b] = curb[jj]; dfeat_t[b] = dfeatb[jj]
                # (2) cur_pose (direct read) + anchor m per sample.
                warm, S = self.goal_warm, self.num_scale
                m_of = {}
                for jj, b in enumerate(idx):
                    k = ks[b]
                    # cur_pose: read the precomputed continuous-stream pose directly (exact,
                    # no cold-start reconstruction) — k is always a real trajectory frame.
                    curp[b] = caches[jj]["cam_pose_enc"][k]                         # [9] current abs pose
                    if self.scale_mode == 'ground':
                        # ground-anchored per-trajectory metric scale. Preferred source:
                        # the whole-episode ground_h_est stored in the cam cache by
                        # precompute (zero runtime cost). Fallback for old caches: the
                        # on-the-fly 64-frame estimate (cached by rgb_dir, ~2s once per
                        # trajectory). Either way None -> pooled-constant fallback.
                        ch = batch.get("batch_camera_height")
                        cam_h = float(ch[b]) if ch is not None else 0.5
                        if caches[jj]["ground_h_est"] is not None:
                            s = ground_scale_from_h_est(caches[jj]["ground_h_est"], cam_h)
                        else:
                            s = self.lingbot.get_metric_scale(
                                batch["rgb_dirs"][b], caches[jj]["cam_pose_enc"], cam_h)
                        if s is not None:
                            mscale[b] = s
                    m = int(anchor[b].clamp(lo, k - 1).item())
                    m_of[b] = m
                    goal_m[b] = m   # post-clamp anchor actually used for goal_pose (may differ from anchor[b])
                # (3) revisit goal pose: goal_append_warm at the anchor (deep warm-recompute,
                # self.goal_warm, not the nominal W — its cold start starves the goal's pose;
                # goal_warm=64 matches a continuous-stream oracle). The warm-frame count
                # L = m - max(num_scale, m-warm+1) + 1 varies per sample AND the replay evicts,
                # so batch only LOCKSTEP groups (equal L); singletons take the scalar path.
                # camera_pose stays per-sample (cheap: one camera-head forward).
                groups = {}
                for jj, b in enumerate(idx):
                    L = m_of[b] - max(S, m_of[b] - warm + 1) + 1
                    groups.setdefault(L, []).append((jj, b))
                for members in groups.values():
                    if len(members) == 1:
                        jj, b = members[0]
                        cache = caches[jj]
                        goal_img = batch["batch_goal_image"][b].to(dev)
                        _, goal_agg = self.lingbot.goal_append_warm(goal_img, cache, m_of[b],
                                                                    batch["rgb_dirs"][b], warm, return_agg=True)
                        goalp[b] = self.lingbot.camera_pose(cache["cam_k"], cache["cam_v"],
                                                            m_of[b] + 1, goal_agg,
                                                            cam_frame_indices=cache.get("cam_frame_indices"))[-1]
                    else:
                        gc = [caches[jj] for jj, b in members]
                        gms = [m_of[b] for jj, b in members]
                        grgb = [batch["rgb_dirs"][b] for jj, b in members]
                        ggoal = torch.stack([batch["batch_goal_image"][b] for jj, b in members], 0).to(dev)
                        _, aggs = self.lingbot.goal_append_warm_batched(ggoal, gc, gms, grgb, warm, return_agg=True)
                        for (jj, b), agg in zip(members, aggs):
                            goalp[b] = self.lingbot.camera_pose(caches[jj]["cam_k"], caches[jj]["cam_v"],
                                                                m_of[b] + 1, agg,
                                                                cam_frame_indices=caches[jj].get("cam_frame_indices"))[-1]  # [9] goal abs pose

        return dict(
            current=torch.stack(cur_t),      # [B, P, 2C]    post-GCT (RGBD branch)
            depth_feat=torch.stack(dfeat_t), # [B, Pf, Cd]   depth-head geometry
            cur_pose=torch.stack(curp),      # [B, 9]        current absolute camera pose (map frame)
            goal_pose=torch.stack(goalp),    # [B, 9]        goal absolute camera pose (map frame)
            metric_scale=torch.tensor(mscale, device=dev, dtype=torch.float32),  # [B] lingbot->meters

            match_idx=match_idx, anchor_idx=anchor, revisit_gate=revisit_gate,
            gate_logit=gate_logit, ret_logits=ret_logits,
            goal_anchor_idx=torch.tensor(goal_m, device=dev, dtype=torch.long),  # [B] post-clamp m used for goal_pose
        )


# --------------------------------------------------------------------------- #
# HF wrapper (for scripts/train/train.py registry: from_pretrained + config)
# --------------------------------------------------------------------------- #
class MemNavModelConfig(PretrainedConfig):
    model_type = 'memnav'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_cfg = kwargs.get('model_cfg', None)


class MemNavPolicy(PreTrainedModel):
    config_class = MemNavModelConfig

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        if hasattr(config, 'model_dump'):                  # pydantic ExpCfg -> wrap
            config = cls.config_class(model_cfg=config)
        model = cls(config)
        path = pretrained_model_name_or_path
        if path and len(str(path)) > 0 and os.path.exists(path):
            sd = torch.load(path, map_location='cpu')
            sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
            inc = model.load_state_dict(sd, strict=False)
            print(f"[memnav] loaded {path}: missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")
        return model

    def __init__(self, config: MemNavModelConfig):
        super().__init__(config)
        il = config.model_cfg['il']
        # runtime LOCAL_RANK (set by torchrun) wins over the static config rank, so each
        # DDP rank builds the frozen LingBot + heads on its own GPU.
        local_rank = int(os.getenv('LOCAL_RANK', config.model_cfg.get('local_rank', 0)))
        self._device = torch.device(f"cuda:{local_rank}")
        # frozen-LingBot paths come from the config so HPC can override without code edits
        lingbot_kwargs = {}
        if il.get('lingbot_repo'):    lingbot_kwargs['lingbot_repo'] = il['lingbot_repo']
        if il.get('lingbot_weights'): lingbot_kwargs['weights'] = il['lingbot_weights']
        # memory-partition geometry — MUST match the precompute + dataset (mp3d: 32/8/2048).
        # LingBotStream sets kv_cache_sliding_window=window, so window here == the precompute
        # --kv_cache_sliding_window; max_frame_num sizes the 3D-RoPE table (long 3leg episodes).
        if il.get('window_size') is not None:   lingbot_kwargs['window'] = il['window_size']
        if il.get('num_scale') is not None:     lingbot_kwargs['num_scale'] = il['num_scale']
        if il.get('max_frame_num') is not None: lingbot_kwargs['max_frame_num'] = il['max_frame_num']
        self.core = MemNavNet(
            token_dim=il['token_dim'], heads=il['heads'], predict_size=il['predict_size'],
            temporal_depth=il['temporal_depth'], num_diffusion_iters=il.get('num_diffusion_iters', 10),
            goal_warm=il.get('goal_warm', 64),
            aux_pose_calibration=il.get('aux_pose_calibration', 'empirical'),
            scale_mode=il.get('scale_mode', 'ground'),
            require_versioned_cache=bool(il.get('require_versioned_cache', False)),
            lingbot_kwargs=lingbot_kwargs or None, device=str(self._device),
        )

    def forward(self, batch):
        return self.core(batch)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run encode_memory on a real batch (needs GPU + cache)")
    args = ap.parse_args()

    B, L, D, P = 4, 60, 1024, 1369
    # retrieval smoke: ranking logits over candidates + affine gate on max-cos
    rh = RetrievalHead()
    goal_cls = torch.randn(B, D)
    mem_cls = torch.randn(B, L, D)
    cand_mask = torch.ones(B, L, dtype=torch.bool)
    cand_mask[0, 40:] = False  # sample 0: fewer candidates
    cand_mask[1, :] = False    # sample 1: no candidate -> novel (gate floor)
    m, gate_logit, logits = rh(goal_cls, mem_cls, cand_mask)
    gate = torch.sigmoid(gate_logit)
    print(f"RetrievalHead: match_idx={m.tolist()} gate={[round(x,3) for x in gate.tolist()]} logits={tuple(logits.shape)}")
    # decoupled losses: InfoNCE (ranking) + BCE (gate)
    pos = torch.zeros(B, L, dtype=torch.bool); pos[[0, 2, 3], [12, 5, 33]] = True
    neg = cand_mask & ~pos
    rank = (logits.masked_fill(~(pos | neg), float("-inf")).logsumexp(-1)
            - logits.masked_fill(~pos, float("-inf")).logsumexp(-1))[[0, 2, 3]].mean()
    is_rev = torch.tensor([1.0, 0.0, 1.0, 1.0])
    gate_ce = F.binary_cross_entropy_with_logits(gate_logit, is_rev)
    print(f"  rank InfoNCE={rank.item():.3f}  gate BCE={gate_ce.item():.3f}  "
          f"grad ok={torch.autograd.grad(rank + gate_ce, rh.log_temp, retain_graph=True)[0] is not None}")

    # novel branch smoke (early fusion on raw images)
    nb = NovelBranch(device="cuda").to("cuda")
    cur_img = torch.rand(B, 3, 518, 518, device="cuda")
    goal_img = torch.rand(B, 3, 518, 518, device="cuda")
    out = nb(cur_img, goal_img)
    print(f"NovelBranch: out={tuple(out.shape)} params={sum(p.numel() for p in nb.parameters())/1e6:.2f}M")

    if args.full:
        import sys
        sys.path.insert(0, "/home/asus/Research/Nav/InternNav")
        from internnav.dataset.memnav_dataset_lerobot import MemNav_Dataset, memnav_collate_fn
        ds = MemNav_Dataset("/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data", predict_size=24)
        batch = memnav_collate_fn([ds[i] for i in range(2)])
        net = MemNavNet(device="cuda")
        out = net.encode_memory(batch)
        print("\nencode_memory readouts:")
        for key, v in out.items():
            if torch.is_tensor(v):
                print(f"  {key}: {tuple(v.shape)} {v.dtype} req_grad={v.requires_grad}")
        print(f"  cur_steps={batch['cur_steps']} goal_steps={batch['goal_steps']} match_idx={out['match_idx'].tolist()}")
        cs = net.build_current_state(out["current"], out["depth_feat"])
        nov = net.novel(batch["batch_window_images"][:, -1].to(net.device), batch["batch_goal_image"].to(net.device))
        rr, ap, _R_rel = net.build_revisit(out["cur_pose"], out["goal_pose"], out["metric_scale"])
        print(f"  current_state (RGBD+depth Perceiver): {tuple(cs.shape)} req_grad={cs.requires_grad}")
        print(f"  novel readout: {tuple(nov.shape)} req_grad={nov.requires_grad}")
        print(f"  revisit_readout: {tuple(rr.shape)} | aux_pose: {tuple(ap.shape)} req_grad={rr.requires_grad}")

        fwd = net(batch)
        print("\nforward outputs:")
        for key, v in fwd.items():
            print(f"  {key}: {tuple(v.shape)} {v.dtype}")
        loss = ((fwd["noise_pred"] - fwd["noise"]).square().mean()
                + fwd["aux_pose"].square().mean())
        loss.backward()
        n_grad = sum(1 for p in net.parameters() if p.requires_grad and p.grad is not None)
        n_train = sum(1 for p in net.parameters() if p.requires_grad)
        print(f"  dummy loss={loss.item():.3f}; params w/ grad={n_grad}/{n_train} trainable")
