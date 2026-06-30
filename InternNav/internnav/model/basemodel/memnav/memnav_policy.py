"""MemNav policy — trainable head over the frozen LingBotStream front-end.

Three goal pathways (see GL.md / memnav-project memory):
  (1) backbone current state      — frozen GCT (LingBotStream.window_forward)
  (2) revisit goal→history        — frozen GCT (LingBotStream.goal_append), visited goals
  (3) novel current→goal (DINO)   — TRAINABLE cross-attention, unseen goals
Retrieval confidence biases the decoder cross-attention toward (2) vs (3) (no multiply,
no goal_cls). NavDP DDPM decoder on top; NO critic (collision is geometric at eval).
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import PretrainedConfig, PreTrainedModel

from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
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
    """goal_cls vs mem_cls (history CLS) → (match_idx, revisit_gate, logits).

    A learnable projection + temperature in a matching space, plus a **learnable
    null** candidate so an *unseen* goal (no real match) lands on null → low gate.
      - match_idx    : argmax real frame (discrete; drives LingBotStream.goal_append)
      - revisit_gate : 1 − P(null)  (differentiable; blends (2) vs (3))
      - logits       : [B, L+1] (last = null) for the retrieval-CE loss
                       (target = k_goal for seen, = L (null) for unseen)
    """

    def __init__(self, dino_dim=1024, proj_dim=256, temp_init=0.07):
        super().__init__()
        self.proj_goal = nn.Linear(dino_dim, proj_dim)
        self.proj_mem = nn.Linear(dino_dim, proj_dim)
        self.null_key = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(temp_init))))

    def forward(self, goal_cls, mem_cls, mem_mask):
        """goal_cls [B,D'], mem_cls [B,L,D'], mem_mask [B,L] bool."""
        gq = F.normalize(self.proj_goal(goal_cls), dim=-1)        # [B,d]
        mk = F.normalize(self.proj_mem(mem_cls), dim=-1)          # [B,L,d]
        nk = F.normalize(self.null_key, dim=-1)                   # [d]
        temp = self.log_temp.exp().clamp(0.01, 1.0)

        scores = (gq.unsqueeze(1) * mk).sum(-1) / temp           # [B,L]
        scores = scores.masked_fill(~mem_mask, float("-inf"))    # ignore padding
        null = (gq * nk).sum(-1, keepdim=True) / temp            # [B,1]
        logits = torch.cat([scores, null], dim=1)               # [B,L+1] (last = null)

        prob = logits.softmax(-1)
        revisit_gate = 1.0 - prob[:, -1]                         # P(some real match)
        match_idx = scores.argmax(-1)                           # best real frame
        return match_idx, revisit_gate, logits


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
# (2.merge) Revisit: merge current pose token + goal pose token — trainable
# --------------------------------------------------------------------------- #
class RevisitMerge(nn.Module):
    """LoGoPlanner `state_decoder` analog: fuse the **current** and **goal** absolute
    camera poses (from the frozen camera head, map frame) and *learn the relative pose*.
    A LingBot-style pose encoder (`embed_pose` = Linear(7→dim) on [T, unit-quat]) embeds each, a shared
    TokenCompressor fuses them, then two own heads:
      - revisit_head  → revisit_readout (the diffusion goal slot)
      - aux_pose_head → (x, y, θ)        (GT-supervised relative pose, like pg_pred_mlp)
    """

    def __init__(self, dim=384, heads=8, n_out=4):
        super().__init__()
        # --- shared: pose encoder (LingBot embed_pose design) + fusion ---
        self.pose_encoder = nn.Linear(7, dim)               # [T(3), unit-quat(4)] -> dim
        self.merge = TokenCompressor(dim, heads, n_out)
        # --- own heads ---
        self.revisit_head = nn.Linear(dim, dim)
        self.aux_pose_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(), nn.Linear(dim // 2, 3))

    @staticmethod
    def _pose7(pose9):
        """9-d (absT[3], quaR[4], FoV[2]) -> 7-d [T, unit-quat]: drop FoV (constant intrinsic),
        normalize the quaternion (head emits raw non-unit quat; magnitude is decoded away)."""
        return torch.cat([pose9[..., :3], F.normalize(pose9[..., 3:7], dim=-1)], dim=-1)

    def forward(self, cur_pose, goal_pose):
        """cur_pose, goal_pose: [B, 9] absolute camera poses (map frame)."""
        toks = torch.stack([self.pose_encoder(self._pose7(cur_pose)),
                            self.pose_encoder(self._pose7(goal_pose))], dim=1)  # [B,2,dim]
        shared = self.merge(toks)                                        # [B, n_out, dim]  (shared)
        return self.revisit_head(shared), self.aux_pose_head(shared.mean(1))   # [B,n_out,dim], [B,3]


# --------------------------------------------------------------------------- #
# MemNavNet — full policy: frozen encode loop + (trainable) gate/compress/decoder
# --------------------------------------------------------------------------- #
class MemNavNet(nn.Module):
    def __init__(self, lingbot_kwargs=None, dino_dim=1024, lingbot_dim=2048, depth_feat_dim=256,
                 token_dim=384, heads=8, m_rgbd=4, m_depth=4, m_revisit=4, m_novel=4,
                 predict_size=24, temporal_depth=8, num_diffusion_iters=10, device="cuda"):
        super().__init__()
        self.lingbot = LingBotStream(device=device, **(lingbot_kwargs or {}))
        self.window = self.lingbot.window
        self.num_scale = self.lingbot.num_scale
        self.device = device
        self.heads = heads
        self.predict_size = predict_size

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
        # revisit: encode current + goal absolute camera poses, learn the relative (+ aux pose head)
        self.revisit_merge = RevisitMerge(token_dim, heads, m_revisit)

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

    def build_revisit(self, cur_pose, goal_pose):
        """cur_pose/goal_pose [B, 2C] camera-head pose features (current frame + goal_append)
        -> (revisit_readout [B,m_revisit,token_dim], aux_pose [B,3])."""
        return self.revisit_merge(cur_pose, goal_pose)

    # ----- DDPM decoder ------------------------------------------------ #
    def _memory(self, current_state, revisit, novel, timestep):
        """[B, mem_len, D] = [time | current_state | revisit | novel] + pos embed."""
        B = current_state.shape[0]
        time_emb = self.time_emb(timestep.to(self.device)).unsqueeze(1).expand(B, 1, -1)
        mem = torch.cat([time_emb, current_state, revisit, novel], dim=1)
        return mem + self.cond_pos_embed(mem)

    def _gate_mask(self, gate, mode):
        """Per-sample cross-attention bias [B*heads, predict_size, mem_len] — directs
        attention without scaling the readouts.
          mg: revisit cols += log(gate), novel cols += log(1-gate)
          ng: revisit+novel cols = -inf  (classifier-free no-goal)"""
        B = gate.shape[0]
        bias = gate.new_zeros(B, self.mem_len)
        rs, re = 1 + self.n_cs, 1 + self.n_cs + self.n_rev
        ns, ne = re, re + self.n_nov
        if mode == "mg":
            g = gate.clamp(1e-4, 1 - 1e-4)
            bias[:, rs:re] = torch.log(g).unsqueeze(1) + self.branch_bias[0]      # revisit
            bias[:, ns:ne] = torch.log(1 - g).unsqueeze(1) + self.branch_bias[1]  # novel
        else:                                          # ng
            bias[:, rs:ne] = float("-inf")
        bias = bias[:, None, None, :].expand(B, self.heads, self.predict_size, self.mem_len)
        return bias.reshape(B * self.heads, self.predict_size, self.mem_len)

    def predict_noise(self, noisy, timestep, current_state, revisit, novel, gate, mode):
        a = self.input_embed(noisy)
        a = a + self.out_pos_embed(a)
        mem = self._memory(current_state, revisit, novel, timestep)
        out = self.decoder(tgt=a, memory=mem, tgt_mask=self.tgt_mask,
                           memory_mask=self._gate_mask(gate, mode))
        return self.action_head(self.layernorm(out))

    def forward(self, batch):
        dev = self.device
        enc = self.encode_memory(batch)
        current_state = self.build_current_state(enc["current"], enc["depth_feat"])
        revisit, aux_pose = self.build_revisit(enc["cur_pose"], enc["goal_pose"])
        novel = self.novel(batch["batch_window_images"][:, -1].to(dev),   # current frame [B,3,H,W]
                           batch["batch_goal_image"].to(dev))             # goal frame
        gate = enc["revisit_gate"]

        labels = batch["batch_labels"].to(dev)          # [B, predict_size, 3]
        B = labels.shape[0]
        noise = torch.randn_like(labels)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=dev)
        noisy = self.noise_scheduler.add_noise(labels, noise, timesteps)

        noise_mg = self.predict_noise(noisy, timesteps, current_state, revisit, novel, gate, "mg")
        noise_ng = self.predict_noise(noisy, timesteps, current_state, revisit, novel, gate, "ng")
        return dict(
            noise_ng=noise_ng, noise_mg=noise_mg, noise=noise,
            aux_pose=aux_pose, ret_logits=enc["ret_logits"], revisit_gate=gate,
        )

    @torch.no_grad()
    def _load_cache(self, path):
        c = np.load(path)
        sk, sv, ak, av = LingBotStream._cache_to_layered(
            c["scale_k"], c["scale_v"], c["anchor_k"], c["anchor_v"], self.device)
        cc = np.load(path.replace("lingbot_cache.npz", "lingbot_cam_cache.npz"))
        ck, cv = LingBotStream._cam_to_device(cc["cam_k"], cc["cam_v"], self.device)
        return dict(scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av, cam_k=ck, cam_v=cv)

    def encode_memory(self, batch):
        """Frozen front-end orchestration. Retrieval (trainable, batched) picks the
        match index; a per-sample loop runs the frozen LingBot ops. Returns the
        readouts the trainable head consumes.
        """
        dev = self.device
        goal_cls = batch["batch_goal_cls"].to(dev)
        mem_cls = batch["batch_mem_cls"].to(dev)
        mem_mask = batch["batch_mem_mask"].to(dev)
        # (trainable) retrieval — match index + gate + logits
        match_idx, revisit_gate, ret_logits = self.retrieval(goal_cls, mem_cls, mem_mask)

        B = len(batch["cache_paths"])
        W, lo = self.window, self.num_scale + self.window - 1
        cur_t, dfeat_t, curp, goalp = [], [], [], []
        for b in range(B):
            k = int(batch["cur_steps"][b])
            rgb_dir = batch["rgb_dirs"][b]
            goal_img = batch["batch_goal_image"][b].to(dev)
            win_img = batch["batch_window_images"][b].to(dev)
            with torch.no_grad():
                cache = self._load_cache(batch["cache_paths"][b])
                ck, cv = cache["cam_k"], cache["cam_v"]
                # (1) current state: post-GCT tokens + depth-head geometry + pose feature
                #  wt: window tokens [W, P, 2C], cur_agg: current frame's multi-layer agg, psi: patch_start_idx
                wt, cur_agg, psi = self.lingbot.window_forward(cache, win_img, k, return_multilayer=True)
                cur = wt[-1]                                                        # [P, 2C]
                dfeat = self.lingbot.depth_feature(cur_agg, win_img[-1:][None], psi)  # [Pf, Cd]
                cur_pose = self.lingbot.camera_pose(ck, cv, k, cur_agg)[-1]         # [9] current abs pose
                # (2) revisit: goal_append at the matched frame (clamped valid) -> goal abs pose
                m = int(match_idx[b].clamp(lo, k - 1).item())
                mw = self.lingbot.load_images([os.path.join(rgb_dir, f"{i}.jpg")
                                               for i in range(m - W + 1, m + 1)]).to(dev)
                _, goal_agg = self.lingbot.goal_append(goal_img, cache, m, mw, return_agg=True)
                goal_pose = self.lingbot.camera_pose(ck, cv, m + 1, goal_agg)[-1]   # [9] goal abs pose
                # (3) novel branch runs on raw images (batched, in forward) — no live dino needed
            cur_t.append(cur); dfeat_t.append(dfeat); curp.append(cur_pose); goalp.append(goal_pose)

        return dict(
            current=torch.stack(cur_t),      # [B, P, 2C]    post-GCT (RGBD branch)
            depth_feat=torch.stack(dfeat_t), # [B, Pf, Cd]   depth-head geometry
            cur_pose=torch.stack(curp),      # [B, 9]        current absolute camera pose (map frame)
            goal_pose=torch.stack(goalp),    # [B, 9]        goal absolute camera pose (map frame)
            match_idx=match_idx, revisit_gate=revisit_gate, ret_logits=ret_logits,
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
        self.core = MemNavNet(
            token_dim=il['token_dim'], heads=il['heads'], predict_size=il['predict_size'],
            temporal_depth=il['temporal_depth'], num_diffusion_iters=il.get('num_diffusion_iters', 10),
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
    # retrieval smoke
    rh = RetrievalHead()
    goal_cls = torch.randn(B, D)
    mem_cls = torch.randn(B, L, D)
    mem_mask = torch.ones(B, L, dtype=torch.bool)
    mem_mask[0, 40:] = False  # pad sample 0
    m, gate, logits = rh(goal_cls, mem_cls, mem_mask)
    print(f"RetrievalHead: match_idx={m.tolist()} gate={[round(x,3) for x in gate.tolist()]} logits={tuple(logits.shape)}")
    # retrieval CE (seen target=k_goal, unseen target=L=null)
    target = torch.tensor([12, 60, 5, 33])   # sample 1 = unseen -> null index L=60
    ce = F.cross_entropy(logits, target)
    print(f"  retrieval CE (sanity) = {ce.item():.3f}; grad ok = {torch.autograd.grad(ce, rh.log_temp, retain_graph=True)[0] is not None}")

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
        rr, ap = net.build_revisit(out["cur_pose"], out["goal_pose"])
        print(f"  current_state (RGBD+depth Perceiver): {tuple(cs.shape)} req_grad={cs.requires_grad}")
        print(f"  novel readout: {tuple(nov.shape)} req_grad={nov.requires_grad}")
        print(f"  revisit_readout: {tuple(rr.shape)} | aux_pose: {tuple(ap.shape)} req_grad={rr.requires_grad}")

        fwd = net(batch)
        print("\nforward outputs:")
        for key, v in fwd.items():
            print(f"  {key}: {tuple(v.shape)} {v.dtype}")
        loss = ((fwd["noise_mg"] - fwd["noise"]).square().mean()
                + (fwd["noise_ng"] - fwd["noise"]).square().mean()
                + fwd["aux_pose"].square().mean())
        loss.backward()
        n_grad = sum(1 for p in net.parameters() if p.requires_grad and p.grad is not None)
        n_train = sum(1 for p in net.parameters() if p.requires_grad)
        print(f"  dummy loss={loss.item():.3f}; params w/ grad={n_grad}/{n_train} trainable")
