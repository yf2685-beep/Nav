"""MemNav inference engine — live image-goal navigation.

Reuses the *trained* InternNav `MemNavNet` (so the `memnav.ckpt` trainable heads load
directly) and drives it from a live frame buffer instead of a precomputed cache:

  per step k:
    1. stream the frame buffer [0..k] through frozen LingBot -> in-memory KV cache
       (reuses precompute `extract_trajectory`, so it is bit-identical to training's cache)
    2. run the SAME encode_memory ops (retrieval -> window_forward / depth / camera_pose /
       goal_append) but fed from memory + the external goal image
    3. DDPM sampling (mg branch, gate-biased) -> local trajectory (cumsum(naction/4))

NOTE (perf): the cache is recomputed from the whole buffer each step (O(k) LingBot
forwards). Correct + consistent with training, but O(k^2) per episode. A later
optimization is true incremental streaming (keep the KV cache live across steps).

Run in the `enerverse` env (same as training). Needs the lingbot-map repo on sys.path
(handled by LingBotStream via lingbot_repo).
"""
import os
import sys

import numpy as np
import torch

# InternNav (training-side model) + precompute cache extractor
NAV_INTERNNAV = "/home/nyuair/yuxuan/1 robot navigation/Nav/InternNav"
if NAV_INTERNNAV not in sys.path:
    sys.path.insert(0, NAV_INTERNNAV)
sys.path.insert(0, os.path.join(NAV_INTERNNAV, "scripts", "dataset_converters"))

from internnav.model.basemodel.memnav.memnav_policy import MemNavNet
from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream
from precompute_lingbot_features import extract_trajectory


class MemNavInference:
    def __init__(self, checkpoint, lingbot_repo, lingbot_weights,
                 predict_size=24, num_diffusion_iters=10, device="cuda:0"):
        self.device = device
        self.net = MemNavNet(
            lingbot_kwargs=dict(lingbot_repo=lingbot_repo, weights=lingbot_weights),
            predict_size=predict_size, num_diffusion_iters=num_diffusion_iters, device=device,
        )
        self.net.eval()
        # load the trained heads (frozen LingBot weights already loaded in MemNavNet ctor)
        if checkpoint and os.path.exists(checkpoint):
            sd = torch.load(checkpoint, map_location="cpu")
            sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
            # HF-Trainer checkpoints wrap the net under a `core.` attribute; the server loads into
            # MemNavNet directly, so strip that prefix (pilot ckpts have none — handle both).
            if any(k.startswith("core.") for k in sd):
                sd = {(k[len("core."):] if k.startswith("core.") else k): v for k, v in sd.items()}
            inc = self.net.load_state_dict(sd, strict=False)
            trained = [k for k in sd if "lingbot." not in k]
            print(f"[MemNavInference] loaded {checkpoint}: {len(trained)} trained tensors, "
                  f"missing={len(inc.missing_keys)} unexpected={len(inc.unexpected_keys)}")
        else:
            print(f"[MemNavInference] WARNING: no checkpoint at {checkpoint} — untrained heads")

        self.model = self.net.lingbot.model            # raw GCTStream (for extract_trajectory)
        self.num_scale = self.net.num_scale
        self.window = self.net.window
        self.lo = self.num_scale + self.window - 1      # min frames for a full window (=15)
        self.predict_size = predict_size
        # DINOv2 CLS capture hook (as in precompute)
        self._dino_cap = [None]
        self.model.aggregator.patch_embed.register_forward_hook(
            lambda m, i, o: self._dino_cap.__setitem__(0, o))

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _build_cache(self, images_1SHW):
        """images [1,S,3,H,W] preprocessed -> device cache dict (scale/anchor/cam layered) + dino_cls."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = extract_trajectory(self.model, images_1SHW.to(self.device),
                                       self.num_scale, self._dino_cap, cam_only=False)
        sk, sv, ak, av = LingBotStream._cache_to_layered(
            feats["scale_k"], feats["scale_v"], feats["anchor_k"], feats["anchor_v"], self.device)
        ck, cv = LingBotStream._cam_to_device(feats["cam_k"], feats["cam_v"], self.device)
        cache = dict(scale_k=sk, scale_v=sv, anchor_k=ak, anchor_v=av, cam_k=ck, cam_v=cv)
        return cache, feats["dino_cls"]

    @torch.no_grad()
    def _encode_live(self, cache, dino_cls_all, goal_cls, goal_img, window_imgs, buffer_frames, k):
        """B=1 mirror of MemNavNet.encode_memory, fed from memory + external goal image."""
        net, dev = self.net, self.device
        mem_cls = torch.as_tensor(dino_cls_all, dtype=torch.float32, device=dev)[None]  # [1,k+1,1024]
        mem_mask = torch.ones(1, mem_cls.shape[1], dtype=torch.bool, device=dev)
        match_idx, gate, ret_logits = net.retrieval(goal_cls[None].to(dev), mem_cls, mem_mask)

        W = self.window
        ck, cv = cache["cam_k"], cache["cam_v"]
        wt, cur_agg, psi = net.lingbot.window_forward(cache, window_imgs.to(dev), k, return_multilayer=True)
        cur = wt[-1]                                                          # [P, 2C]
        dfeat = net.lingbot.depth_feature(cur_agg, window_imgs[-1:][None].to(dev), psi)
        cur_pose = net.lingbot.camera_pose(ck, cv, k, cur_agg)[-1]            # [9]
        m = int(match_idx[0].clamp(self.lo, k - 1).item())
        mw = torch.stack([buffer_frames[i] for i in range(m - W + 1, m + 1)]).to(dev)
        _, goal_agg = net.lingbot.goal_append(goal_img.to(dev), cache, m, mw, return_agg=True)
        goal_pose = net.lingbot.camera_pose(ck, cv, m + 1, goal_agg)[-1]      # [9]
        return cur, dfeat, cur_pose, goal_pose, gate, match_idx, ret_logits

    @torch.no_grad()
    def predict(self, buffer_frames, goal_img, sample_num=4):
        """buffer_frames: list of [3,H,W] preprocessed frames (index 0..k, current = last).
        goal_img: [3,H,W] preprocessed goal. Returns (trajectories [sample_num, predict_size, 3]
        local waypoints, info dict{match_idx, gate, logits}).  (None, None) if warming up."""
        k = len(buffer_frames) - 1
        if k < self.lo:
            return None, None                                    # warmup: caller nudges forward
        images = torch.stack(buffer_frames)[None]                # [1,S,3,H,W]
        cache, dino_cls_all = self._build_cache(images)
        goal_cls = self.net.lingbot.dino(goal_img[None].to(self.device))["cls"][0]   # [1024]
        window_imgs = torch.stack(buffer_frames[k - self.window + 1: k + 1])         # [W,3,H,W]
        cur, dfeat, cur_pose, goal_pose, gate, match_idx, ret_logits = self._encode_live(
            cache, dino_cls_all, goal_cls, goal_img, window_imgs, buffer_frames, k)

        net = self.net
        current_state = net.build_current_state(cur[None], dfeat[None])       # [1,n_cs,D]
        revisit, _aux = net.build_revisit(cur_pose[None], goal_pose[None])    # [1,n_rev,D]
        novel = net.novel(window_imgs[-1][None].to(self.device), goal_img[None].to(self.device))

        # expand conditioning to sample_num
        S = sample_num
        cs = current_state.expand(S, -1, -1)
        rev = revisit.expand(S, -1, -1)
        nov = novel.expand(S, -1, -1)
        g = gate.expand(S)

        # DDPM sampling (mg branch, gate-biased cross-attention)
        naction = torch.randn(S, self.predict_size, 3, device=self.device)
        sched = net.noise_scheduler
        sched.set_timesteps(sched.config.num_train_timesteps)
        for t in sched.timesteps:
            tt = torch.full((S,), int(t), device=self.device, dtype=torch.long)
            noise_pred = net.predict_noise(naction, tt, cs, rev, nov, g, "mg")
            naction = sched.step(noise_pred, int(t), naction).prev_sample
        traj = torch.cumsum(naction / 4.0, dim=1)                # [S, predict_size, 3] local waypoints
        info = dict(match_idx=int(match_idx[0].item()),
                    gate=float(gate[0].item()),
                    logits=ret_logits[0].float().cpu().numpy())
        return traj.float().cpu().numpy(), info
