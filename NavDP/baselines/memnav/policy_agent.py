"""MemNav live closed-loop agent.

Wraps the trained `MemNavPolicy` (InternNav, branch main) for frame-by-frame
inference: the agent ingests RGB frames one at a time (building the same
LingBot KV caches the training precompute writes to disk, incrementally and
in-memory), and on request plans a trajectory toward a goal image.

Design rule: every op is the TRAINING op. The live capture stream mirrors
`scripts/dataset_converters/precompute_lingbot_features.py:extract_trajectory`
step by step (scale block -> per-frame stream, write-once capture of the
newest cache slot); planning mirrors `MemNavNet.encode_memory` + the DDPM
reverse loop implied by its scheduler config. Where training reads an npz,
we read the identical in-memory dict.

Plan-time LingBot ops (window_forward / goal_append_warm / camera_pose)
destroy the streaming KV state, so plan() snapshots the aggregator +
camera-head caches and restores them afterwards — the capture stream stays
continuous, which is what makes the captured cam_pose_enc match precompute's.

Run inside the `memnav` conda env. Requires InternNav on sys.path (the
server adds it).
"""

import os
import shutil

import numpy as np
import torch


# ----------------------------------------------------------------------------- #
# helpers
# ----------------------------------------------------------------------------- #
class MemNavAgent:
    def __init__(self, checkpoint, internnav_root, device="cuda:0",
                 exclude_recent=83, num_samples=16, buffer_root=None,
                 gate_skip_below=0.0, retrieval_mode="raw", anchor_switch_margin=0.01):
        import sys
        if internnav_root not in sys.path:
            sys.path.insert(0, internnav_root)
        from internnav.model.basemodel.memnav.memnav_policy import (   # noqa: E402
            MemNavModelConfig, MemNavPolicy)
        from scripts.train.configs.memnav import memnav_exp_cfg        # noqa: E402

        self.policy = MemNavPolicy.from_pretrained(
            checkpoint, config=MemNavModelConfig(model_cfg=memnav_exp_cfg.model_dump()))
        self.policy.eval()
        self.core = self.policy.core
        self.lb = self.core.lingbot
        self.device = self.core.device
        self.S = self.lb.num_scale                      # 8
        self.W = self.lb.window                         # 32 (mp3d geometry)
        self.amargin = self.S + self.W - 1              # 39
        self.exclude_recent = int(exclude_recent)       # dataset default 83
        self.num_samples = int(num_samples)
        self.gate_skip_below = float(gate_skip_below)
        # "raw": match by RAW dino-cls cosine (frozen features; measured corr +0.29 with
        # GT covis, top-5 all in the GT neighborhood) instead of the trained projection
        # (measured corr -0.75 at ckpt-1500, top-5 all covis~0). Gate stays the trained
        # head's (decoder conditioning must match training).
        self.retrieval_mode = retrieval_mode
        # anchor hysteresis: per-frame scores are STATIC (goal fixed, cls write-once),
        # so the argmax moves only when a NEW candidate beats the incumbent. Switch
        # only on a clear win to keep novel-goal anchors sticky (no wasted warms).
        self.anchor_switch_margin = float(anchor_switch_margin)
        self.L_depth = self.lb.depth                    # aggregator layers
        self.psi = self.lb.num_special                  # 6 special tokens

        self.buffer_root = buffer_root or "/tmp/memnav_agent_buffer"
        os.makedirs(self.buffer_root, exist_ok=True)
        self._episode_counter = -1

        # dino-cls capture hook on the aggregator's patch_embed — same values the
        # precompute stores (`x_norm_clstoken` of the streaming forward itself).
        self._dino_out = [None]

        def _hook(_m, _i, out):
            self._dino_out[0] = out
        self.lb.agg.patch_embed.register_forward_hook(_hook)

        self.reset(camera_height=0.5)

    # ------------------------------------------------------------------ #
    # episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, camera_height=0.5):
        self._episode_counter += 1
        self.rgb_dir = os.path.join(self.buffer_root, f"ep_{self._episode_counter:04d}")
        shutil.rmtree(self.rgb_dir, ignore_errors=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
        self.camera_height = float(camera_height)
        self.n = 0                       # frames streamed so far
        self._pending = []               # preprocessed frames waiting for the scale block
        self._window_imgs = []           # last W preprocessed frames (cpu), for window_forward
        self.dino_cls = []               # per-frame [1024] fp32 cpu
        self.anchor_k = []               # per phase-2 frame [L,H,6,d] bf16 gpu
        self.anchor_v = []
        self.cam_k = []                  # per-frame [NI,TD,H,d] bf16 gpu (stacked lazily)
        self.cam_v = []
        self.cam_pose = []               # per-frame [9] fp32
        self.scale_k = None              # [L,H,S,P,d] bf16 gpu
        self.scale_v = None
        self._metric_scale = None        # lazy ground-anchored scale
        self._goal_cache = {}            # (goal_md5, anchor) -> goal_pose; goal_md5 -> goal_cls
        self._anchor_state = {}          # goal_md5 -> dict(m, score): sticky-anchor ratchet
        # tower-1 live capture: the current frame's post-GCT tokens + agg list from the
        # CONTINUOUS stream. Training used window_forward's cold-cache recompute only
        # because samples load from disk; at eval the live stream supersedes it.
        self._last_tokens = None         # [1, P, 2C] current frame post-GCT tokens
        self._last_agg = None            # list of [1,1,P,2C] (selected layers, current frame)
        self._psi = None                 # patch_start_idx from the scale block
        self.lb.model.clean_kv_cache()
        self.lb.model.camera_head.clean_kv_cache()

    # ------------------------------------------------------------------ #
    # capture-stream internals (mirrors precompute extract_trajectory)
    # ------------------------------------------------------------------ #
    def _pop_cls(self, n_frames):
        out = self._dino_out[0]
        cls = out["x_norm_clstoken"].reshape(n_frames, -1).float().cpu()
        return [cls[i] for i in range(n_frames)]

    def _read_anchor_newest(self):
        kv = self.lb.agg.kv_cache
        ak = torch.stack([kv[f"k_{i}"][0, :, -1, :self.psi].to(torch.bfloat16)
                          for i in range(self.L_depth)])
        av = torch.stack([kv[f"v_{i}"][0, :, -1, :self.psi].to(torch.bfloat16)
                          for i in range(self.L_depth)])
        return ak, av                                   # [L,H,6,d]

    def _read_cam_newest(self, n_new):
        ch = self.lb.model.camera_head
        NI, TD = ch.num_iterations, ch.trunk_depth
        ks, vs = [], []
        for it in range(NI):
            d = ch.kv_cache[it]
            ks.append(torch.stack([d[f"k_{bl}"][0, :, -n_new:, 0] for bl in range(TD)], 0))
            vs.append(torch.stack([d[f"v_{bl}"][0, :, -n_new:, 0] for bl in range(TD)], 0))
        # ks: list[NI] of [TD, H, n_new, d] -> [n_new, NI, TD, H, d]
        k = torch.stack(ks, 0).permute(3, 0, 1, 2, 4).to(torch.bfloat16)
        v = torch.stack(vs, 0).permute(3, 0, 1, 2, 4).to(torch.bfloat16)
        return [k[i] for i in range(n_new)], [v[i] for i in range(n_new)]

    def add_frame(self, jpg_bytes):
        """Ingest one RGB frame (jpg bytes). Returns the frame index."""
        idx = self.n
        path = os.path.join(self.rgb_dir, f"{idx}.jpg")
        with open(path, "wb") as f:
            f.write(jpg_bytes)
        img = self.lb.load_images([path])[0]            # [3,518,518] pad-518 (cpu)
        self._window_imgs.append(img)
        if len(self._window_imgs) > self.W:
            self._window_imgs.pop(0)

        ch = self.lb.model.camera_head
        if idx < self.S - 1:
            self._pending.append(img)
            self.n += 1
            return idx
        if idx == self.S - 1:
            # scale block: first S frames as ONE bidirectional block
            self._pending.append(img)
            blk = torch.stack(self._pending, 0)[None].to(self.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                agg, psi = self.lb.model._aggregate_features(
                    blk, num_frame_for_scale=self.S, num_frame_per_block=self.S)
                pl = ch(agg, causal_inference=True,
                        num_frame_per_block=self.S, num_frame_for_scale=self.S)
            self._psi = psi
            self._last_tokens = agg[-1][:, -1]
            self._last_agg = [layer[:, -1:] for layer in agg]
            self.dino_cls.extend(self._pop_cls(self.S))
            kv = self.lb.agg.kv_cache
            self.scale_k = torch.stack([kv[f"k_{i}"][0, :, :self.S].to(torch.bfloat16)
                                        for i in range(self.L_depth)]).contiguous()
            self.scale_v = torch.stack([kv[f"v_{i}"][0, :, :self.S].to(torch.bfloat16)
                                        for i in range(self.L_depth)]).contiguous()
            pose = pl[-1][0].float()                    # [S,9]
            self.cam_pose.extend([pose[i].cpu() for i in range(self.S)])
            ck, cv = self._read_cam_newest(self.S)
            self.cam_k.extend(ck); self.cam_v.extend(cv)
            self._pending = []
        else:
            # causal per-frame stream (num_frame_per_block=1) + write-once capture
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                agg, _ = self.lb.model._aggregate_features(
                    img[None, None].to(self.device),
                    num_frame_for_scale=self.S, num_frame_per_block=1)
                pl = ch(agg, causal_inference=True,
                        num_frame_per_block=1, num_frame_for_scale=self.S)
            self._last_tokens = agg[-1][:, -1]
            self._last_agg = [layer for layer in agg]
            self.dino_cls.extend(self._pop_cls(1))
            ak, av = self._read_anchor_newest()
            self.anchor_k.append(ak); self.anchor_v.append(av)
            self.cam_pose.append(pl[-1][0].float()[-1].cpu())
            ck, cv = self._read_cam_newest(1)
            self.cam_k.extend(ck); self.cam_v.extend(cv)
        self.n += 1
        return idx

    # ------------------------------------------------------------------ #
    # stream-state snapshot (plan ops destroy the KV caches)
    # ------------------------------------------------------------------ #
    def _snapshot(self):
        # Snapshot by REFERENCE, not clone: plan-time ops (window_forward /
        # goal_append_warm / camera_pose) start with clean_kv_cache + _inject,
        # which REPLACE dict entries — they never mutate the existing KV tensors
        # in place. Holding references keeps the old tensors alive at zero copy
        # cost (a full clone of the 32-frame window KV is ~5.5 GB and OOMs).
        agg = self.lb.agg
        ch = self.lb.model.camera_head
        return dict(
            kv=dict(agg.kv_cache),
            total=int(agg.total_frames_processed),
            cam=list(ch.kv_cache) if ch.kv_cache is not None else None,
            cam_idx=int(getattr(ch, "frame_idx", 0)),
        )

    def _restore(self, snap):
        agg = self.lb.agg
        ch = self.lb.model.camera_head
        self.lb.model.clean_kv_cache()
        agg.kv_cache.update(snap["kv"])
        agg.total_frames_processed = snap["total"]
        ch.kv_cache = snap["cam"]
        ch.frame_idx = snap["cam_idx"]
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # planning
    # ------------------------------------------------------------------ #
    def _live_cache(self):
        """The in-memory equivalent of MemNavNet._load_cache's dict."""
        n_anchor = len(self.anchor_k)
        if n_anchor > 0:
            ak = torch.stack(self.anchor_k, 2)          # [L,H,N,6,d]
            av = torch.stack(self.anchor_v, 2)
        else:
            L, H, d = self.scale_k.shape[0], self.scale_k.shape[1], self.scale_k.shape[-1]
            ak = self.scale_k.new_zeros((L, H, 0, self.psi, d))
            av = self.scale_k.new_zeros((L, H, 0, self.psi, d))
        return dict(
            scale_k=self.scale_k, scale_v=self.scale_v,
            anchor_k=ak, anchor_v=av,
            cam_k=torch.stack(self.cam_k, 0), cam_v=torch.stack(self.cam_v, 0),
            cam_pose_enc=torch.stack(self.cam_pose, 0).to(self.device),
            ground_h_est=None,
        )

    def _get_metric_scale(self):
        if self._metric_scale is None and self.n >= self.S:
            cam_pose = torch.stack(self.cam_pose, 0)
            s = self.lb.get_metric_scale(self.rgb_dir, cam_pose, self.camera_height)
            from internnav.model.basemodel.memnav.memnav_policy import RevisitMerge
            self._metric_scale = s if s is not None else RevisitMerge._SCALE
        return self._metric_scale

    @torch.no_grad()
    def plan(self, goal_jpg_bytes):
        """Plan toward a goal image. Returns dict with metre-space waypoints in the
        current camera planar frame (x forward, y left, theta CCW)."""
        k = self.n - 1
        lo = self.amargin
        if k < self.S + self.W:
            return dict(error=f"need >= {self.S + self.W + 1} frames, have {self.n}")

        gpath = os.path.join(self.rgb_dir, "_goal.jpg")
        with open(gpath, "wb") as f:
            f.write(goal_jpg_bytes)
        goal_img = self.lb.load_images([gpath])[0]      # [3,518,518]

        snap = self._snapshot()
        try:
            cache = self._live_cache()
            dev = self.device
            goal_t = goal_img[None].to(dev)

            # retrieval over candidates E(k) = [amargin .. k - exclude_recent]
            import hashlib
            gkey = hashlib.md5(goal_jpg_bytes).hexdigest()
            if ("cls", gkey) not in self._goal_cache:
                self._goal_cache[("cls", gkey)] = self.lb.dino(goal_t)["cls"]
            goal_cls = self._goal_cache[("cls", gkey)]                   # [1,1024]
            mem_cls = torch.stack(self.dino_cls, 0)[None].to(dev)        # [1,k+1,1024]
            cand = torch.zeros(1, k + 1, dtype=torch.bool, device=dev)
            hi = k - self.exclude_recent
            if hi >= lo:
                cand[0, lo:hi + 1] = True
            match_idx, gate_logit, _ = self.core.retrieval(goal_cls, mem_cls, cand)
            gate = torch.sigmoid(gate_logit)     # trained gate: decoder soft-bias, as in training

            raw_score = None
            if self.retrieval_mode == "raw" and cand.any():
                import torch.nn.functional as Fnn
                raw_cos = Fnn.cosine_similarity(goal_cls.unsqueeze(1), mem_cls, dim=-1)[0]
                raw_cos = raw_cos.masked_fill(~cand[0], -1.0)
                cand_best = int(raw_cos.argmax().item())
                raw_score = float(raw_cos[cand_best].item())
                st = self._anchor_state.get(gkey)
                # ratchet: keep the incumbent unless the new best clearly beats it
                if st is not None and raw_score <= st["score"] + self.anchor_switch_margin:
                    match = st["m"]
                else:
                    match = cand_best
                    self._anchor_state[gkey] = dict(m=cand_best, score=raw_score)
                match_idx = torch.tensor([match], device=dev)
            anchor = int(match_idx.clamp(lo, k - 1).item())

            # current state (tower 1): LIVE stream tokens — no window recompute at eval.
            # Training's window_forward cold-cache pass exists only because samples load
            # from disk; the continuous stream's tokens carry full causal context.
            cur_t = self._last_tokens                                    # [1,P,2C]
            cur_img = self._window_imgs[-1]
            dfeat = self.lb.depth_feature(self._last_agg, cur_img[None][None], self._psi)[None]

            # poses: current from the continuous capture stream; goal via warm re-insert.
            # goal_pose depends only on (goal image, anchor, caches[<=anchor]) and the
            # captured caches are write-once, so this cache is EXACT — recompute only
            # when retrieval moves the anchor (saves the 64-frame warm ~10s per plan).
            cur_pose = cache["cam_pose_enc"][k][None]                    # [1,9]
            pkey = ("pose", gkey, anchor)
            # Gate-conditioned tower 2: only pay the goal-insert when retrieval says
            # revisit (or the pose is already cached). When skipped, the revisit
            # readout is zeroed — NOTE this is a deliberate eval-time deviation:
            # training always computes revisit tokens and lets the soft gate
            # (log(gate) attention bias) mask them; zeroing changes what low-gate
            # steps condition on. Revisit if warm-arm results look off.
            goal_pose = self._goal_cache.get(pkey)
            if goal_pose is None and float(gate.item()) >= self.gate_skip_below:
                # warm all the way back to the scale block (n_hist=0). Injected
                # compressed-history frames poison the camera head's goal pose
                # (measured: n_hist=34 -> 34° yaw err, 100 -> 169°; 0 -> ~0°
                # — Aiden_eval/memnav_eval/FINDINGS.md §2.5/§3). Cost is once
                # per (goal, m) via the cache, not per step.
                warm_full = max(self.core.goal_warm, anchor - self.S + 1)
                _, goal_agg = self.lb.goal_append_warm(
                    goal_img, cache, anchor, self.rgb_dir, warm_full, return_agg=True)
                goal_pose = self.lb.camera_pose(
                    cache["cam_k"], cache["cam_v"], anchor + 1, goal_agg)[-1][None]
                self._goal_cache[pkey] = goal_pose

            mscale = torch.tensor([self._get_metric_scale()], device=dev, dtype=torch.float32)
            current_state = self.core.build_current_state(cur_t, dfeat)
            if goal_pose is not None:
                revisit, aux_pose, _ = self.core.build_revisit(cur_pose.to(dev), goal_pose.to(dev), mscale)
            else:
                revisit = torch.zeros((1, self.core.n_rev, self.core.action_head.in_features), device=dev)
                aux_pose = torch.zeros((1, 2), device=dev)
            novel = self.core.novel(cur_img[None].to(dev), goal_t)

            # DDPM reverse loop (no critic in this model)
            N = self.num_samples
            cs = current_state.expand(N, -1, -1)
            rv = revisit.expand(N, -1, -1)
            nv = novel.expand(N, -1, -1)
            gt = gate.expand(N)
            sched = self.core.noise_scheduler
            naction = torch.randn((N, self.core.predict_size, 3), device=dev)
            sched.set_timesteps(sched.config.num_train_timesteps)
            for t in sched.timesteps:
                eps = self.core.predict_noise(naction, t[None].to(dev), cs, rv, nv, gt)
                naction = sched.step(eps, t, naction).prev_sample

            # decode: normalized deltas / 4 -> metres; cumsum -> waypoints
            deltas = (naction / 4.0).float().cpu().numpy()               # [N,24,3]
            paths = np.cumsum(deltas, axis=1)
            ends = paths[:, -1, :2]
            medoid = int(np.argmin(np.linalg.norm(ends - ends.mean(0), axis=1)))
            return dict(
                trajectory=paths[medoid].tolist(),
                all_trajectory=paths.tolist(),
                all_values=[0.0] * N,
                gate=float(gate.item()),
                match_idx=int(match_idx.item()),
                raw_score=raw_score,
                anchor=anchor,
                aux_pose=aux_pose[0].float().cpu().tolist(),
                frame_idx=k,
            )
        finally:
            self._restore(snap)
