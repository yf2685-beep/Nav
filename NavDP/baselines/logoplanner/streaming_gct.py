"""Streaming Geometric-Context conditioning for LoGoPlanner.

This module turns LoGoPlanner's geometry backbone into a *streaming* navigation
context, faithful to LingBot-Map's Geometric Context Transformer (GCT, arXiv
2604.14141 §3.2). Instead of crushing a ~6.5 m episode into 12 evenly-sampled
frames (the old `policy_agent.get_indices` bottleneck), the policy now sees:

  * **anchor context**   — the first ``n_anchor`` frames of the episode (scale /
    coordinate grounding; "directional attention" / bidirectional scale frames),
  * **trajectory memory** — a compact summary of all frames between the anchors
    and the recent window (long-range drift context),
  * **local pose-reference window** — the most recent ``n_window`` frames
    (dense relative-pose cues),
  * **current frame**    — the decision-step observation,

pooled into a small, fixed-size set of "GCT summary tokens" that condition the
diffusion policy.

Two paths share one ``GCTSummaryAssembler``:
  * Training  → :func:`partition_window_tokens` over a bounded sampled window
    ``[anchor | trajectory keyframes | recent window]`` produced by a single
    parallel causal backbone forward (``num_frame_for_scale = n_anchor``). The
    sliding-window / token-dropping in GCT is an *inference* efficiency device;
    over a bounded training window full causal attention is faithful and simpler.
  * Inference → :class:`StreamingContextBuffer` accumulates per-frame tokens
    emitted by the backbone's frame-by-frame KV-cache streaming, then assembles
    the same summary every decision step.

Per-frame tokens are the 384-d ``state_token`` (from the camera token) and
``scene_token`` (pooled patch tokens) that ``GeometryModel_LingBot`` already
produces, so nothing here touches LingBot's attention internals.
"""

from collections import deque
from typing import Optional, Tuple, List

import torch
import torch.nn as nn

from policy_backbone import TokenCompressor


# ---------------------------------------------------------------------------
# Window layout helpers (shared by dataset + assembler so indices never drift)
# ---------------------------------------------------------------------------
def window_total_frames(n_anchor: int, n_traj: int, n_window: int) -> int:
    """Total frames in a training window = anchors + trajectory keyframes + window."""
    return n_anchor + n_traj + n_window


def partition_window_tokens(
    state_tok: torch.Tensor,
    scene_tok: torch.Tensor,
    n_anchor: int,
    n_window: int,
) -> dict:
    """Split per-frame tokens of a bounded window into GCT regions.

    Frames must be ordered ``[anchor_0..n_anchor-1 | trajectory... | window...]``
    with the **decision frame as the very last** entry.

    Args:
        state_tok / scene_tok: ``(B, N, D)`` per-frame tokens.
        n_anchor: number of leading anchor frames.
        n_window: number of trailing pose-reference-window frames.

    Returns:
        dict of ``(B, k, D)`` slices: ``anchor_state/anchor_scene``,
        ``traj_state/traj_scene`` (may have ``k==0``),
        ``window_state/window_scene``, and ``cur_state/cur_scene`` ``(B, 1, D)``.
    """
    B, N, D = state_tok.shape
    assert N >= n_anchor + n_window, (
        f"window too short: N={N} < n_anchor({n_anchor})+n_window({n_window})"
    )
    a_e = n_anchor
    w_s = N - n_window
    return {
        "anchor_state": state_tok[:, :a_e],
        "anchor_scene": scene_tok[:, :a_e],
        "traj_state": state_tok[:, a_e:w_s],
        "traj_scene": scene_tok[:, a_e:w_s],
        "window_state": state_tok[:, w_s:],
        "window_scene": scene_tok[:, w_s:],
        "cur_state": state_tok[:, -1:],
        "cur_scene": scene_tok[:, -1:],
    }


# ---------------------------------------------------------------------------
# Trainable assembler: per-region attention pooling -> fixed summary tokens
# ---------------------------------------------------------------------------
class GCTSummaryAssembler(nn.Module):
    """Pool GCT regions into a fixed-size conditioning set for the diffusion head.

    Output token order (each ``(B, 1, D)``), total ``8`` tokens::

        [anchor_state, anchor_scene,
         traj_state,   traj_scene,
         window_state, window_scene,
         cur_state,    cur_scene]

    Anchor / trajectory / window regions are pooled with independent
    :class:`TokenCompressor` (cross-attention to a learned query); the current
    frame's tokens are passed through directly. Empty trajectory regions (short
    episodes) fall back to a learned placeholder so the summary length is fixed.
    """

    N_SUMMARY = 8

    def __init__(self, dim: int = 384, heads: int = 8):
        super().__init__()
        self.dim = dim
        self.anchor_state_pool = TokenCompressor(dim, heads, target_length=1)
        self.anchor_scene_pool = TokenCompressor(dim, heads, target_length=1)
        self.traj_state_pool = TokenCompressor(dim, heads, target_length=1)
        self.traj_scene_pool = TokenCompressor(dim, heads, target_length=1)
        self.window_state_pool = TokenCompressor(dim, heads, target_length=1)
        self.window_scene_pool = TokenCompressor(dim, heads, target_length=1)
        # Learned placeholders for an empty trajectory-memory region.
        self.empty_traj_state = nn.Parameter(torch.zeros(1, 1, dim))
        self.empty_traj_scene = nn.Parameter(torch.zeros(1, 1, dim))

    def _pool(self, pool: TokenCompressor, x: torch.Tensor,
              empty: Optional[torch.Tensor]) -> torch.Tensor:
        # x: (B, k, D) -> (B, 1, D); k may be 0.
        B = x.shape[0]
        if x.shape[1] == 0:
            assert empty is not None, "empty region with no placeholder"
            return empty.expand(B, -1, -1).to(x.dtype)
        return pool(x)

    def forward(self, parts: dict) -> torch.Tensor:
        """parts: output of :func:`partition_window_tokens`. -> (B, 8, D)."""
        anchor_s = self._pool(self.anchor_state_pool, parts["anchor_state"], None)
        anchor_c = self._pool(self.anchor_scene_pool, parts["anchor_scene"], None)
        traj_s = self._pool(self.traj_state_pool, parts["traj_state"], self.empty_traj_state)
        traj_c = self._pool(self.traj_scene_pool, parts["traj_scene"], self.empty_traj_scene)
        window_s = self._pool(self.window_state_pool, parts["window_state"], None)
        window_c = self._pool(self.window_scene_pool, parts["window_scene"], None)
        cur_s = parts["cur_state"]
        cur_c = parts["cur_scene"]
        summary = torch.cat(
            [anchor_s, anchor_c, traj_s, traj_c, window_s, window_c, cur_s, cur_c],
            dim=1,
        )
        return summary  # (B, 8, D)


# ---------------------------------------------------------------------------
# Inference-time per-frame token buffer (mirrors the GCT streaming state)
# ---------------------------------------------------------------------------
class StreamingContextBuffer:
    """Accumulate per-frame ``(state, scene)`` tokens during streaming inference.

    Bookkeeping only — the actual GCT attention + KV cache live in the backbone
    (``GeometryModel_LingBot.step_streaming``). This buffer just keeps enough
    384-d tokens to rebuild the GCT summary each decision step:

      * the first ``n_anchor`` frames (anchors, never evicted),
      * a deque of the last ``n_window`` frames (pose-reference window),
      * every in-between frame (trajectory memory; 384-d so cheap even for
        thousands of frames).

    One buffer instance per environment. Call :meth:`reset` at episode start.
    """

    def __init__(self, n_anchor: int = 8, n_window: int = 64):
        self.n_anchor = n_anchor
        self.n_window = n_window
        self.reset()

    def reset(self):
        self.count = 0
        self._anchor_s: List[torch.Tensor] = []
        self._anchor_c: List[torch.Tensor] = []
        self._traj_s: List[torch.Tensor] = []
        self._traj_c: List[torch.Tensor] = []
        self._win_s: deque = deque(maxlen=self.n_window)
        self._win_c: deque = deque(maxlen=self.n_window)

    def step(self, state_t: torch.Tensor, scene_t: torch.Tensor):
        """Add the current frame's tokens. Each is ``(D,)`` or ``(1, D)``."""
        s = state_t.reshape(-1)
        c = scene_t.reshape(-1)
        if self.count < self.n_anchor:
            self._anchor_s.append(s)
            self._anchor_c.append(c)
        else:
            # A frame leaving the recent window becomes trajectory memory.
            if len(self._win_s) == self.n_window:
                self._traj_s.append(self._win_s[0])
                self._traj_c.append(self._win_c[0])
            self._win_s.append(s)
            self._win_c.append(c)
        self.count += 1

    @staticmethod
    def _stack(items: List[torch.Tensor], dim: int, device, dtype) -> torch.Tensor:
        if len(items) == 0:
            return torch.zeros(1, 0, dim, device=device, dtype=dtype)
        return torch.stack(list(items), dim=0).unsqueeze(0)  # (1, k, D)

    def build_parts(self) -> dict:
        """Return the same dict structure as :func:`partition_window_tokens`.

        Anchors that haven't filled yet are taken from whatever exists; the
        current frame is the most recent token seen (window's last, else the
        last anchor).
        """
        assert self.count > 0, "build_parts called on empty buffer"
        # infer dim/device/dtype from any stored token
        ref = (self._anchor_s or list(self._win_s))[-1]
        dim, device, dtype = ref.shape[0], ref.device, ref.dtype
        anchor_state = self._stack(self._anchor_s, dim, device, dtype)
        anchor_scene = self._stack(self._anchor_c, dim, device, dtype)
        traj_state = self._stack(self._traj_s, dim, device, dtype)
        traj_scene = self._stack(self._traj_c, dim, device, dtype)
        window_state = self._stack(list(self._win_s), dim, device, dtype)
        window_scene = self._stack(list(self._win_c), dim, device, dtype)
        # current frame = most recent token
        if len(self._win_s) > 0:
            cur_state = self._win_s[-1].view(1, 1, dim)
            cur_scene = self._win_c[-1].view(1, 1, dim)
        else:
            cur_state = self._anchor_s[-1].view(1, 1, dim)
            cur_scene = self._anchor_c[-1].view(1, 1, dim)
        # Anchor pool must be non-empty for GCTSummaryAssembler; guarantee >=1.
        if anchor_state.shape[1] == 0:
            anchor_state, anchor_scene = cur_state, cur_scene
        if window_state.shape[1] == 0:
            window_state, window_scene = cur_state, cur_scene
        return {
            "anchor_state": anchor_state, "anchor_scene": anchor_scene,
            "traj_state": traj_state, "traj_scene": traj_scene,
            "window_state": window_state, "window_scene": window_scene,
            "cur_state": cur_state, "cur_scene": cur_scene,
        }


if __name__ == "__main__":
    # Smoke test: assembler + buffer shapes, empty-traj fallback, NaN check.
    torch.manual_seed(0)
    D = 384
    asm = GCTSummaryAssembler(dim=D, heads=8)

    # (1) Training path: bounded window [8 anchor + 16 traj + 64 window] = 88.
    n_anchor, n_traj, n_window = 8, 16, 64
    N = window_total_frames(n_anchor, n_traj, n_window)
    state = torch.randn(2, N, D)
    scene = torch.randn(2, N, D)
    parts = partition_window_tokens(state, scene, n_anchor, n_window)
    print("[train] window N =", N,
          "| anchor", parts["anchor_state"].shape[1],
          "traj", parts["traj_state"].shape[1],
          "window", parts["window_state"].shape[1])
    summ = asm(parts)
    print("[train] summary", tuple(summ.shape), "finite:", torch.isfinite(summ).all().item())
    assert summ.shape == (2, GCTSummaryAssembler.N_SUMMARY, D)

    # (2) Short episode -> empty trajectory region uses placeholder.
    short = partition_window_tokens(torch.randn(2, n_anchor + n_window, D),
                                    torch.randn(2, n_anchor + n_window, D),
                                    n_anchor, n_window)
    print("[short] traj frames:", short["traj_state"].shape[1])
    summ_s = asm(short)
    assert summ_s.shape == (2, GCTSummaryAssembler.N_SUMMARY, D)
    print("[short] summary", tuple(summ_s.shape), "finite:", torch.isfinite(summ_s).all().item())

    # (3) Inference buffer: stream 200 frames, bounded window, growing traj memory.
    buf = StreamingContextBuffer(n_anchor=8, n_window=64)
    for t in range(200):
        buf.step(torch.randn(D), torch.randn(D))
    bparts = buf.build_parts()
    print("[stream] after 200 frames -> anchor", bparts["anchor_state"].shape[1],
          "traj", bparts["traj_state"].shape[1],
          "window", bparts["window_state"].shape[1])
    assert bparts["window_state"].shape[1] == 64
    assert bparts["traj_state"].shape[1] == 200 - 8 - 64
    summ_b = asm(bparts)
    assert summ_b.shape == (1, GCTSummaryAssembler.N_SUMMARY, D)
    print("[stream] summary", tuple(summ_b.shape), "finite:", torch.isfinite(summ_b).all().item())
    print("ALL SMOKE CHECKS PASSED")
