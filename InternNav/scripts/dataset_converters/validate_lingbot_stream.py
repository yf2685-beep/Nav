"""Validate LingBotStream's window_forward / _stream_one against the official
LingBot-Map GCTStream streaming path.

Two checks:
  (A) DIRECT-vs-FORWARD: our `_aggregate_features(frame, block=1)` call must produce
      the SAME aggregator tokens (and same KV-cache evolution) as the official
      `GCTBase.forward(frame, ..., causal_inference=True)`, which is what
      `inference_streaming` uses. forward only adds _normalize_input (shape no-op)
      + prediction heads (which don't touch the aggregator KV cache) and routes
      causal_inference to the camera head only. Expect cosine ~ 1.0.
  (B) STREAM-ONE: _stream_one(frame) must equal one streaming `_aggregate_features`
      step (it is literally that, [None,None] + a[-1][:,-1]). Cross-checked inside (A).

This isolates the wrapper equivalence from the precompute-injection approximation
(that one is the only intended numerical difference and is checked separately by
comparing window_forward against full native streaming, not here).

Run in the `memnav` (or `lingbot-map`) env, needs GPU.
"""
import os
import sys

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from internnav.model.basemodel.memnav.lingbot_stream import LingBotStream

RGB_DIR = ("/home/asus/Research/datasets/InternData-N1/vln_n1/traj_data/"
           "matterport3d_d435i/17DRP5sb8fy/trajectory_89/videos/chunk-000/"
           "observation.images.rgb")
N = 14   # total frames to stream (scale block + a few streaming frames)


def cosine(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def main():
    ls = LingBotStream(device="cuda")
    model = ls.model
    S = ls.num_scale
    dev = ls.device
    dtype = torch.bfloat16

    paths = [os.path.join(RGB_DIR, f"{i}.jpg") for i in range(N)]
    imgs = ls.load_images(paths).to(dev)          # [N,3,H,W], LingBot-preprocessed
    print(f"frames={N} scale={S} img={tuple(imgs.shape)}")

    # ---- Run OFFICIAL path: GCTBase.forward, hooking the aggregator output ----
    captured = []
    h = model.aggregator.register_forward_hook(lambda m, i, o: captured.append(o))
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model.forward(imgs[:S][None], num_frame_for_scale=S,
                      num_frame_per_block=S, causal_inference=True)
        for j in range(S, N):
            model.forward(imgs[j:j + 1][None], num_frame_for_scale=S,
                          num_frame_per_block=1, causal_inference=True)
    h.remove()
    # captured[0] = scale block (S frames); captured[1:] = streaming frames
    off_stream = [c[0][-1][:, -1] for c in captured[1:]]    # [1,P,2C] each (frames S..N-1)

    # ---- Run OUR path: direct _aggregate_features (what window_forward/_stream_one use) ----
    model.clean_kv_cache()
    our_stream = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model._aggregate_features(imgs[:S][None], num_frame_for_scale=S, num_frame_per_block=S)
        for j in range(S, N):
            a, psi = model._aggregate_features(imgs[j:j + 1][None],
                                               num_frame_for_scale=S, num_frame_per_block=1)
            our_stream.append(a[-1][:, -1])

    print("\n(A) direct _aggregate_features  vs  official forward  (per streaming frame):")
    worst = 1.0
    for n, (o, u) in enumerate(zip(off_stream, our_stream), start=S):
        c = cosine(o, u)
        mad = (o.float() - u.float()).abs().max().item()
        worst = min(worst, c)
        print(f"  frame {n:2d}: cos={c:.6f}  max|Δ|={mad:.3e}  shape={tuple(u.shape)}")
    print(f"  -> worst cosine = {worst:.6f}")

    # ---- (B) _stream_one matches a streaming _aggregate_features step ----
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model._aggregate_features(imgs[:S][None], num_frame_for_scale=S, num_frame_per_block=S)
        for j in range(S, N - 1):
            model._aggregate_features(imgs[j:j + 1][None], num_frame_for_scale=S, num_frame_per_block=1)
        # last frame two ways from the SAME cache state -> snapshot via re-run
        tok_stream_one = ls._stream_one(imgs[N - 1])           # [1,P,2C]
    # reference: same final frame via direct aggregate from an identical cache
    model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        model._aggregate_features(imgs[:S][None], num_frame_for_scale=S, num_frame_per_block=S)
        for j in range(S, N - 1):
            model._aggregate_features(imgs[j:j + 1][None], num_frame_for_scale=S, num_frame_per_block=1)
        a, _ = model._aggregate_features(imgs[N - 1:N][None], num_frame_for_scale=S, num_frame_per_block=1)
        tok_ref = a[-1][:, -1]
    cB = cosine(tok_stream_one, tok_ref)
    print(f"\n(B) _stream_one vs direct streaming step: cos={cB:.6f} "
          f"shape={tuple(tok_stream_one.shape)}")

    ok = worst > 0.999 and cB > 0.999
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"(wrapper is {'lossless vs official forward' if ok else 'DIVERGING — investigate'})")


if __name__ == "__main__":
    main()
