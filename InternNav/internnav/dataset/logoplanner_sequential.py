"""LogoPlanner-style sequential / streaming sampling for LoGoPlanner_Dataset.

Stage 2 of the RGB-only + LingBot-KV-cache plan. The dataloader is responsible
ONLY for delivering episode frames in temporal order with the metadata the model
needs to manage a per-episode KV cache (Stage 3). The KV cache itself lives in
the model, never here.

Two pieces:

  * ``StreamingEpisodeBatchSampler`` — a ``batch_sampler`` that keeps ``B`` lanes
    in flight. Each lane walks one episode's frames in order; when a lane's
    episode ends it picks up the next unused episode. Batch ``k`` is
    ``[lane_0 frame, lane_1 frame, ...]`` so row ``j`` of consecutive batches is
    the same episode until that episode finishes (then ``episode_start`` fires on
    the next frame and the model resets that row's cache). Frames of different
    episodes are never mixed inside one lane.

  * ``build_streaming_dataloader`` — convenience wrapper that wires the sampler to
    a ``LoGoPlanner_Dataset`` constructed with ``sequential=True``.

The dataset's ``__getitem__`` (flat index → (episode, timestep)) and the per-frame
metadata fields (episode_id / timestep / episode_start / reset_cache / done /
history masks) are defined in ``logoplanner_dataset_lerobot.py``.
"""

from collections import deque

from torch.utils.data import DataLoader, Sampler


class StreamingEpisodeBatchSampler(Sampler):
    """Yield lists of flat indices, lane-major, keeping ``batch_size`` episodes live.

    Args:
        episodes: list of episodes, each a list of flat dataset indices in
            temporal order (``dataset.episodes``).
        batch_size: number of concurrent lanes (= model batch rows).
        drop_ragged_tail: if True, stop once fewer than ``batch_size`` lanes remain
            active, so every emitted batch has exactly ``batch_size`` rows and
            lane→row alignment is preserved for the whole run (recommended when the
            model carries KV cache across batches). If False, the trailing
            partial batches are still emitted (rows compacted).
    """

    def __init__(self, episodes, batch_size, drop_ragged_tail=True):
        self.episodes = [ep for ep in episodes if len(ep) > 0]
        self.batch_size = batch_size
        self.drop_ragged_tail = drop_ragged_tail
        self._schedule = self._build_schedule()

    def _build_schedule(self):
        B = self.batch_size
        lanes = [None] * B            # each: [episode_idx, pos]
        queue = deque(range(len(self.episodes)))
        for i in range(B):
            if queue:
                lanes[i] = [queue.popleft(), 0]
        schedule = []
        while any(lane is not None for lane in lanes):
            if self.drop_ragged_tail and any(lane is None for lane in lanes):
                break
            batch = []
            for i in range(B):
                lane = lanes[i]
                if lane is None:
                    continue
                ep_idx, pos = lane
                batch.append(self.episodes[ep_idx][pos])
                pos += 1
                if pos >= len(self.episodes[ep_idx]):
                    lanes[i] = [queue.popleft(), 0] if queue else None
                else:
                    lanes[i][1] = pos
            if batch:
                schedule.append(batch)
        return schedule

    def __iter__(self):
        return iter(self._schedule)

    def __len__(self):
        return len(self._schedule)


def build_streaming_dataloader(dataset, batch_size, num_workers=0, collate_fn=None,
                               pin_memory=True, drop_ragged_tail=True):
    """DataLoader that streams ``dataset`` (sequential=True) episode-by-episode."""
    assert getattr(dataset, 'sequential', False), \
        'build_streaming_dataloader requires a dataset built with sequential=True'
    sampler = StreamingEpisodeBatchSampler(
        dataset.episodes, batch_size, drop_ragged_tail=drop_ragged_tail
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
