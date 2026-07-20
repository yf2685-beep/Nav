# FROZEN — phase-1 checkout (LoGoPlanner)

Retired 2026-07-20. Do not modify, run, or import from this tree.

Phase 1 was the LoGoPlanner reproduction and improvement work. Development has moved to
the memnav line, in a separate checkout that tracks upstream `glbreeze/Nav`:

    ../memnav/code        (dev tree; see its GIT_WORKFLOW.md)
    186:/home/nyuair/memnav_src   (the tree experiments actually run from)

## Where this tree's work lives now

Everything here is on the fork (`yf2685-beep/Nav`), consolidated into two branches:

| branch | contents |
|---|---|
| `navdp` | all phase-1 LoGoPlanner work: the old fork `main` (14 commits incl. PR #1), Method 1 (frozen LingBot-Map + Adapter), Method 2 (AggregatorStream + DA-S fusion, streaming GCT, RGB-only, multi-stop subgoal, collision critic), the EXPERIMENTS_LINGBOT.md report, and the imagegoal/pointgoal eval metric dumps |
| `memnav` | phase-2 work: upstream/main + local run-time conveniences + the action-label rotation-frame fix, the retrieval-loss branch folded in for the record, and the first-generation memnav scaffolding that used to be loose in this tree |

Method 1 and Method 2 are preserved as alternative backends selected at runtime by
`LOGO_BACKBONE=lingbot_map` / `lingbot_v2` — merging them did not force a choice.

Worth knowing: `NavDP/baselines/memnav/eval_seen_unseen_stats.py` originated here. It was
believed lost with the old 131 machine and was re-implemented on dgx; the original is on
the `memnav` branch now.

## Size

~88 GB, mostly data and eval outputs rather than source. Safe to delete once you are
satisfied the two branches above cover what you need — but check for large local artifacts
that were never committed first.
