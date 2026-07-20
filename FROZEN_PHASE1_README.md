# FROZEN — phase-1 checkout (LoGoPlanner)

Retired 2026-07-20. Do not modify, run, or import from this tree.

Phase 1 was the LoGoPlanner reproduction and improvement work. Development has
moved to the memnav line in a separate checkout that tracks upstream
`glbreeze/Nav`:

    ../memnav/code        (main mirrors upstream; see its GIT_WORKFLOW.md)

Everything in this tree is preserved on the fork (`yf2685-beep/Nav`):

  phase1-freeze-20260720          full snapshot incl. previously-loose files
  logoplanner-phase1              the old fork main (14 commits, incl. PR #1)
  method1-lingbot-map             Method 1 — frozen LingBot-Map + Adapter
  method2-lingbot-v2              Method 2 — AggregatorStream + DA-S fusion
  lingbot-experiments-report      EXPERIMENTS_LINGBOT.md
  yuxuan-retrieval-loss-and-fixes windowed soft-label retrieval loss + fixes

Worth knowing: `NavDP/baselines/memnav/eval_seen_unseen_stats.py` lives here.
It was thought lost with the old 131 machine and was rebuilt on dgx; the
original is on the freeze branch.

This directory is ~88 GB, mostly data/outputs rather than source. Safe to
delete once you are confident the branches above cover what you need.
