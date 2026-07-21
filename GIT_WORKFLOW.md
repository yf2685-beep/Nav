# Git workflow (fork + upstream)

Set up 2026-07-20. Two remotes, following the standard fork convention:

| remote | points at | you |
|---|---|---|
| `upstream` | `glbreeze/Nav` (the shared lab repo) | **pull from** — never push |
| `origin` | `yf2685-beep/Nav` (your fork) | **push to** |

## Branches

| branch | tracks | what it is |
|---|---|---|
| `main` | `upstream/main` | **A pure mirror of upstream. Never commit here.** Always fast-forwards, so it can never conflict. |
| `yuxuan-experiments` | `origin/yuxuan-experiments` | Long-lived. `main` + your run-time conveniences (open3d guard, `MEMNAV_MAX_STEPS`, `MEMNAV_LIMIT`). Rebase it onto `main` whenever you sync. |
| `logoplanner-phase1` | `origin/logoplanner-phase1` | **Archive** of phase-1 work: the LoGoPlanner reproduction + improvements (14 commits, incl. PR #1). This used to be your fork's `main`. Frozen — don't build on it. |
| `fix/rotation-frame-conjugation` | `origin/fix/...` | Feature branch: the action-label forward-channel fix, ready to PR upstream. |

Phase-1 (LoGoPlanner) and phase-2 (MemNav) share only the initial commit — they are
independent lines of development. That is why `logoplanner-phase1` is an archive rather
than something to merge.

## Daily operations

**Get the latest upstream code** (always clean, never conflicts):
```bash
git checkout main
git pull                      # main tracks upstream/main -> fast-forward only
git push origin main          # optional: keep the fork's main mirror current
```

**Carry your experiment tweaks onto the new upstream:**
```bash
git checkout yuxuan-experiments
git rebase main               # replay your tweaks on top of the new upstream
git push --force-with-lease origin yuxuan-experiments
```
Use rebase, not merge, so this branch stays a thin readable stack of your own changes.

**Start a new piece of work destined for upstream:**
```bash
git checkout main
git checkout -b fix/something-descriptive
# ... work, commit ...
git push -u origin fix/something-descriptive
# then open a PR on GitHub: yf2685-beep:fix/... -> glbreeze:main
```
Branch off `main` (not `yuxuan-experiments`) so the PR contains only the change itself,
with none of your local conveniences mixed in.

## Rules that keep this from degrading

1. **Never commit to `main`.** It exists only to mirror upstream. If `git status` on `main`
   ever shows commits ahead of `upstream/main`, something went wrong — move them to a branch.
2. **Never push to `upstream`.** Contribute via PR from a branch on your fork.
3. **One change per branch.** Local conveniences live on `yuxuan-experiments`; anything you
   want upstream to take gets its own branch off `main`.
4. `--force-with-lease`, never bare `--force` — it refuses to clobber work you haven't seen.

## Things in this working tree that are NOT in git (deliberately)

- `InternNav/internnav/model/basemodel/LongCLIP` — upstream tracks this as a **symlink**;
  locally it is a real 17 MB directory so imports resolve. `git status` will permanently
  show it as deleted. **Do not commit that deletion**, and do not `git checkout` it back —
  that would replace the real content with a dangling symlink.
- `memnav_policy.py.ay2710_online_port` — the ported online-inference model
  (`predict_action` + the `online_caches` branch of `encode_memory`). Upstream has no online
  inference; this is what the closed-loop Habitat eval needs. Kept as a file, not a commit,
  because it conflicts with upstream's `batchify` rewrite of the same loop.
- `memnav_policy.py.main_bak` — an older backup of upstream's version.

## Recovery

- Pre-reorg fork `main` tip: `9453b0e`, preserved as `origin/logoplanner-phase1`.
- Local tag `backup/main-before-sync-20260720` marks the old local `main`.
- A stash from the 2026-07-20 pull is still present (`git stash list`).
