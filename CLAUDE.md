# Project memory

This repo's working memory lives at the repo root, **symlinked from the second-brain vault**
(canonical copy in `07 Projects/gbp-horse-prediction/`). Edit through the symlink — don't fork a copy.

- `Notes.md`     — current truth: overview, state, blockers.
- `LOG.md`       — append-only session log, newest first.
- `PLAN.md`      — active plan for in-flight work.

## Session start — read before acting

1. Read `Notes.md`, the top 2–3 `LOG.md` entries, and `PLAN.md` if present.
2. The latest open items in `LOG.md`/`Notes.md` are the starting point.

Don't re-derive what's already written. If `PLAN.md` and the ask disagree, confirm first.

## Session end / before auto-compact — write before stopping

1. Prepend or refresh a dated `LOG.md` entry (What changed / Decisions / Learnings / Open-next),
   written for a reader who wasn't there. Do this *before* compaction, not only at session end.
2. Update `Notes.md` if state changed.
3. A learning useful on a *different* project → flag for promotion to the vault.

These memory files are not code — touching them never requires touching the codebase.
Full convention: `Working with Claude.md` in the vault's `04 Indexes/`.
