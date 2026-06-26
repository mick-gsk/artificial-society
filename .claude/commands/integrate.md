---
description: Integrate the current branch — rebase on main, run the gate, push, open a PR.
---
Integrate the current feature branch for review. Do **not** push to or merge into
`main` directly (it is protected; the merge happens via the PR after CI + review).

1. `git fetch origin` then `git rebase origin/main`. Resolve conflicts only within
   **your lane**; a conflict in a hot/core file means coordinate with core-lead.
2. Run `bash scripts/check.sh`. If it fails, fix and repeat — do not open a PR red.
3. `git push -u origin HEAD`.
4. `gh pr create --fill` (or with a short title/body). State which lane the change
   is in and confirm no hot/core files were touched (unless this is a core-lead branch).
5. Report the PR URL. CI (ruff + full pytest incl. golden) gates the merge.
