---
description: Run the local gate — ruff on changed files + full pytest (mirrors CI).
---
Run `bash scripts/check.sh` from the repo root and report the result concisely.

- A green run predicts a green CI run.
- If ruff fails, fix the lint/format issues in the files **you** changed.
- If pytest fails, the determinism contract (golden trajectory / headless digest)
  or a unit test broke. **Never edit a determinism test to make it pass** — a red
  golden trajectory means your change altered behaviour; fix the change. See
  `docs/ownership.md`.
