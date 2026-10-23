# AGENTS.md

Instructions for any AI agent working in this repository.

## Master document

`project-summary.md` (repo root) is the project's master document. It holds the canonical, team-wide agreed scope: core question, model list, evaluation plan, course-requirements mapping, and work breakdown.

- Read it before doing any project work.
- If any project scope change occurs, update the master document to reflect the new agreed reality.
- DO NOT bloat the master document. It contains only canonical, verified, explicitly team-wide agreed information. Unverified results, speculation, proposals not yet agreed by the team, implementation details, and running commentary do not belong there — keep those in the appropriate `docs/` file or a branch for team review.

## Tasks document

`tasks.md` (repo root) is the live task list and project state tracker. All state details — what is done, in progress, and remaining, per workstream — live there.

- Update task statuses as work advances; add tasks when new work is identified.
- Record completed work in `tasks.md`, not in the master document — the master document holds scope, not progress.
- Task lists may hold in-progress and not-yet-final information; the master document may not.

## Working agreement

- This is a one-off school project. Keep fixed experiment settings in `src/config.py` and commands argument-free. Do not add configurable pipeline frameworks, frozen-configuration systems, manifests, or software/artifact fingerprinting. New models belong in `src/models/` and follow `src/contracts.py`.
- The instructor feedback motivating the current direction is in `docs/instructor-feedback.md`.
- All models share identical inputs and protocol (title + abstract, latest 20 history articles, eligibility filters, same users and impressions); any new model must preserve this controlled-comparison contract.
- The supplied validation split is reserved for the final evaluation; develop only on a chronological split of the training data.
