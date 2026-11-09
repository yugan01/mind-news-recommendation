# Project Tasks

The live task list and project state tracker. Everything about current progress — what is done, in progress, and remaining, per workstream — lives here. Canonical, team-wide agreed scope and direction live in the master document, `project-summary.md`; this file defers to it.

Legend: `[x]` done · `[ ]` open · `[~]` in progress. Update statuses as work advances.

## Data and evaluation infrastructure — Kevin

- [x] Chronological development/tuning split carved from the original-format training data; all model development runs only on it. [MVP] Latest 20% by timestamp; full timestamp ties stay together. Original training archive obtained from the Recommenders/MIND mirror.
- [x] Shared infrastructure rebuilt under `src/`: fixed settings in `src/config.py`, label-free model contract, popularity in `src/models/popularity.py`, centralized history preprocessing, and strict paired-example checks. Popularity development run: 5,000 users, 8,332 impressions; NDCG@5 0.214301, MRR 0.242211.
- [ ] Final evaluation: single validation run after model and preprocessing decisions are frozen. [MVP] Marcus will execute it with the completed models. Separate `python -m src.final` command and protocol documented; original validation has not been downloaded or evaluated.
- [x] Frozen semantic embeddings (`google/embeddinggemma-2`, text-only 270M): encode title+abstract once per article, cache vectors, average latest-20 history vectors, cosine scoring; use the shared model/evaluation interface. [MVP] Implementation, synthetic tests, real-model smoke, and full five-model development run complete; final validation remains unopened.
- [x] Initial test run with the frozen embeddings. [MVP] `python -m scripts.smoke_semantic` passed on four synthetic articles and two impressions, including finite normalized vectors, exports, an unseen user, and identical predictions after cache reuse. Shared processor dependencies installed via the standard image extra. No full development or validation experiment was run; synthetic metrics are not reportable results.
- [x] Semantic-model handoff: method notes, cache/run instructions, and Marcus's integration/development/final run sequence in `docs/model-handoff.md`. [MVP] Real-MIND smoke passed with 237 referenced articles, two training impressions, and two development impressions, exercising exports and paired analysis. No full semantic development experiment or final validation run was performed.

Current Kevin scope: implementation, small functional smoke tests, and handoff, now including beyond-accuracy metrics. Marcus will run the full shared experiments and final validation with his completed models; Kevin will not generate full semantic comparison results now. LLM judging remains deferred; whether to do it is undecided.
- [x] Beyond-accuracy metrics on prediction exports: top-5 mean pairwise cosine distance, add-one-smoothed training-popularity self-information, and candidate-catalogue coverage. [Extra] Implemented in `src/beyond.py`; synthetic tests, real-MIND smoke, and finite full development outputs for all five models passed. Definitions are in `docs/beyond-accuracy.md`.
- [ ] LLM-as-judge pipeline: deferred; need not yet established. [Extra] If pursued: sampled user subset, fixed published rubric, manual calibration spot-checks.

## Neural graph and trained models — Marcus

- [x] BPR-MF baseline; also establishes the shared BPR training loop (embedding tables, pair sampling, loss) that LightGCN reuses. [MVP] Development evidence: `python -m src.run` at `352b0c9`; see `docs/marcus-model-handoff.md`.
- [x] LightGCN on the user-article click graph, reusing the shared BPR training loop (normalized neighbor aggregation is the delta over BPR-MF). [MVP] Development evidence: `python -m src.run` at `352b0c9`; see `docs/marcus-model-handoff.md`.
- [x] Two-tower extension (user tower over history, article tower over frozen text vectors, trained on impression click labels). [Strongly recommended] Implemented only after the core gate opened; five-model development evidence from `python -m src.run` at `062dff7`, with byte-identical model rerun and method notes in `docs/marcus-model-handoff.md`.
- [x] Export results, predictions, configurations, and model-method notes. [MVP] Five paired development exports, ten comparisons, beyond-accuracy results, and method notes audited; generated outputs remain ignored and are handed off by path in `docs/marcus-model-handoff.md`.

## Evaluation, testing, and statistical comparison — Yugan

Completed on behalf of the original infrastructure package (counts toward this share):

- [x] Public GitHub repository created.
- [x] MIND-small dataset download and parsing (`scripts/download_mind.py`, `src/data.py`; original infrastructure by Yugan, subsequently rebuilt by Kevin).
- [x] Eligibility filters (history >= 10; impressions with >= 1 clicked and >= 1 non-clicked candidate).
- [x] NDCG@5 and MRR evaluation (`src/metrics.py`).
- [x] User-level paired bootstrap confidence intervals (`src/analyze.py`).
- [x] Per-impression prediction exports for paired analysis.
- [x] Automated tests for parser, ranking, and metrics (`tests/`).
- [x] Data pipeline plus a popularity baseline and an early TF-IDF prototype (prototype removed during the 19 Oct cleanup; BPR-MF is the sole classical baseline going forward).
- [x] Fixed 5,000-user smoke run (popularity vs the early TF-IDF prototype) confirming the pipeline works end to end; the run record was removed with the prototype.

Remaining:

- [~] Paired statistical comparisons and bootstrap CIs across all model pairs. [MVP] All ten development comparisons were generated and mechanically audited in `outputs/dev/comparisons.json`; Yugan's interpretation/review remains.
- [ ] Subgroup analyses: history length (10–19 vs 20+), interest narrowness (category entropy of history), user connectivity in the click graph. [MVP]
- [ ] History-length sensitivity: latest 5 / 10 / 20 articles on users with at least 20. [Extra]
- [ ] Failure-case collection: where one representation succeeds and another fails; graph cold-start users; unexpected popularity wins. Feeds the error analysis. [MVP]
- [ ] Test coverage for the new components (BPR-MF, embeddings, LightGCN, metrics tooling).
- [ ] Final charts and what-the-evidence-supports narrative.

## Deliverables and reporting — Sood

- [ ] Motivation, thesis, and related-work summary.
- [ ] Ethical analysis and limitations write-up, grounded in the exposure and diversity metrics.
- [ ] Poster/presentation design and assembly from method notes and analysis.
- [ ] Final deliverable editing for consistent methods, figures, and claims.
- [ ] Contribution statement for each team member.
- [ ] Submission coordination and rehearsals.

## Undecided / pending team decision

- [ ] Whether to redo the project critique deck for feedback on the new direction (Prof Kan offered; the old deck still counts for grades) — no owner assigned.
- [ ] Whether to present at the optional Week 10–12 critique session — no owner assigned.

## Milestones

- 26 October: implementation complete (Kevin and Marcus); handoff of working code, experiments, per-example predictions, and method notes to Yugan and Sood.
- 9 November: analysis and deliverables ready.
- 11 November: presentation.
