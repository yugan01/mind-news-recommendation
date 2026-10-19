# Project Tasks

The live task list and project state tracker. Everything about current progress — what is done, in progress, and remaining, per workstream — lives here. Canonical, team-wide agreed scope and direction live in the master document, `project-summary.md`; this file defers to it.

Legend: `[x]` done · `[ ]` open · `[~]` in progress. Update statuses as work advances.

## Data and evaluation infrastructure — Kevin

- [ ] Chronological development/tuning split carved from the training data; all further development runs only on it.
- [ ] Final-evaluation protocol: single validation run after decisions are frozen, disclosing the earlier smoke use.
- [ ] BPR-MF baseline; also establishes the shared BPR training loop (embedding tables, pair sampling, loss) that LightGCN reuses.
- [ ] Frozen semantic embeddings (`all-MiniLM-L6-v2`): encode title+abstract once per article, cache vectors, average latest-20 history vectors, cosine scoring; evaluate on the same users and impressions.
- [ ] Initial test run with the frozen embeddings.
- [ ] Beyond-accuracy metrics on prediction exports: intra-list diversity, novelty (inverse popularity), catalogue coverage, exposure concentration.
- [ ] LLM-as-judge pipeline: sampled user subset, fixed published rubric (alignment, novelty, diversity, article quality), manual calibration spot-checks.

## Neural graph and trained models — Marcus

- [ ] LightGCN on the user-article click graph, reusing the shared BPR training loop (normalized neighbor aggregation is the delta over BPR-MF).
- [ ] Two-tower extension (user tower over history, article tower over text, trained on impression click labels) — only after the core comparison is stable.
- [ ] Export results, predictions, configurations, and model-method notes for both.

## Evaluation, testing, and statistical comparison — Yugan

Completed on behalf of the original infrastructure package (counts toward this share):

- [x] Public GitHub repository created.
- [x] MIND-small dataset download and parsing (`scripts/download_mind.py`, `mind_baselines/data.py`).
- [x] Eligibility filters (history >= 10; impressions with >= 1 clicked and >= 1 non-clicked candidate).
- [x] NDCG@5 and MRR evaluation (`mind_baselines/metrics.py`).
- [x] User-level paired bootstrap confidence intervals (`mind_baselines/analyze.py`).
- [x] Per-impression prediction exports for paired analysis.
- [x] Automated tests for parser, ranking, and metrics (`tests/`).
- [x] Popularity and TF-IDF baselines (`mind_baselines/models.py`).
- [x] Fixed 5,000-user smoke run on the supplied validation split (13 Oct, seed 4285): popularity 0.2301 vs TF-IDF 0.2929 NDCG@5; bootstrap CI entirely positive. Preliminary — validation split was consumed by this check.

Remaining:

- [ ] Paired statistical comparisons and bootstrap CIs across all model pairs.
- [ ] Subgroup analyses: history length (10–19 vs 20+), interest narrowness (category entropy of history), user connectivity in the click graph.
- [ ] History-length sensitivity: latest 5 / 10 / 20 articles on users with at least 20.
- [ ] Failure-case collection: where one representation succeeds and another fails; graph cold-start users; unexpected popularity wins. Feeds the error analysis.
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
