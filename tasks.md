# Project Tasks

The live task list and project state tracker. Everything about current progress — what is done, in progress, and remaining, per workstream — lives here. Canonical, team-wide agreed scope and direction live in the master document, `project-summary.md`; this file defers to it.

Legend: `[x]` done · `[ ]` open · `[~]` in progress. Update statuses as work advances.

## Data and evaluation infrastructure — Kevin

- [ ] Chronological development/tuning split carved from the training data; all model development runs only on it. [MVP]
- [ ] Final evaluation: single validation run after model and preprocessing decisions are frozen. [MVP]
- [ ] Frozen semantic embeddings (`all-MiniLM-L6-v2`): encode title+abstract once per article, cache vectors, average latest-20 history vectors, cosine scoring; evaluate on the same users and impressions. [MVP]
- [ ] Initial test run with the frozen embeddings. [MVP]
- [ ] Beyond-accuracy metrics on prediction exports, standard formulations only: intra-list diversity (average pairwise 1 − cosine dissimilarity of article embeddings within each top-5), novelty (self-information −log2 p of training popularity, Vargas & Castells 2011), catalogue coverage, exposure concentration (Gini coefficient). [Extra]
- [ ] LLM-as-judge pipeline: sampled user subset, fixed published rubric (alignment, novelty, diversity, article quality), manual calibration spot-checks. [Extra]

## Neural graph and trained models — Marcus

- [ ] BPR-MF baseline; also establishes the shared BPR training loop (embedding tables, pair sampling, loss) that LightGCN reuses. [MVP]
- [ ] LightGCN on the user-article click graph, reusing the shared BPR training loop (normalized neighbor aggregation is the delta over BPR-MF). [MVP]
- [ ] Two-tower extension (user tower over history, article tower over text, trained on impression click labels) — after the core comparison is stable. [Strongly recommended]
- [ ] Export results, predictions, configurations, and model-method notes. [MVP]

## Evaluation, testing, and statistical comparison — Yugan

Completed on behalf of the original infrastructure package (counts toward this share):

- [x] Public GitHub repository created.
- [x] MIND-small dataset download and parsing (`scripts/download_mind.py`, `mind_baselines/data.py`).
- [x] Eligibility filters (history >= 10; impressions with >= 1 clicked and >= 1 non-clicked candidate).
- [x] NDCG@5 and MRR evaluation (`mind_baselines/metrics.py`).
- [x] User-level paired bootstrap confidence intervals (`mind_baselines/analyze.py`).
- [x] Per-impression prediction exports for paired analysis.
- [x] Automated tests for parser, ranking, and metrics (`tests/`).
- [x] Data pipeline plus a popularity baseline and an early TF-IDF prototype (prototype removed during the 19 Oct cleanup; BPR-MF is the sole classical baseline going forward).
- [x] Fixed 5,000-user smoke run (popularity vs the early TF-IDF prototype) confirming the pipeline works end to end; the run record was removed with the prototype.

Remaining:

- [ ] Paired statistical comparisons and bootstrap CIs across all model pairs. [MVP]
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
