# Marcus Models Design

**Status:** Proposed for Marcus's review  
**Date:** 25 October 2026  
**Owner:** Marcus  
**Implementation target:** BPR-MF and LightGCN by 26 October; two-tower only after the core gate passes

## 1. Purpose

Implement Marcus's share of the CP4285 news-recommendation project without changing the team's agreed experiment protocol. The required result is a controlled comparison between:

- training-click popularity;
- frozen semantic text embeddings;
- BPR matrix factorization; and
- LightGCN.

BPR-MF is both a classical baseline and the no-propagation ablation for LightGCN. A small two-tower model is a strongly recommended extension, but it must not delay or destabilize the required comparison.

This design follows `project-summary.md`, `tasks.md`, `AGENTS.md`, the current interfaces under `src/`, and the course treatment of BPR and LightGCN. It does not change dataset eligibility, the chronological development split, the final-validation protocol, history truncation, evaluation metrics, or user weighting.

## 2. Scope and gates

### Phase A — required MVP

1. Add a deterministic shared pair-construction and BPR-training implementation.
2. Add `BPRMFModel` using learned user and article embedding tables.
3. Add `LightGCNModel` using the same training objective and pairs, with normalized graph propagation as the intended model delta.
4. Register both models in the existing argument-free experiment commands.
5. Add unit, contract, determinism, cold-start, and real-MIND smoke tests.
6. Run the development experiment only; do not open the supplied validation split.
7. Export predictions, metrics, configuration values, and method notes for Yugan and Sood.

### Core gate

Phase A passes only when all of the following are true:

- every automated test passes;
- both models pass a tiny exact synthetic test and the real-MIND smoke path;
- repeated seeded fits produce identical scores within numerical tolerance;
- `python -m src.run` completes on development data and exports all four model outputs;
- paired comparison tooling accepts all four prediction files;
- method notes accurately describe training pairs, graph edges, hyperparameters, scoring, and cold-start behavior;
- no code or report text claims that development metrics are final results.

### Phase B — two-tower extension

Begin only after the core gate passes and there is enough time to finish and test the extension without changing the frozen core protocol. If the gate is not reached in time, leave two-tower explicitly unimplemented rather than shipping an unverified model.

## 3. Fixed experimental contract

All models continue to implement `src.contracts.Model`:

```python
class Model(Protocol):
    @property
    def name(self) -> str: ...
    def fit(self, data: TrainingData) -> None: ...
    def score(self, inputs: RankingInput) -> NDArray[np.float64]: ...
```

The following invariants are mandatory:

- `fit` receives only fitting data.
- `score` receives `RankingInput`, never held-out targets.
- returned scores are finite, one-dimensional, and aligned with `candidate_ids`.
- development uses the chronological training-data split and fixed 5,000-user cohort already implemented.
- final validation remains unopened until model and preprocessing choices are frozen.
- each model ranks the same candidates for the same impressions.
- BPR-MF and LightGCN use identical pair construction, random seed, embedding dimension, optimizer, learning rate, regularization, epoch count, and negative-sampling schedule.
- standard BPR-MF and LightGCN use inner-product scoring. The frozen semantic model uses cosine similarity over normalized text vectors. Project prose must say the models use vector similarity, not falsely claim that every model uses cosine.

## 4. Training evidence and leakage boundary

### Pairwise training examples

For every fitting impression that contains at least one displayed positive and one displayed negative:

- positive articles are displayed candidates with label `1`;
- eligible negatives are displayed candidates with label `0`;
- for each positive, sample exactly one negative from that same impression per epoch;
- use a local seeded generator derived from `SEED` and the epoch number;
- never sample from a held-out impression or from the development/final evaluation cohort.

Sampling displayed negatives keeps the BPR claim narrow: the model learns that a recorded click should outrank a displayed but unclicked alternative. It does not treat every unobserved article as a dislike.

Fitting impressions with only one label class contribute no BPR pair; this is expected because `TrainingData.examples` contains the complete fitting partition rather than only evaluation-eligible rows. The builder records the included and skipped impression counts. The sampler returns index tensors for `(user, positive_article, negative_article)`. It validates that all included IDs resolve through the supplied index mappings and that the complete fitting partition yields at least one pair.

### LightGCN training graph

Build one undirected bipartite graph from fitting interactions only. An edge exists for each distinct:

- user–article occurrence in a fitting impression's supplied history; or
- user–article displayed candidate with fitting label `1`.

Duplicate user–article edges are collapsed so repeated appearances do not silently become edge weights. Displayed negatives never become graph edges. Evaluation histories and labels never become graph edges.

The normalized adjacency uses the standard coefficient `1 / sqrt(degree(user) * degree(article))`. User and article nodes occupy disjoint contiguous index ranges. Zero-degree nodes remain zero after propagation and are tracked explicitly for cold-start handling.

## 5. Shared BPR optimization

Use PyTorch in the project's Python 3.11 environment. Do not implement manual NumPy gradients or introduce a general training framework.

The shared trainer owns:

- seed initialization;
- pair sampling;
- parameter initialization;
- optimizer construction;
- BPR loss calculation;
- epoch execution;
- finite-loss checks; and
- extraction of detached inference embeddings.

For each epoch, sample one within-impression negative per positive and take one full-batch optimization step. The MIND-small pair count makes this feasible, and it avoids recomputing a full LightGCN graph once per mini-batch. Graph propagation is computed exactly once per epoch. An out-of-memory condition is a failed feasibility gate that must be reported and resolved in the reviewed design; the implementation must not silently switch optimization semantics.

The objective is:

```text
mean(softplus(-(score(u, positive) - score(u, negative))))
+ L2 coefficient * sampled layer-0 embedding penalty
```

Regularization applies to layer-0 user and article embeddings, including for LightGCN, matching the intended ablation. The trainer rejects non-finite loss or embeddings immediately.

Initial fixed settings live in `src/config.py` and are included in exported run metadata:

- embedding dimension: 64;
- epochs: 20;
- negatives per positive per epoch: 1;
- learning rate: 0.001;
- L2 coefficient: 0.0001;
- optimizer: Adam;
- initialization: normal distribution with standard deviation 0.1;
- random seed: 4285;
- LightGCN propagation layers: 2;
- LightGCN layer combination: uniform mean of layers 0, 1, and 2.

These are one declared development configuration, not a hidden sweep. Any later change must be made in `src/config.py`, rerun for every learned model affected, and recorded in the method notes before final freeze.

## 6. BPR-MF model

`BPRMFModel` creates layer-0 user and article embedding tables and passes raw embeddings to the shared BPR objective. At inference it scores a candidate with the inner product between the resolved user vector and candidate article vector.

The model stores only detached inference arrays plus interaction masks after `fit`; it does not retain an optimizer or computation graph. Calling `fit` again resets all learned state.

## 7. LightGCN model

`LightGCNModel` owns the fixed normalized training graph. During an epoch it:

1. starts with trainable layer-0 user and article embeddings;
2. propagates once per layer through the normalized bipartite adjacency;
3. combines layer 0 and every propagated layer with a uniform mean; and
4. supplies the combined user and article embeddings to the same BPR loss.

There are no feature transforms, nonlinear activations, or learnable edge weights. The only intended representational difference from BPR-MF is normalized neighbor aggregation and layer combination.

At inference the model uses the detached combined embeddings from the final trained layer-0 parameters. It does not rebuild the graph from evaluation data.

## 8. Cold-start and scoring policy

The policy is identical for BPR-MF and LightGCN wherever the same state exists:

### User vector

- If the evaluation user has at least one positive fitting interaction, use the learned user embedding.
- Otherwise, average the final embeddings of history articles that had at least one positive fitting interaction.
- If no such history article exists, use the all-zero vector.

The history fallback uses only article embeddings learned from fitting interactions. It does not update parameters or add evaluation edges.

### Candidate vector and score

- A candidate with at least one positive fitting interaction uses its learned final article embedding.
- A candidate with no positive fitting interaction receives score `0.0`; its random layer-0 initialization is never exposed as evidence.
- If the user vector is all zero, every candidate receives `0.0` and the existing stable candidate order breaks ties deterministically.

This is a declared non-personalized fallback, not a claim that the graph model understands new articles. Failure analysis must separately inspect cold users and cold articles.

## 9. File boundaries

The implementation is intentionally small and repository-native:

- `src/models/bpr.py` — immutable training-pair data, deterministic pair builder/sampler, shared BPR loss and training loop.
- `src/models/bpr_mf.py` — BPR-MF state, fit, cold-start resolution, and scoring.
- `src/models/lightgcn.py` — graph construction, normalized propagation, LightGCN fit, and scoring.
- `src/models/common.py` — only genuinely shared fitted-state and cold-start helpers used by both models; omit this file if the abstraction would merely hide two short methods.
- `src/config.py` — fixed training settings and model registration.
- `requirements.txt` — add `torch>=2.5,<3`, compatible with Python 3.11 and the existing sentence-transformers stack.
- `requirements-dev.txt` — add `ruff>=0.8,<1` and `mypy>=1.13,<3` so the declared quality gates are installable rather than assumed.
- `src/run.py` — include learned-model settings in `metrics.json`; do not add command-line configuration.
- `scripts/smoke_mind.py` — include both collaborative models and select two fitting impressions that each contain positive and negative displayed candidates, while keeping the real-data smoke tiny.
- `tests/test_bpr.py` — sampler, loss, exact update, determinism, reset, and cold-start behavior.
- `tests/test_lightgcn.py` — exact normalized propagation, no-propagation equivalence, graph leakage boundary, determinism, and cold-start behavior.
- `tests/test_infrastructure.py` — only cross-model contract or output-metadata assertions.
- `docs/marcus-model-handoff.md` — exact method, configuration, commands, runtime observations, development outputs, limitations, and downstream file locations.
- `tasks.md` — statuses updated only after their evidence exists.

For the optional extension:

- `src/models/two_tower.py` — small trainable user and article projections over cached frozen text vectors.
- `tests/test_two_tower.py` — feature alignment, pairwise training, scoring, determinism, and text cold-start tests.
- `src/text_embeddings.py` — created only when Phase B begins by extracting the existing encoder/cache behavior from `src/models/semantic.py`; both semantic and two-tower models then consume the same helper, with cache compatibility and all semantic tests preserved.

## 10. Two-tower design

The two-tower model consumes the same frozen article text vectors used by the semantic baseline; it does not fine-tune EmbeddingGemma. The article tower projects each cached article vector into a 64-dimensional recommendation space. The user tower mean-pools up to the latest 20 cached history vectors, then applies a separate projection into the same space. Each projection is a small `Linear(768, 64) -> ReLU -> Linear(64, 64)` network.

Train with the same impression-derived pair builder and BPR loss semantics. The two-tower model naturally supports users and articles unseen in the interaction graph when their text/history vectors exist. Missing or empty text/history resolves to a zero vector and is reported as a fallback.

Phase B starts by extracting the shared text-vector helper described above, then adds its own fixed settings to `src/config.py`, tests first on synthetic cached vectors, runs the real-MIND smoke, and finally reruns the development experiment for every registered model. It does not alter Phase A hyperparameters after the core gate.

## 11. Failure behavior

Raise `DataError` or a model-specific `ValueError` with actionable context when:

- training data contains no usable BPR pairs;
- an indexed ID is missing or mappings are inconsistent;
- graph degrees or normalization produce non-finite values;
- loss or learned embeddings become non-finite;
- a score request occurs before fitting;
- a cached feature matrix is misaligned with article indices; or
- a dependency/device required for training is unavailable.

Do not silently skip malformed examples, resample from evaluation data, or fall back from LightGCN to BPR-MF. Device selection must be recorded. CUDA is used when available; CPU is the compatibility fallback. MPS is excluded from the sparse graph path for this project so identical code paths are used on Marcus's machine and CPU-only environments.

## 12. Verification strategy

### Exact synthetic tests

- Pair builder: a two-impression fixture yields the exact expected positive and eligible-negative index sets.
- Sampler: the same seed and epoch yields identical triples; a different epoch changes eligible choices without crossing impression boundaries.
- Loss: fixed one-dimensional embeddings match a hand-computed BPR loss.
- BPR update: one optimization step increases the positive-minus-negative margin on a tiny fixture.
- Graph: a tiny bipartite graph produces exact degrees and normalized coefficients.
- Propagation: one LightGCN layer matches a hand-computed neighbor-weighted result.
- Ablation: LightGCN with zero propagation layers yields the same embedding definition and scores as BPR-MF from identical layer-0 weights.
- Leakage: changing held-out examples does not change fitted pairs, graph edges, or learned scores.
- Cold start: seen/unseen users, partly known history, interaction-unseen candidates, and empty fallback all match the declared policy.
- Refit: a second `fit` starts cleanly and produces the seeded result.

### Repository gates

Run, in order:

1. `python3.11 -m venv .venv`
2. `.venv/bin/pip install -r requirements.txt -r requirements-dev.txt`
3. `.venv/bin/python -m unittest discover -s tests -v`
4. `.venv/bin/ruff check .`
5. `.venv/bin/mypy src tests`
6. `.venv/bin/python -m scripts.smoke_mind`
7. `.venv/bin/python -m src.run`
8. `.venv/bin/python -m src.analyze`

The unit, lint, type, and smoke gates must pass before a full development run. `src.analyze` runs only after `src.run` has produced the expected prediction files. The supplied validation directory must remain unopened during all design and development work.

### Output audit

For every registered model verify:

- `predictions.csv` and `impression_metrics.csv` exist and are nonempty;
- impression IDs, candidate IDs, labels, and candidate order match exactly across models;
- scores are finite and ranks form a permutation per impression;
- `metrics.json` records all fixed model settings and the actual training device;
- paired comparisons accept every model pair;
- development metrics are labeled development-only; and
- method notes contain no unsupported causal or satisfaction claim.

## 13. Handoff and truthful reporting

Marcus hands Yugan and Sood:

- committed model and test code;
- the exact development command and environment;
- per-model predictions and impression metrics;
- paired-analysis-compatible filenames;
- training/runtime observations;
- method notes defining evidence, sampling, graph construction, scoring, and cold starts;
- a clear list of completed, skipped, and failed experiments.

The contribution statement must distinguish team and AI work. A truthful draft is:

> Marcus owned the BPR-MF, shared pairwise-training design, LightGCN implementation, experiment execution, and model-method handoff. Codex helped inspect the shared repository, propose and audit the architecture and implementation plan, draft code and tests, and diagnose verification failures. Marcus reviewed the design and code, ran or supervised the reported experiments, made final project decisions, and is responsible for the submitted claims.

Marcus must edit that statement to match what actually happens. Sood owns final report editing and submission coordination; Marcus reviews and uploads or otherwise submits only through the team's agreed process. No agent submits anything.

Before Marcus uploads any assessed artifact, the team must scan both visible content and file metadata for Marcus's name and matriculation number, then reconcile that anonymity requirement with the required contribution statement using the submission brief's permitted identifier format. This design does not assume that a public Git commit is itself the assessed anonymous artifact.

## 14. Deadline sequence

### 25 October

- approve this design;
- write and audit the implementation plan;
- implement and test pair construction plus BPR-MF;
- implement and exact-test LightGCN propagation.

### 26 October

- finish LightGCN integration;
- pass repository gates and real-MIND smoke;
- run the development experiment;
- audit paired outputs;
- write the Marcus handoff and update `tasks.md` from evidence;
- decide whether the two-tower gate is open.

### 27 October–9 November

- implement two-tower only if the core gate passed and the team still wants it;
- freeze choices before the one final-validation run;
- provide outputs and method notes to Yugan for CIs, subgroups, failures, and tests;
- answer Sood's method and limitation questions without changing frozen results.

### 11 November

- present the defensible comparison and limitations; do not overstate development findings or click-prediction evidence.

## 15. Non-goals

- no TF-IDF model;
- no LLM-as-judge work in Marcus's implementation package;
- no general experiment framework, CLI, manifest, or configuration sweep system;
- no changes to cohort eligibility, split logic, metrics, user weighting, or final-evaluation timing;
- no arbitrary unobserved-item negatives when displayed negatives exist;
- no evaluation labels in model scoring;
- no final-validation run before the team freeze;
- no submission or external posting by an agent.
