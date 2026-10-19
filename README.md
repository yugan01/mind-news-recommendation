# MIND-small news recommendation

Reproducible experiments for the CP4285 group project: comparing embedding-based
recommenders — frozen semantic text embeddings, LightGCN over the click graph,
and a trained two-tower model — on which signal best personalizes news ranking,
for which users, and at what cost in diversity, novelty, and exposure.

Full project context lives in the repo root and `docs/`. The **master document
is `project-summary.md`** — it holds the canonical, team-wide agreed scope,
direction, and work breakdown, and takes precedence over every other document:

- `project-summary.md` — the master document: core question, model
  list, evaluation plan, course-requirements mapping, and work breakdown
  (start here).
- `tasks.md` — the live task list: completed work, remaining work, and each
  member's responsibilities.
- `docs/instructor-feedback.md` — the instructor feedback that motivated the
  revised direction.

The current implementation includes:

- a training-click popularity baseline;
- a TF-IDF cosine-similarity baseline (kept as a reference row);
- user-averaged NDCG@5 and MRR evaluation;
- per-impression prediction exports for paired statistical analysis; and
- synthetic tests for the parser, ranking, and metrics.

## Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Download MIND-small

```powershell
python scripts/download_mind.py
```

This downloads and extracts the RecZoo MIND-small mirror into `data/raw/`.
The dataset is intentionally excluded from Git.

## Run the baselines

Start with a deterministic 5,000-user smoke run:

```powershell
python -m mind_baselines.run `
  --train-dir data/raw/MIND_small_x1 `
  --eval-dir data/raw/MIND_small_x1 `
  --sample-users 5000 `
  --output-dir outputs/smoke
```

Run all eligible validation users by omitting `--sample-users`:

```powershell
python -m mind_baselines.run `
  --train-dir data/raw/MIND_small_x1 `
  --eval-dir data/raw/MIND_small_x1 `
  --output-dir outputs/validation
```

An eligible user has at least 10 articles in the supplied history. An eligible
impression contains at least one clicked and one non-clicked candidate. Each
user profile uses at most the latest 20 history articles.

Outputs include `metrics.json` and one prediction CSV per model. Candidate
scores and ranks are exported before evaluation labels are used.

Compare two models with a paired user-level bootstrap confidence interval:

```powershell
python -m mind_baselines.analyze `
  --baseline outputs/smoke/popularity_predictions.csv `
  --challenger outputs/smoke/tfidf_predictions.csv `
  --output outputs/smoke/paired_comparison.json
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Planned next steps

1. Carve a chronological development split from the training data; the supplied
   validation set is reserved for the final evaluation.
2. Add frozen semantic embeddings (`all-MiniLM-L6-v2`, cached article vectors,
   averaged history profiles, cosine scoring).
3. Add BPR-MF as the classical baseline and the shared BPR training loop.
4. Add LightGCN on the click graph, reusing that training loop.
5. Add beyond-accuracy metrics and the LLM-as-judge panel.
6. Add the two-tower extension after the core comparison is stable.

See `tasks.md` for the full task list and owners.

