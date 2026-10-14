# MIND-small news recommendation

Reproducible baseline experiments for the CP4285 group project comparing
classical keyword representations with semantic text embeddings on MIND-small.

The current implementation includes:

- a training-click popularity baseline;
- a TF-IDF cosine-similarity baseline using article title and abstract;
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

1. Add frozen semantic embeddings with the same history averaging and cosine
   scoring procedure.
2. Compare histories of 5, 10, and 20 articles on users with at least 20.
3. Add the two-tower extension after the core comparison is stable.

