# MIND-small news recommendation

CP4285 project comparing frozen semantic embeddings, LightGCN, and task-learned representations for personalized news ranking, with popularity and BPR-MF baselines.

`project-summary.md` defines the agreed scope. `tasks.md` tracks progress and owners. Instructor feedback and course requirements are in `docs/`.

## Setup and development run

Use Python 3.11 or newer. Create a virtual environment and install `requirements.txt`.

```sh
python -m venv .venv
```

Activate the virtual environment: on Linux/macOS use `source .venv/bin/activate`; on Windows PowerShell use `.venv\Scripts\Activate.ps1`. Then run:

```sh
python -m pip install -r requirements.txt
python -m scripts.download_mind
python -m src.run
```

The download command retrieves the original-format training archive from [Recommenders/MIND](https://huggingface.co/datasets/Recommenders/MIND). It preserves full timestamps in `behaviors.tsv`. RecZoo's CSV only retains hour-of-day and is unsuitable for the chronological split.

All shared values and paths live in `src/config.py`:

- Latest 20% of training impressions form the chronological development holdout. Equal timestamps stay together.
- Evaluate a deterministic sample of 5,000 eligible development users, seed 4285. Sampling does not reduce the training partition.
- Require at least 10 available history articles and both clicked and non-clicked candidates. Eligibility is checked before capping history at the latest 20.
- Use title plus abstract for article text. Retain category and timestamps for analysis.

Popularity and frozen semantic embeddings are enabled. Outputs go to `outputs/dev/`: aggregate `metrics.json` and a directory per model containing `predictions.csv` and `impression_metrics.csv`. Rerunning replaces these outputs.

The semantic model uses `google/embeddinggemma-2` with only its 270M-parameter text encoder, the `SentenceSimilarity` prompt, and full 768-dimensional normalized article vectors. It averages history vectors and ranks by cosine similarity; empty or cancelling histories receive tied zero scores. It needs no learned user IDs or click-label training.

The first semantic run downloads the encoder and encodes the shared article corpus. CPU inference uses float32, two threads, and batches of eight. Article vectors are cached in `outputs/cache/embeddinggemma-2.npz`; later runs encode only new or changed article texts. Delete that cache if you change the encoder or its encoding settings. A complete `src.run` is an experiment, not a small functionality test; current Kevin work is limited to smoke verification before handing full runs to Marcus.

NDCG@5 and MRR are averaged within each user, then across users. Score ties preserve the supplied candidate order.

## Adding a model

Put each model in its own file under `src/models/`. Follow `Model` in `src/contracts.py`:

- `name` identifies the model's output directory.
- `fit(training)` receives the shared article corpus, chronological training examples, and deterministic user/article index mappings. Model implementations must not independently read behavior files or change the cohort.
- `score(inputs)` receives user/history/candidate information without evaluation labels. Return one finite score per candidate in the supplied order.

History limits are applied centrally to training and evaluation inputs. Training includes all pre-cutoff impressions, including those ineligible for evaluation. Models must document their unseen-user/article behavior; popularity assigns unseen articles zero clicks.

Add the model class to `MODELS` in `src/config.py`. Every enabled model is fitted and evaluated by the same run command. Article metadata is shared; held-out interaction labels are never fitting data. Final evaluation also supplies validation article metadata so text models can encode unseen candidates.

```sh
python -m src.analyze
```

This compares every enabled model pair on development predictions using 10,000 paired user-level bootstrap samples. It rejects differences in impression sets, users, candidate order, or labels. Differences are challenger minus baseline. Results go to `outputs/dev/comparisons.json`.

## Final evaluation

**Do not use supplied validation for development.** After the team freezes model and preprocessing decisions, obtain `MINDsmall_dev.zip` from the same source and place its original `news.tsv` and `behaviors.tsv` in `data/raw/validation/`.

```sh
python -m src.final
```

Run this once for the final comparison. It fits every enabled model on all supplied training impressions, evaluates all eligible validation users without sampling, and writes metrics, predictions, and paired comparisons to `outputs/final/`. Freezing decisions is a team checkpoint; there is no software freezing subsystem.

## Tests

Small real-model functionality check (downloads the encoder on first use, but embeds only four synthetic articles):

```sh
python -m scripts.smoke_semantic
```

This checks ranking/export plumbing, finite normalized vectors, and cache reuse, including a previously unseen user. Its synthetic outputs and separate vector cache live in `outputs/smoke/`; the metrics are not experiment results. It does not open MIND validation or encode the full article corpus.

Tiny check using actual MIND training data (download it first):

```sh
python -m scripts.smoke_mind
```

This runs two training and two chronological development impressions, embedding only their referenced articles. It checks the real data/model/export/paired-analysis path without a full experiment or validation access. Outputs are isolated in `outputs/smoke/mind/`.

See `docs/model-handoff.md` for Marcus's model integration, test commands, and full development/final experiment sequence.

Fast synthetic unit tests (no encoder download or loading):

```sh
python -m unittest discover -s tests
```
