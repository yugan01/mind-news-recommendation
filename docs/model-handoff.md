# Model implementation and experiment handoff

Kevin supplies the shared infrastructure and frozen semantic model. Marcus implements BPR-MF and LightGCN, then runs the full shared development experiments and final validation. The two-tower extension follows the agreed project priorities. Current progress lives in `tasks.md`.

## Check setup without a full experiment

Install `requirements.txt` in the environment described in the README, then run:

```sh
python -m unittest discover -s tests
python -m scripts.smoke_semantic
python -m scripts.download_mind
python -m scripts.smoke_mind
```

The unit tests use synthetic data/vectors and do not load an encoder. `smoke_semantic` uses the real encoder on four synthetic articles and two impressions, checking normalized finite vectors, exports, unseen-user scoring, and identical predictions from cached vectors. `smoke_mind` reads the original training files through the real chronological development loader, then restricts fitting and evaluation to two impressions each and encodes only their referenced articles. It exercises both models, metric exports, and paired analysis. Neither smoke command opens supplied validation; neither produces reportable experiment results. Their caches and outputs are isolated under `outputs/smoke/`.

The shared EmbeddingGemma processor requires the image dependencies even though the vision/audio model encoders are disabled. On CPU-only machines, install matching CPU Torch/Torchvision wheels from the official PyTorch CPU index before installing the requirements if necessary:

```sh
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Add Marcus's models

Put each model in its own file under `src/models/`, implement `Model` in `src/contracts.py`, and add its class to `MODELS` in `src/config.py`. Keep its fixed hyperparameters there too. BPR-MF and LightGCN should share their BPR training code rather than duplicate it.

`fit` receives chronological training interactions, the article metadata, and user/article index mappings. `score` receives a user ID, centrally truncated history, and candidate IDs without click labels. Return one finite score per candidate in supplied order. Do not read behavior files independently or select a different evaluation cohort. Document unseen-user/article behavior, especially for graph/collaborative models; article metadata may include articles without fitting clicks. Add focused synthetic tests for each new model.

The semantic method is fixed: text-only `google/embeddinggemma-2`, title plus abstract, `SentenceSimilarity` prompting, 768-dimensional normalized article vectors, mean of latest-20 history vectors, and cosine scoring. Unseen users use their supplied history; empty/cancelling profiles tie at zero. CPU encoding uses float32, two threads, and batches of eight. It requires no click-label training or fine-tuning. Delete `outputs/cache/embeddinggemma-2.npz` if encoder/prompt settings change; otherwise cached vectors are reused and only new/changed article texts are encoded.

## Run actual experiments

Once the models and their tests are ready:

```sh
python -m src.run
python -m src.analyze
python -m src.beyond
```

These fit every enabled model on the full chronological training partition and evaluate the same 5,000-user development cohort. The initial semantic run encodes the full training article metadata corpus, unlike the smoke commands. Results appear in `outputs/dev/metrics.json`, per-model `predictions.csv` and `impression_metrics.csv`, and `comparisons.json`. NDCG@5 and MRR weight users equally; comparisons use paired user bootstrap intervals and reject mismatched examples. Rerunning replaces outputs, so retain any experiment results needed for analysis before rerunning.

After the team fixes all model and preprocessing decisions, obtain `MINDsmall_dev.zip` from https://huggingface.co/datasets/Recommenders/MIND and place its `news.tsv` and `behaviors.tsv` in `data/raw/validation/`. Then run:

```sh
python -m src.final
python -m src.beyond_final
```

This fits all enabled models on all supplied training interactions, evaluates all eligible validation users without sampling, and writes metrics, predictions, and paired comparisons to `outputs/final/`. Additional validation article metadata can be embedded without using its interaction labels for fitting. This final path has synthetic test coverage, but Kevin has not run it on actual validation data. Full semantic development experiments have likewise not been run by Kevin.

The `beyond` commands only post-process existing exports and cached embeddings. See `docs/beyond-accuracy.md` for metric definitions; the real-MIND smoke also exercises these metrics on its tiny sample. LLM judging is deferred and is not required by these commands.
