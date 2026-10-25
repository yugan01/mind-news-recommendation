# Two-Tower Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the gated task-learned two-tower recommender over frozen article text vectors, integrate it into the paired development experiment, and hand off verified outputs without changing the frozen core comparison.

**Architecture:** Extract the existing EmbeddingGemma encoding/cache behavior into one shared helper, preserving the semantic baseline exactly. The two-tower model mean-pools frozen history vectors, learns separate user/article projection networks with the existing impression-conditioned pair sampler and pairwise BPR ranking loss, and scores candidates by inner product in the learned 64-dimensional space.

**Tech Stack:** Python 3.11, NumPy 2.x, PyTorch 2.x, sentence-transformers 6.x, `unittest`, Ruff, mypy.

**Spec:** `docs/superpowers/specs/2026-10-25-marcus-models-design.md`

## Entry gate

Do not start this plan unless `docs/marcus-model-handoff.md` contains `Two-tower gate: open` backed by all Phase A tests, smoke, full development run, paired-output audit, and deterministic-output check. If the gate is closed, this plan remains unexecuted; do not weaken Phase A settings to open it.

## Global Constraints

- Preserve every constraint and fixed setting in `2026-10-25-bpr-lightgcn-implementation.md`.
- Do not fine-tune EmbeddingGemma; article text vectors remain frozen.
- Preserve the semantic baseline's article ordering, text construction, cache validation, normalized vectors, and scores byte-for-byte on the semantic tests.
- Use the existing `PairPool`, `sample_pairs`, and `pairwise_bpr_loss`; do not create another negative sampler or ranking-loss implementation.
- Train from each pair's own fitting-impression history via `PairPool.example_indices`; never replace it with a static user-ID embedding.
- Use title plus abstract through the shared cache helper and up to the existing latest 20 history articles.
- Use a separate `Linear(768, 64) -> ReLU -> Linear(64, 64)` network for each tower.
- Use inner-product scoring, Adam with no global weight decay, explicit parameter L2, seed 4285, learning rate 0.001, and 20 epochs.
- Empty histories receive all-zero scores; articles with valid cached text vectors support interaction cold start.
- Do not change Phase A hyperparameters or rerun final validation.
- Development outputs are not final results.
- Phase A remains the 26 October 2026 priority; two-tower proceeds only after its gate and must feed the 9 November 2026 analysis/deliverable handoff for the 11 November 2026 presentation.

## File map

| File | Responsibility |
|---|---|
| `src/text_embeddings.py` | Shared frozen encoder and validated ID/text/vector cache. |
| `src/models/semantic.py` | Thin semantic scorer consuming the shared helper. |
| `src/models/two_tower.py` | Feature alignment, two projection towers, deterministic training, fit, and score. |
| `src/config.py` | Fixed two-tower settings and final development model registration. |
| `scripts/smoke_mind.py` | Include two-tower with reduced smoke epochs only. |
| `tests/test_semantic.py` | Prove semantic behavior and cache compatibility survive extraction. |
| `tests/test_two_tower.py` | Exact features, gradient direction, determinism, cold-start, and contract tests. |
| `tests/test_infrastructure.py` | Five-model metadata/output schema checks. |
| `docs/marcus-model-handoff.md` | Two-tower method, runtime, development results, limitations, and new artifact map. |
| `tasks.md` | Evidence-backed status for the strongly recommended extension. |

---

### Task 1: Extract one shared frozen-text cache without changing semantic scores

**Files:**
- Create: `src/text_embeddings.py`
- Modify: `src/models/semantic.py`
- Modify: `tests/test_semantic.py`

**Interfaces:**
- Consumes: `Mapping[str, Article]`, cache path, and encoder callback.
- Produces:
  - `encode_articles(texts: list[str]) -> NDArray[np.float32]`
  - `load_article_vectors(articles: Mapping[str, Article], cache_path: Path, encoder: Encoder) -> tuple[dict[str, int], NDArray[np.float32]]`
- Preserves: `src.models.semantic.encode_articles` as an imported patch point used by current tests.

- [ ] **Step 1: Add a failing direct cache-helper test**

In `tests/test_semantic.py`, add imports for `DataError` and `load_article_vectors`, then add these tests:

```python
def test_shared_cache_returns_sorted_alignment_and_reuses_vectors(self) -> None:
    cache = Path(self.temp.name) / "shared.npz"
    calls: list[list[str]] = []

    def encoder(texts: list[str]) -> NDArray[np.float32]:
        calls.append(texts)
        return self.vectors[: len(texts)]

    indices, vectors = load_article_vectors(self.articles, cache, encoder)
    repeated_indices, repeated_vectors = load_article_vectors(
        dict(reversed(tuple(self.articles.items()))),
        cache,
        lambda texts: (_ for _ in ()).throw(AssertionError(texts)),
    )
    self.assertEqual(indices, {"A": 0, "B": 1, "C": 2})
    self.assertEqual(indices, repeated_indices)
    self.assertEqual(
        calls,
        [["First title First abstract", "Second title", "Third title Third abstract"]],
    )
    np.testing.assert_array_equal(vectors, repeated_vectors)

def test_shared_cache_rejects_wrong_vector_shape(self) -> None:
    cache = Path(self.temp.name) / "broken.npz"
    np.savez(
        cache,
        article_ids=np.asarray(["A"]),
        texts=np.asarray(["First title First abstract"]),
        vectors=np.zeros((1, 2), dtype=np.float32),
    )
    with self.assertRaisesRegex(DataError, "Invalid semantic cache shape"):
        load_article_vectors(self.articles, cache, lambda texts: self.vectors)
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `.venv/bin/python -m unittest tests.test_semantic.SemanticTests.test_shared_cache_returns_sorted_alignment_and_reuses_vectors tests.test_semantic.SemanticTests.test_shared_cache_rejects_wrong_vector_shape -v`  
Expected: `ImportError` because `src.text_embeddings` does not exist.

- [ ] **Step 3: Move the encoder and cache logic into the shared module**

Create `src/text_embeddings.py`:

```python
"""Frozen text encoding and validated article-vector cache."""

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from numpy.typing import NDArray

from src.contracts import Article
from src.data import DataError

Encoder = Callable[[list[str]], NDArray[np.float32]]


def encode_articles(texts: list[str]) -> NDArray[np.float32]:
    import torch
    from sentence_transformers import SentenceTransformer

    from src import config

    torch.set_num_threads(config.SEMANTIC_CPU_THREADS)
    encoder = SentenceTransformer(
        config.SEMANTIC_ENCODER,
        device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    return np.asarray(
        encoder.encode(
            texts,
            prompt_name="SentenceSimilarity",
            batch_size=config.SEMANTIC_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )


def load_article_vectors(
    articles: Mapping[str, Article], cache_path: Path, encoder: Encoder
) -> tuple[dict[str, int], NDArray[np.float32]]:
    article_ids = sorted(articles)
    texts = [articles[article_id].text for article_id in article_ids]
    article_indices = {article_id: index for index, article_id in enumerate(article_ids)}
    vectors = np.empty((len(article_ids), 768), dtype=np.float32)
    cached: dict[str, int] = {}
    cached_texts: list[str] = []
    cached_vectors: NDArray[np.float32] = np.empty((0, 768), dtype=np.float32)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if set(cache.files) != {"article_ids", "texts", "vectors"}:
                raise DataError(f"Invalid semantic cache fields: {cache_path}")
            cached = {article: index for index, article in enumerate(cache["article_ids"])}
            cached_texts = cache["texts"].tolist()
            cached_vectors = np.asarray(cache["vectors"], dtype=np.float32)
            if (
                cached_vectors.shape != (len(cached), 768)
                or len(cached_texts) != len(cached)
                or not np.isfinite(cached_vectors).all()
            ):
                raise DataError(f"Invalid semantic cache shape: {cache_path}")
    missing: list[int] = []
    for index, article_id in enumerate(article_ids):
        cached_index = cached.get(article_id)
        if cached_index is not None and cached_texts[cached_index] == texts[index]:
            vectors[index] = cached_vectors[cached_index]
        else:
            missing.append(index)
    if missing:
        encoded = np.asarray(encoder([texts[index] for index in missing]), dtype=np.float32)
        if encoded.shape != (len(missing), 768) or not np.isfinite(encoded).all():
            raise DataError("Text encoder returned invalid article vectors")
        vectors[missing] = encoded
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            article_ids=np.asarray(article_ids),
            texts=np.asarray(texts),
            vectors=vectors,
        )
    return article_indices, vectors
```

- [ ] **Step 4: Make `SemanticModel.fit` delegate without changing its patch point**

Replace the cache implementation in `src/models/semantic.py` with imports and delegation:

```python
from src.text_embeddings import encode_articles, load_article_vectors


def fit(self, data: TrainingData) -> None:
    from src import config

    self.article_indices, self.vectors = load_article_vectors(
        data.articles, config.SEMANTIC_CACHE, encode_articles
    )
```

Keep `SemanticModel.score` unchanged.

- [ ] **Step 5: Run semantic regression, lint, and types**

Run:

```bash
.venv/bin/python -m unittest tests.test_semantic -v
.venv/bin/ruff check src/text_embeddings.py src/models/semantic.py tests/test_semantic.py
.venv/bin/mypy src/text_embeddings.py src/models/semantic.py tests/test_semantic.py
```

Expected: every existing semantic score/cache test and the new direct helper test passes.

- [ ] **Step 6: Commit the behavior-preserving extraction**

```bash
git add src/text_embeddings.py src/models/semantic.py tests/test_semantic.py
git commit -m "refactor: share frozen article vector cache"
```

---

### Task 2: Implement deterministic two-tower training and scoring

**Files:**
- Modify: `src/config.py`
- Create: `src/models/two_tower.py`
- Create: `tests/test_two_tower.py`

**Interfaces:**
- Consumes: `build_pair_pool`, `sample_pairs`, `pairwise_bpr_loss`, `training_device`, and `load_article_vectors`.
- Produces:
  - `history_features(data: TrainingData, vectors: NDArray[np.float32], vector_indices: Mapping[str, int]) -> NDArray[np.float32]`
  - `TwoTowerNetwork(input_dimension=768, hidden=64, output=64, seed=4285)`
  - `TwoTowerModel.name == "two_tower"`, `fit`, `score`, and `run_metadata`.

- [ ] **Step 1: Add fixed settings and a failing feature-alignment test**

Add to `src/config.py`:

```python
TWO_TOWER_HIDDEN = 64
TWO_TOWER_OUTPUT = 64
TWO_TOWER_EPOCHS = 20
TWO_TOWER_LEARNING_RATE = 0.001
TWO_TOWER_L2 = 0.0001
```

Create `tests/test_two_tower.py` with a four-article fixture and test:

```python
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import torch
from numpy.typing import NDArray

from src import config
from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError
from src.models.two_tower import TwoTowerModel, TwoTowerNetwork, history_features


def tower_fixture() -> tuple[TrainingData, NDArray[np.float32]]:
    articles = {
        article: Article(article, "topic", article, "")
        for article in ("A", "B", "C", "D")
    }
    examples = (
        Example(
            RankingInput("I1", "U1", datetime(2019, 1, 1), ("A", "B"), 2, ("B", "C")),
            Targets((1, 0)),
        ),
    )
    data = TrainingData(
        articles,
        examples,
        {"U1": 0},
        {"A": 0, "B": 1, "C": 2, "D": 3},
    )
    vectors = np.zeros((4, 768), dtype=np.float32)
    vectors[0, 0] = 1.0
    vectors[1, 1] = 1.0
    vectors[2, 2] = 1.0
    vectors[3, 3] = 1.0
    return data, vectors


class TwoTowerTests(unittest.TestCase):
    def test_history_features_follow_each_example_not_static_user_id(self) -> None:
        data, vectors = tower_fixture()
        changed = replace(data.examples[0], inputs=replace(data.examples[0].inputs, history=("C",)))
        data = replace(data, examples=(data.examples[0], changed))
        features = history_features(data, vectors, {"A": 0, "B": 1, "C": 2, "D": 3})
        np.testing.assert_array_equal(features[0, :3], [0.5, 0.5, 0.0])
        np.testing.assert_array_equal(features[1, :3], [0.0, 0.0, 1.0])
```

- [ ] **Step 2: Run the feature test and verify import failure**

Run: `.venv/bin/python -m unittest tests.test_two_tower.TwoTowerTests.test_history_features_follow_each_example_not_static_user_id -v`  
Expected: `ImportError` because `src.models.two_tower` does not exist.

- [ ] **Step 3: Implement feature alignment and the network**

Create the first part of `src/models/two_tower.py`:

```python
"""Task-learned two-tower recommender over frozen text vectors."""

from typing import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from src import config
from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import (
    build_pair_pool,
    pairwise_bpr_loss,
    sample_pairs,
    training_device,
)
from src.text_embeddings import encode_articles, load_article_vectors


def history_features(
    data: TrainingData,
    vectors: NDArray[np.float32],
    vector_indices: Mapping[str, int],
) -> NDArray[np.float32]:
    features = np.zeros((len(data.examples), vectors.shape[1]), dtype=np.float32)
    for row, example in enumerate(data.examples):
        try:
            indices = [vector_indices[article] for article in example.inputs.history]
        except KeyError as error:
            raise DataError(f"Two-tower history is missing article {error.args[0]}") from error
        if indices:
            features[row] = vectors[indices].mean(axis=0)
    return features


class TwoTowerNetwork(nn.Module):
    def __init__(self, input_dimension: int, hidden: int, output: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.user_tower = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )
        self.article_tower = nn.Sequential(
            nn.Linear(input_dimension, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output),
        )

    def user_vectors(self, features: Tensor) -> Tensor:
        return self.user_tower(features)

    def article_vectors(self, features: Tensor) -> Tensor:
        return self.article_tower(features)
```

- [ ] **Step 4: Run the feature test and verify it passes**

Run the command from Step 2.  
Expected: pass.

- [ ] **Step 5: Add failing fit/determinism/cold-start tests**

Append to `TwoTowerTests`:

```python
def test_fit_is_deterministic_and_uses_text_for_unseen_user_and_article(self) -> None:
    data, vectors = tower_fixture()
    temp = tempfile.TemporaryDirectory()
    self.addCleanup(temp.cleanup)
    cache = Path(temp.name) / "vectors.npz"
    with (
        patch.object(config, "SEMANTIC_CACHE", cache),
        patch.object(config, "TWO_TOWER_EPOCHS", 5),
        patch("src.models.two_tower.encode_articles", return_value=vectors),
    ):
        left = TwoTowerModel()
        right = TwoTowerModel()
        left.fit(data)
        right.fit(data)
    unseen = replace(
        data.examples[0].inputs,
        user_id="NEW",
        history=("D",),
        candidate_ids=("D", "B"),
    )
    np.testing.assert_allclose(left.score(unseen), right.score(unseen), rtol=0, atol=1e-7)
    self.assertTrue(np.isfinite(left.score(unseen)).all())

    class IdentityTower(torch.nn.Module):
        def user_vectors(self, features: torch.Tensor) -> torch.Tensor:
            return features[:, :4]

        def article_vectors(self, features: torch.Tensor) -> torch.Tensor:
            return features[:, :4]

    left.network = cast(TwoTowerNetwork, IdentityTower())
    np.testing.assert_array_equal(left.score(unseen), [1.0, 0.0])
    empty = replace(unseen, history=())
    np.testing.assert_array_equal(left.score(empty), np.zeros(2))

def test_score_before_fit_fails(self) -> None:
    data, _ = tower_fixture()
    model = TwoTowerModel()
    with self.assertRaisesRegex(DataError, "before fit"):
        model.score(data.examples[0].inputs)
```

- [ ] **Step 6: Implement training, fitted state, and scoring**

Append to `src/models/two_tower.py`:

```python
class TwoTowerModel:
    def __init__(self) -> None:
        self.network: TwoTowerNetwork | None = None
        self.vector_indices: dict[str, int] = {}
        self.vectors: NDArray[np.float32] = np.empty((0, 768), dtype=np.float32)
        self.run_metadata: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return "two_tower"

    def fit(self, data: TrainingData) -> None:
        self.network = None
        self.vector_indices = {}
        self.vectors = np.empty((0, 768), dtype=np.float32)
        self.run_metadata = {}
        self.vector_indices, self.vectors = load_article_vectors(
            data.articles, config.SEMANTIC_CACHE, encode_articles
        )
        pool = build_pair_pool(data)
        article_features = np.empty((len(data.article_indices), 768), dtype=np.float32)
        for article, index in data.article_indices.items():
            article_features[index] = self.vectors[self.vector_indices[article]]
        histories = history_features(data, self.vectors, self.vector_indices)
        device = training_device()
        torch.manual_seed(config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.SEED)
        torch.use_deterministic_algorithms(True)
        network = TwoTowerNetwork(
            768,
            config.TWO_TOWER_HIDDEN,
            config.TWO_TOWER_OUTPUT,
            config.SEED,
        ).to(device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.TWO_TOWER_LEARNING_RATE, weight_decay=0.0
        )
        history_tensor = torch.as_tensor(histories, device=device)
        article_tensor = torch.as_tensor(article_features, device=device)
        losses: list[float] = []
        for epoch in range(config.TWO_TOWER_EPOCHS):
            pairs = sample_pairs(pool, epoch, config.SEED, device)
            optimizer.zero_grad(set_to_none=True)
            users = network.user_vectors(history_tensor[pairs.example_indices])
            positives = network.article_vectors(article_tensor[pairs.positive_indices])
            negatives = network.article_vectors(article_tensor[pairs.negative_indices])
            ranking = pairwise_bpr_loss(users, positives, negatives)
            penalty = torch.stack(
                [parameter.square().mean() for parameter in network.parameters()]
            ).sum()
            loss = ranking + config.TWO_TOWER_L2 * penalty
            if not torch.isfinite(loss):
                raise DataError(f"Two-tower produced non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        network.eval()
        self.network = network.to("cpu")
        self.run_metadata = {
            "device": str(device),
            "epochs": len(losses),
            "final_loss": losses[-1],
            "included_impressions": pool.included_impressions,
            "skipped_impressions": pool.skipped_impressions,
        }

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if self.network is None:
            raise DataError("Two-tower score called before fit")
        if not inputs.history:
            return np.zeros(len(inputs.candidate_ids), dtype=np.float64)
        try:
            history = [self.vector_indices[article] for article in inputs.history]
            candidates = [self.vector_indices[article] for article in inputs.candidate_ids]
        except KeyError as error:
            raise DataError(f"Two-tower score is missing article {error.args[0]}") from error
        history_feature = torch.as_tensor(self.vectors[history].mean(axis=0)[None, :])
        candidate_features = torch.as_tensor(self.vectors[candidates])
        with torch.no_grad():
            user = self.network.user_vectors(history_feature)[0]
            articles = self.network.article_vectors(candidate_features)
            scores = (articles * user).sum(dim=1).numpy().astype(np.float64)
        if not np.isfinite(scores).all():
            raise DataError(f"Two-tower produced non-finite scores for {inputs.impression_id}")
        return scores
```

- [ ] **Step 7: Run all two-tower and repository tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_two_tower -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src/models/two_tower.py tests/test_two_tower.py src/config.py
.venv/bin/mypy src/models/two_tower.py tests/test_two_tower.py src/config.py
```

Expected: all commands pass and repeated fits produce equal scores.

- [ ] **Step 8: Commit the model**

```bash
git add src/config.py src/models/two_tower.py tests/test_two_tower.py
git commit -m "feat: add task-learned two-tower model"
```

---

### Task 3: Integrate and smoke-test the fifth model

**Files:**
- Modify: `src/config.py`
- Modify: `scripts/smoke_mind.py`
- Modify: `tests/test_infrastructure.py`
- Modify: `tests/test_two_tower.py`

**Interfaces:**
- Consumes: `TwoTowerModel` and the four-model runner.
- Produces: registered order ending in `two_tower`, smoke outputs for five models, and ten pairwise comparisons.

- [ ] **Step 1: Add failing registration and five-model smoke assertions**

Add to `tests/test_two_tower.py`:

```python
def test_two_tower_is_registered_last(self) -> None:
    self.assertEqual(tuple(model().name for model in config.MODELS)[-1], "two_tower")
```

Extend the smoke artifact test in `tests/test_infrastructure.py` to expect the five names:

```python
("popularity", "semantic", "bpr_mf", "lightgcn", "two_tower")
```

Extend the `metrics.json` assertion with:

```python
self.assertEqual(summary["settings"]["two_tower_hidden"], config.TWO_TOWER_HIDDEN)
self.assertEqual(summary["settings"]["two_tower_output"], config.TWO_TOWER_OUTPUT)
self.assertEqual(summary["settings"]["two_tower_epochs"], config.TWO_TOWER_EPOCHS)
```

- [ ] **Step 2: Run the focused tests and verify registration failure**

Run: `.venv/bin/python -m unittest tests.test_two_tower.TwoTowerTests.test_two_tower_is_registered_last -v`  
Expected: assertion failure because `two_tower` is not registered.

- [ ] **Step 3: Register the model and bound smoke cost**

Import `TwoTowerModel` in `src/config.py` and append it to `MODELS` after `LightGCNModel`.

In `scripts/smoke_mind.py`, import `TwoTowerModel`, include it in the patched model tuple, and add:

```python
TWO_TOWER_EPOCHS=2,
TWO_TOWER_HIDDEN=8,
TWO_TOWER_OUTPUT=8,
```

In `src/run.py`, extend the existing `settings` dictionary with:

```python
settings.update(
    {
        "two_tower_hidden": config.TWO_TOWER_HIDDEN,
        "two_tower_output": config.TWO_TOWER_OUTPUT,
        "two_tower_epochs": config.TWO_TOWER_EPOCHS,
        "two_tower_learning_rate": config.TWO_TOWER_LEARNING_RATE,
        "two_tower_l2": config.TWO_TOWER_L2,
    }
)
```

- [ ] **Step 4: Run quality gates and the real-MIND smoke**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check .
.venv/bin/mypy src tests
/usr/bin/time -l .venv/bin/python -m scripts.smoke_mind 2>&1 | tee /tmp/cp4285-two-tower-smoke.log
```

Expected: all gates pass; smoke outputs contain five models and `comparisons.json` contains ten pairwise combinations.

- [ ] **Step 5: Audit cache reuse and paired identities**

Run:

```bash
.venv/bin/python - <<'PY'
import csv
import json
from itertools import combinations
from pathlib import Path

root = Path("outputs/smoke/mind")
names = ("popularity", "semantic", "bpr_mf", "lightgcn", "two_tower")
identities = []
for name in names:
    with (root / name / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    identities.append([(r["impression_id"], r["candidate_index"], r["article_id"], r["label"]) for r in rows])
assert identities[1:] == identities[:-1]
with (root / "comparisons.json").open() as handle:
    comparisons = json.load(handle)
assert len(comparisons) == len(tuple(combinations(names, 2))) == 10
print("five-model smoke audit passed")
PY
```

Expected: `five-model smoke audit passed`; the direct shared-cache test from Task 1 proves unchanged article text does not invoke the encoder twice.

- [ ] **Step 6: Commit integration**

```bash
git add src/config.py src/run.py scripts/smoke_mind.py tests/test_infrastructure.py tests/test_two_tower.py
git commit -m "feat: integrate two-tower experiment"
```

---

### Task 4: Run the five-model development comparison and handoff

**Files:**
- Modify: `docs/marcus-model-handoff.md`
- Modify: `tasks.md`
- Generated but ignored: `outputs/dev/**`

**Interfaces:**
- Consumes: frozen five-model development code.
- Produces: paired outputs for five models, ten comparisons, deterministic two-tower predictions, updated method notes, and an evidence-backed Phase B status.

- [ ] **Step 1: Record clean revision and rerun development**

Run:

```bash
git status --short
git rev-parse HEAD
/usr/bin/time -l .venv/bin/python -m src.run 2>&1 | tee /tmp/cp4285-five-model-development.log
.venv/bin/python -m src.analyze
.venv/bin/python -m src.beyond
```

Expected: clean pre-run source state; all five models finish; ten comparisons are written.

- [ ] **Step 2: Audit five-model identity, finiteness, ranks, and pair count**

Run the Task 7 paired-output audit from the core plan with:

```python
names = ("popularity", "semantic", "bpr_mf", "lightgcn", "two_tower")
```

Expected: identical impression/user/candidate/label rows across models, finite scores, rank permutations, ten exact comparison keys, and finite beyond-accuracy records for all five models.

- [ ] **Step 3: Prove two-tower determinism without rerunning other models**

Run:

```bash
cp outputs/dev/two_tower/predictions.csv /tmp/two-tower-first.csv
.venv/bin/python - <<'PY'
from pathlib import Path
from unittest.mock import patch

from src import config
from src.data import load_development
from src.models.two_tower import TwoTowerModel
from src.run import run_experiment

with patch.object(config, "MODELS", (TwoTowerModel,)):
    run_experiment(load_development(), Path("/tmp/cp4285-two-tower-repeat"))
PY
cmp /tmp/two-tower-first.csv /tmp/cp4285-two-tower-repeat/two_tower/predictions.csv
```

Expected: `cmp` exits `0`.

- [ ] **Step 4: Add observed two-tower method/results to the handoff**

Add a `## Two-tower extension` section to `docs/marcus-model-handoff.md` containing observed values for revision, device, epochs, final loss, runtime, peak resident memory, development NDCG@5, development MRR, and exact output paths. State explicitly:

```markdown
- Frozen EmbeddingGemma article vectors were inputs; the encoder was not fine-tuned.
- Each fitting pair used its own impression history, not a persistent user-ID embedding.
- Separate 768→64→64 user and article projections were optimized with the shared within-impression BPR pairs.
- Results are development-only and do not establish satisfaction or causal preference.
```

Extend `## AI-use record for Sood` to state that Codex also drafted the shared cache refactor, two-tower implementation, tests, and verification fixes, while Marcus remains responsible for review and final claims.

- [ ] **Step 5: Update tasks only after evidence passes**

Mark the two-tower task complete in `tasks.md` only after Steps 1–4 pass. Add the development command and `docs/marcus-model-handoff.md` as evidence. Do not mark final evaluation complete.

- [ ] **Step 6: Run the final extension gate**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check .
.venv/bin/mypy src tests
git diff --check
rg -n -i 'TBD|TODO|insert value|replace me|fill in' docs/marcus-model-handoff.md
git status --short
```

Expected: every quality command passes, placeholder scan is empty, only the handoff and task tracker are modified, and generated data/outputs remain ignored.

- [ ] **Step 7: Commit the extension handoff**

```bash
git add docs/marcus-model-handoff.md tasks.md
git commit -m "docs: hand off two-tower development results"
```

---

## Extension-plan completion audit

| Requirement | Evidence |
|---|---|
| Core gate honored | `Two-tower gate: open` with Phase A evidence before first Phase B commit |
| Frozen shared text features | semantic regression tests and shared-cache direct test |
| No encoder fine-tuning | encoder absent from optimizer parameters; handoff statement |
| Per-impression history tower | exact changing-history feature test and `example_indices` use |
| Separate tower architecture | network structure inspection in unit tests and implementation |
| Shared pair semantics | imports of `build_pair_pool`, `sample_pairs`, `pairwise_bpr_loss` |
| Deterministic training | repeated synthetic fits and byte-identical full-development predictions |
| Interaction cold start | unseen user/article text test with finite non-tied scores |
| Empty-history fallback | exact zero-score test |
| Five paired outputs | identity/rank/finiteness audit across all five models |
| Ten comparisons | exact `comparisons.json` combination count and keys |
| Beyond-accuracy outputs | finite five-model `beyond_accuracy.json` records |
| Truthful handoff | observed method/runtime/metrics plus development-only limitations |
| Final split preserved | no `src.final` invocation and final task remains open |

Any missing, contradictory, or indirect evidence keeps the extension incomplete.
