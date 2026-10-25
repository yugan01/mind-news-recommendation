# BPR-MF and LightGCN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic BPR-MF and LightGCN recommenders to the shared MIND experiment, prove that graph propagation is the controlled model delta, and produce development-only outputs and handoff notes.

**Architecture:** `src/models/bpr.py` owns impression-conditioned pair construction, deterministic sampling, the BPR objective, and the shared optimizer. `BPRMFModel` supplies raw embedding tables; `LightGCNModel` supplies embeddings after fixed normalized neighbor propagation. Both models use one collaborative scoring helper with explicit trained-evidence masks, and both plug into the existing label-free `Model.fit/score` contract.

**Tech Stack:** Python 3.11, NumPy 2.x, PyTorch 2.x, `unittest`, Ruff, mypy, existing MIND-small pipeline.

**Spec:** `docs/superpowers/specs/2026-10-25-marcus-models-design.md`

## Global Constraints

- Develop under Python 3.11; the host Python 3.14 environment is not the project runtime.
- Keep commands argument-free and fixed experiment settings in `src/config.py`.
- Never open `data/raw/validation` during these tasks; only `python -m src.final` may open it after the team freeze.
- Preserve `MIN_HISTORY = 10`, `MAX_HISTORY = 20`, `DEV_USERS = 5000`, `DEV_FRACTION = 0.2`, and `SEED = 4285`.
- Use only fitting-impression labels to construct BPR pairs; sample negatives from displayed unclicked candidates in the same impression.
- Build the LightGCN graph only from fitting histories and fitting clicked candidates; collapse duplicate user–article edges.
- Use identical dimension, epochs, sampler, optimizer, learning rate, regularization, initialization, and seed for BPR-MF and LightGCN.
- Use inner products for BPR-MF and LightGCN; do not claim all project models use cosine similarity.
- Use `torch.optim.Adam(..., weight_decay=0.0)` and the explicit sampled layer-0 L2 penalty.
- Use CUDA when available and CPU otherwise; do not use MPS for the sparse graph path.
- Never expose untouched random embeddings: scoring must obey the trained-evidence masks.
- Development results are not final results and must be labelled accordingly.
- Do not change the existing semantic, popularity, split, eligibility, metric, or user-weighting behavior.
- Phase A implementation target is 26 October 2026; analysis/deliverable handoff is due 9 November 2026; presentation is 11 November 2026.

## File map

| File | Responsibility |
|---|---|
| `requirements.txt` | Runtime dependencies, including explicit PyTorch bound. |
| `requirements-dev.txt` | Ruff and mypy quality-tool dependencies. |
| `src/config.py` | Fixed BPR/LightGCN settings and registered model order. |
| `src/models/bpr.py` | Pair pool, sampled triples, shared embedding provider, objective, training result, and optimizer. |
| `src/models/collaborative.py` | Detached fitted state, user fallback, candidate masking, and inner-product scoring. |
| `src/models/bpr_mf.py` | `BPRMFModel` wrapper around the shared trainer. |
| `src/models/lightgcn.py` | Graph construction, normalized adjacency, propagation, trained-mask expansion, and `LightGCNModel`. |
| `src/run.py` | Export fixed settings and per-model runtime metadata. |
| `scripts/smoke_mind.py` | Tiny real-data smoke with valid BPR fitting impressions. |
| `tests/test_bpr.py` | Pair, sampling, loss, training, state reset, and BPR scoring tests. |
| `tests/test_lightgcn.py` | Exact graph, propagation, ablation, leakage, and mask tests. |
| `tests/test_infrastructure.py` | Cross-model metadata and output-contract assertions. |
| `docs/marcus-model-handoff.md` | Exact method, commands, results, runtime, limitations, and downstream artifact map. |
| `tasks.md` | Evidence-backed completion status only. |

---

### Task 1: Pin the training environment and fixed settings

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `src/config.py`
- Test: `tests/test_bpr.py`

**Interfaces:**
- Consumes: existing Python 3.11 project configuration.
- Produces: `BPR_DIM`, `BPR_EPOCHS`, `BPR_NEGATIVES`, `BPR_LEARNING_RATE`, `BPR_L2`, `BPR_INIT_STD`, and `LIGHTGCN_LAYERS` constants used by every later task.

- [ ] **Step 1: Write the failing fixed-settings test**

Create `tests/test_bpr.py` with the imports and first test:

```python
import unittest

from src import config


class BPRTests(unittest.TestCase):
    def test_fixed_training_settings(self) -> None:
        self.assertEqual(config.BPR_DIM, 64)
        self.assertEqual(config.BPR_EPOCHS, 20)
        self.assertEqual(config.BPR_NEGATIVES, 1)
        self.assertEqual(config.BPR_LEARNING_RATE, 0.001)
        self.assertEqual(config.BPR_L2, 0.0001)
        self.assertEqual(config.BPR_INIT_STD, 0.1)
        self.assertEqual(config.LIGHTGCN_LAYERS, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing constants fail**

Run: `python3 -m unittest tests.test_bpr.BPRTests.test_fixed_training_settings -v`  
Expected: `ERROR` with `AttributeError: module 'src.config' has no attribute 'BPR_DIM'`.

- [ ] **Step 3: Add exact dependencies and constants**

Append to `requirements.txt`:

```text
torch>=2.5,<3
```

Create `requirements-dev.txt`:

```text
mypy>=1.13,<3
ruff>=0.8,<1
```

Add to `src/config.py` immediately after `BOOTSTRAP_ITERATIONS`:

```python
BPR_DIM = 64
BPR_EPOCHS = 20
BPR_NEGATIVES = 1
BPR_LEARNING_RATE = 0.001
BPR_L2 = 0.0001
BPR_INIT_STD = 0.1
LIGHTGCN_LAYERS = 2
```

- [ ] **Step 4: Create and populate the project environment**

Run:

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
```

Expected: installation succeeds, then `.venv/bin/python -c "import torch; print(torch.__version__)"` prints a `2.x` version. If the download times out, rerun the same pip command; do not modify the version constraints or use the externally managed system Python.

- [ ] **Step 5: Run the fixed-settings test and baseline suite**

Run:

```bash
.venv/bin/python -m unittest tests.test_bpr.BPRTests.test_fixed_training_settings -v
.venv/bin/python -m unittest discover -s tests -v
```

Expected: the settings test and all 20 pre-existing tests pass.

- [ ] **Step 6: Commit the environment contract**

```bash
git add requirements.txt requirements-dev.txt src/config.py tests/test_bpr.py
git commit -m "build: add fixed collaborative training environment"
```

---

### Task 2: Build deterministic impression-conditioned BPR training

**Files:**
- Create: `src/models/bpr.py`
- Modify: `tests/test_bpr.py`

**Interfaces:**
- Consumes: `TrainingData`, `DataError`, and the Task 1 constants.
- Produces:
  - `PairPool(user_indices, positive_indices, example_indices, negative_options, included_impressions, skipped_impressions)`
  - `SampledPairs(user_indices, positive_indices, negative_indices, example_indices)`
  - `build_pair_pool(data: TrainingData) -> PairPool`
  - `sample_pairs(pool: PairPool, epoch: int, seed: int, device: torch.device) -> SampledPairs`
  - `EmbeddingProvider(num_users: int, num_articles: int, dimension: int, seed: int)`
  - `pairwise_bpr_loss(user_vectors, positive_vectors, negative_vectors) -> Tensor`
  - `bpr_objective(provider: EmbeddingProvider, pairs: SampledPairs, l2: float) -> Tensor`
  - `train_bpr(provider: EmbeddingProvider, pool: PairPool, *, epochs: int, learning_rate: float, l2: float, seed: int) -> TrainingResult`

- [ ] **Step 1: Add failing pair-pool and deterministic-sampler tests**

Add these imports and fixtures to `tests/test_bpr.py`:

```python
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError
from src.models.bpr import build_pair_pool, sample_pairs


def make_training_data() -> TrainingData:
    articles = {
        article: Article(article, "topic", f"Title {article}", "")
        for article in ("A", "B", "C", "D")
    }
    rows = (
        Example(
            RankingInput("I1", "U1", datetime(2019, 11, 1), ("A",), 1, ("B", "C", "D")),
            Targets((1, 0, 0)),
        ),
        Example(
            RankingInput("I2", "U2", datetime(2019, 11, 2), ("B",), 1, ("A", "D")),
            Targets((1, 1)),
        ),
    )
    return TrainingData(
        articles,
        rows,
        {"U1": 0, "U2": 1},
        {article: index for index, article in enumerate(articles)},
    )
```

Add methods to `BPRTests`:

```python
def test_pair_pool_skips_one_class_impressions(self) -> None:
    pool = build_pair_pool(make_training_data())
    np.testing.assert_array_equal(pool.user_indices, [0])
    np.testing.assert_array_equal(pool.positive_indices, [1])
    np.testing.assert_array_equal(pool.example_indices, [0])
    np.testing.assert_array_equal(pool.negative_options[0], [2, 3])
    self.assertEqual((pool.included_impressions, pool.skipped_impressions), (1, 1))

def test_sampling_is_epoch_deterministic_and_stays_in_impression(self) -> None:
    pool = build_pair_pool(make_training_data())
    first = sample_pairs(pool, epoch=0, seed=4285, device=torch.device("cpu"))
    repeat = sample_pairs(pool, epoch=0, seed=4285, device=torch.device("cpu"))
    self.assertTrue(torch.equal(first.negative_indices, repeat.negative_indices))
    self.assertIn(int(first.negative_indices[0]), (2, 3))
    self.assertEqual(int(first.example_indices[0]), 0)

def test_pair_pool_rejects_no_usable_pairs(self) -> None:
    data = make_training_data()
    with self.assertRaisesRegex(DataError, "no usable BPR pairs"):
        build_pair_pool(
            TrainingData(
                data.articles,
                data.examples[1:],
                data.user_indices,
                data.article_indices,
            )
        )

def test_pair_pool_rejects_noncontiguous_indices(self) -> None:
    data = make_training_data()
    broken = TrainingData(
        data.articles,
        data.examples,
        data.user_indices,
        {"A": 0, "B": 2, "C": 3, "D": 4},
    )
    with self.assertRaisesRegex(DataError, "contiguous article indices"):
        build_pair_pool(broken)
```

- [ ] **Step 2: Run the pair tests and verify import failure**

Run: `.venv/bin/python -m unittest tests.test_bpr.BPRTests.test_pair_pool_skips_one_class_impressions tests.test_bpr.BPRTests.test_sampling_is_epoch_deterministic_and_stays_in_impression tests.test_bpr.BPRTests.test_pair_pool_rejects_no_usable_pairs tests.test_bpr.BPRTests.test_pair_pool_rejects_noncontiguous_indices -v`  
Expected: `ERROR` because `src.models.bpr` does not exist.

- [ ] **Step 3: Implement the pair pool and sampler**

Create `src/models/bpr.py` with these exact public structures and behavior:

```python
"""Shared deterministic BPR pairs, objective, and optimization."""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from src import config
from src.contracts import TrainingData
from src.data import DataError


@dataclass(frozen=True)
class PairPool:
    user_indices: NDArray[np.int64]
    positive_indices: NDArray[np.int64]
    example_indices: NDArray[np.int64]
    negative_options: tuple[NDArray[np.int64], ...]
    included_impressions: int
    skipped_impressions: int


@dataclass(frozen=True)
class SampledPairs:
    user_indices: Tensor
    positive_indices: Tensor
    negative_indices: Tensor
    example_indices: Tensor


def _validate_indices(data: TrainingData) -> None:
    for name, mapping in (
        ("user", data.user_indices),
        ("article", data.article_indices),
    ):
        if sorted(mapping.values()) != list(range(len(mapping))):
            raise DataError(f"Training data requires contiguous {name} indices")


def build_pair_pool(data: TrainingData) -> PairPool:
    _validate_indices(data)
    users: list[int] = []
    positives: list[int] = []
    example_indices: list[int] = []
    negative_options: list[NDArray[np.int64]] = []
    included = 0
    skipped = 0
    for example_index, example in enumerate(data.examples):
        candidate_labels = tuple(
            zip(example.inputs.candidate_ids, example.targets.labels, strict=True)
        )
        positive_ids = [article for article, label in candidate_labels if label == 1]
        negative_ids = [article for article, label in candidate_labels if label == 0]
        if not positive_ids or not negative_ids:
            skipped += 1
            continue
        try:
            user_index = data.user_indices[example.inputs.user_id]
            negative_indices = np.asarray(
                [data.article_indices[article] for article in negative_ids], dtype=np.int64
            )
            positive_indices = [data.article_indices[article] for article in positive_ids]
        except KeyError as error:
            raise DataError(f"BPR index mapping is missing {error.args[0]}") from error
        included += 1
        for positive_index in positive_indices:
            users.append(user_index)
            positives.append(positive_index)
            example_indices.append(example_index)
            negative_options.append(negative_indices)
    if not users:
        raise DataError("Training data contains no usable BPR pairs")
    return PairPool(
        np.asarray(users, dtype=np.int64),
        np.asarray(positives, dtype=np.int64),
        np.asarray(example_indices, dtype=np.int64),
        tuple(negative_options),
        included,
        skipped,
    )


def sample_pairs(pool: PairPool, epoch: int, seed: int, device: torch.device) -> SampledPairs:
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    negatives = np.asarray(
        [options[rng.integers(len(options))] for options in pool.negative_options],
        dtype=np.int64,
    )

    def to_tensor(values: NDArray[np.int64]) -> Tensor:
        return torch.as_tensor(values, dtype=torch.long, device=device)

    return SampledPairs(
        to_tensor(pool.user_indices),
        to_tensor(pool.positive_indices),
        to_tensor(negatives),
        to_tensor(pool.example_indices),
    )
```

- [ ] **Step 4: Run the pair tests and verify they pass**

Run the command from Step 2.  
Expected: all four tests pass.

- [ ] **Step 5: Add failing exact-loss and one-step-update tests**

Extend imports from `src.models.bpr` to include `EmbeddingProvider`, `bpr_objective`, and `train_bpr`. Add:

```python
def test_exact_bpr_objective(self) -> None:
    provider = EmbeddingProvider(1, 2, 1, seed=4285)
    with torch.no_grad():
        provider.user_embeddings.weight[:] = torch.tensor([[1.0]])
        provider.article_embeddings.weight[:] = torch.tensor([[2.0], [0.0]])
    pairs = sample_pairs(
        build_pair_pool(
            TrainingData(
                {"P": Article("P", "x", "P", ""), "N": Article("N", "x", "N", "")},
                (
                    Example(
                        RankingInput("I", "U", datetime(2019, 1, 1), (), 0, ("P", "N")),
                        Targets((1, 0)),
                    ),
                ),
                {"U": 0},
                {"P": 0, "N": 1},
            )
        ),
        epoch=0,
        seed=4285,
        device=torch.device("cpu"),
    )
    loss = bpr_objective(provider, pairs, l2=0.0)
    self.assertAlmostEqual(float(loss), float(np.log1p(np.exp(-2.0))), places=6)

def test_shared_training_increases_positive_margin(self) -> None:
    data = make_training_data()
    provider = EmbeddingProvider(2, 4, 4, seed=4285)
    pool = build_pair_pool(data)
    with torch.no_grad():
        before_users, before_articles = provider()
        before = float(np.mean([
            float((before_users[0] * before_articles[1]).sum())
            - float((before_users[0] * before_articles[index]).sum())
            for index in (2, 3)
        ]))
    result = train_bpr(provider, pool, epochs=5, learning_rate=0.01, l2=0.0, seed=4285)
    after = float(np.mean([
        result.user_vectors[0] @ result.article_vectors[1]
        - result.user_vectors[0] @ result.article_vectors[index]
        for index in (2, 3)
    ]))
    self.assertGreater(after, before)
    self.assertTrue(np.isfinite(result.losses).all())

def test_shared_training_rejects_nonfinite_loss(self) -> None:
    data = make_training_data()
    provider = EmbeddingProvider(2, 4, 4, seed=4285)
    with (
        patch(
            "src.models.bpr.bpr_objective",
            return_value=torch.tensor(float("nan"), requires_grad=True),
        ),
        self.assertRaisesRegex(DataError, "non-finite loss at epoch 0"),
    ):
        train_bpr(
            provider,
            build_pair_pool(data),
            epochs=1,
            learning_rate=0.01,
            l2=0.0,
            seed=4285,
        )
```

- [ ] **Step 6: Run the new tests and verify missing interfaces fail**

Run: `.venv/bin/python -m unittest tests.test_bpr.BPRTests.test_exact_bpr_objective tests.test_bpr.BPRTests.test_shared_training_increases_positive_margin tests.test_bpr.BPRTests.test_shared_training_rejects_nonfinite_loss -v`  
Expected: `ImportError` for the not-yet-defined public interfaces.

- [ ] **Step 7: Implement the shared provider, objective, and trainer**

Append to `src/models/bpr.py`:

```python
@dataclass(frozen=True)
class TrainingResult:
    user_vectors: NDArray[np.float32]
    article_vectors: NDArray[np.float32]
    trained_users: NDArray[np.bool_]
    trained_articles: NDArray[np.bool_]
    losses: tuple[float, ...]
    device: str


class EmbeddingProvider(nn.Module):
    def __init__(
        self, num_users: int, num_articles: int, dimension: int, seed: int
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.user_embeddings = nn.Embedding(num_users, dimension)
        self.article_embeddings = nn.Embedding(num_articles, dimension)
        with torch.no_grad():
            self.user_embeddings.weight.normal_(0.0, config.BPR_INIT_STD, generator=generator)
            self.article_embeddings.weight.normal_(0.0, config.BPR_INIT_STD, generator=generator)

    def forward(self) -> tuple[Tensor, Tensor]:
        return self.user_embeddings.weight, self.article_embeddings.weight

    def expand_trained_masks(
        self, users: NDArray[np.bool_], articles: NDArray[np.bool_]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        return users, articles


def pairwise_bpr_loss(
    user_vectors: Tensor, positive_vectors: Tensor, negative_vectors: Tensor
) -> Tensor:
    margin = (user_vectors * positive_vectors).sum(dim=1) - (
        user_vectors * negative_vectors
    ).sum(dim=1)
    return F.softplus(-margin).mean()


def bpr_objective(provider: EmbeddingProvider, pairs: SampledPairs, l2: float) -> Tensor:
    users, articles = provider()
    user_vectors = users[pairs.user_indices]
    positive_vectors = articles[pairs.positive_indices]
    negative_vectors = articles[pairs.negative_indices]
    ranking = pairwise_bpr_loss(user_vectors, positive_vectors, negative_vectors)
    layer0_users = provider.user_embeddings(pairs.user_indices)
    layer0_positives = provider.article_embeddings(pairs.positive_indices)
    layer0_negatives = provider.article_embeddings(pairs.negative_indices)
    penalty = (
        layer0_users.square().sum(dim=1)
        + layer0_positives.square().sum(dim=1)
        + layer0_negatives.square().sum(dim=1)
    ).mean()
    return ranking + l2 * penalty


def training_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_bpr(
    provider: EmbeddingProvider,
    pool: PairPool,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> TrainingResult:
    if epochs <= 0:
        raise ValueError("BPR epochs must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = training_device()
    provider = provider.to(device)
    optimizer = torch.optim.Adam(provider.parameters(), lr=learning_rate, weight_decay=0.0)
    trained_users = np.zeros(provider.user_embeddings.num_embeddings, dtype=np.bool_)
    trained_articles = np.zeros(provider.article_embeddings.num_embeddings, dtype=np.bool_)
    losses: list[float] = []
    for epoch in range(epochs):
        pairs = sample_pairs(pool, epoch, seed, device)
        trained_users[pairs.user_indices.detach().cpu().numpy()] = True
        trained_articles[pairs.positive_indices.detach().cpu().numpy()] = True
        trained_articles[pairs.negative_indices.detach().cpu().numpy()] = True
        optimizer.zero_grad(set_to_none=True)
        loss = bpr_objective(provider, pairs, l2)
        if not torch.isfinite(loss):
            raise DataError(f"BPR produced non-finite loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    provider.eval()
    with torch.no_grad():
        users, articles = provider()
        if not torch.isfinite(users).all() or not torch.isfinite(articles).all():
            raise DataError("BPR produced non-finite embeddings")
        user_vectors = users.detach().cpu().numpy().astype(np.float32, copy=True)
        article_vectors = articles.detach().cpu().numpy().astype(np.float32, copy=True)
    trained_users, trained_articles = provider.expand_trained_masks(
        trained_users, trained_articles
    )
    return TrainingResult(
        user_vectors,
        article_vectors,
        trained_users,
        trained_articles,
        tuple(losses),
        str(device),
    )
```

- [ ] **Step 8: Run all BPR tests and static checks**

Run:

```bash
.venv/bin/python -m unittest tests.test_bpr -v
.venv/bin/ruff check src/models/bpr.py tests/test_bpr.py
.venv/bin/mypy src/models/bpr.py tests/test_bpr.py
```

Expected: all commands pass. If the exact loss test warns about converting a grad tensor, change only the assertion expression to `float(loss.detach())`.

- [ ] **Step 9: Commit the shared BPR engine**

```bash
git add src/models/bpr.py tests/test_bpr.py
git commit -m "feat: add deterministic shared BPR training"
```

---

### Task 3: Add shared collaborative scoring and BPR-MF

**Files:**
- Create: `src/models/collaborative.py`
- Create: `src/models/bpr_mf.py`
- Modify: `tests/test_bpr.py`

**Interfaces:**
- Consumes: `TrainingResult`, `build_pair_pool`, `EmbeddingProvider`, `train_bpr`.
- Produces:
  - `CollaborativeState`
  - `make_collaborative_state(data: TrainingData, result: TrainingResult) -> CollaborativeState`
  - `score_collaborative(state: CollaborativeState, inputs: RankingInput) -> NDArray[np.float64]`
  - `BPRMFModel.name == "bpr_mf"`, `fit`, `score`, and `run_metadata`.

- [ ] **Step 1: Add failing cold-start, refit, and contract tests**

Extend `tests/test_bpr.py` imports with `dataclasses.replace` and `BPRMFModel`. Add:

```python
def test_bpr_mf_fit_score_cold_start_and_refit(self) -> None:
    data = make_training_data()
    with patch.multiple(config, BPR_EPOCHS=3, BPR_DIM=4):
        model = BPRMFModel()
        model.fit(data)
        first = model.score(data.examples[0].inputs)
        model.fit(data)
        second = model.score(data.examples[0].inputs)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    self.assertEqual(first.shape, (3,))
    self.assertTrue(np.isfinite(first).all())
    unseen = replace(data.examples[0].inputs, user_id="NEW", history=("B",))
    self.assertTrue(np.isfinite(model.score(unseen)).all())
    empty = replace(unseen, history=())
    np.testing.assert_array_equal(model.score(empty), np.zeros(3))

def test_bpr_mf_masks_untrained_candidates(self) -> None:
    data = make_training_data()
    with patch.multiple(config, BPR_EPOCHS=1, BPR_DIM=4):
        model = BPRMFModel()
        model.fit(data)
    inputs = replace(data.examples[0].inputs, candidate_ids=("A",))
    np.testing.assert_array_equal(model.score(inputs), np.zeros(1))

def test_bpr_mf_rejects_score_before_fit(self) -> None:
    with self.assertRaisesRegex(DataError, "before fit"):
        BPRMFModel().score(make_training_data().examples[0].inputs)
```

- [ ] **Step 2: Run the model tests and verify import failure**

Run: `.venv/bin/python -m unittest tests.test_bpr.BPRTests.test_bpr_mf_fit_score_cold_start_and_refit tests.test_bpr.BPRTests.test_bpr_mf_masks_untrained_candidates tests.test_bpr.BPRTests.test_bpr_mf_rejects_score_before_fit -v`  
Expected: `ImportError` because `src.models.bpr_mf` does not exist.

- [ ] **Step 3: Implement detached collaborative state and scoring**

Create `src/models/collaborative.py`:

```python
"""Shared fitted state and cold-start scoring for collaborative models."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import TrainingResult


@dataclass(frozen=True)
class CollaborativeState:
    user_indices: Mapping[str, int]
    article_indices: Mapping[str, int]
    user_vectors: NDArray[np.float32]
    article_vectors: NDArray[np.float32]
    trained_users: NDArray[np.bool_]
    trained_articles: NDArray[np.bool_]


def make_collaborative_state(
    data: TrainingData, result: TrainingResult
) -> CollaborativeState:
    return CollaborativeState(
        dict(data.user_indices),
        dict(data.article_indices),
        result.user_vectors,
        result.article_vectors,
        result.trained_users,
        result.trained_articles,
    )


def score_collaborative(
    state: CollaborativeState, inputs: RankingInput
) -> NDArray[np.float64]:
    user_index = state.user_indices.get(inputs.user_id)
    if user_index is not None and state.trained_users[user_index]:
        user_vector = state.user_vectors[user_index]
    else:
        history = [
            index
            for article in inputs.history
            if (index := state.article_indices.get(article)) is not None
            and state.trained_articles[index]
        ]
        if not history:
            return np.zeros(len(inputs.candidate_ids), dtype=np.float64)
        user_vector = state.article_vectors[history].mean(axis=0)
    scores = np.zeros(len(inputs.candidate_ids), dtype=np.float64)
    for position, article in enumerate(inputs.candidate_ids):
        article_index = state.article_indices.get(article)
        if article_index is not None and state.trained_articles[article_index]:
            scores[position] = float(user_vector @ state.article_vectors[article_index])
    if not np.isfinite(scores).all():
        raise DataError(f"Non-finite collaborative scores for {inputs.impression_id}")
    return scores
```

- [ ] **Step 4: Implement `BPRMFModel`**

Create `src/models/bpr_mf.py`:

```python
"""Matrix factorization trained with the shared BPR objective."""

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from src import config
from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import EmbeddingProvider, build_pair_pool, train_bpr
from src.models.collaborative import (
    CollaborativeState,
    make_collaborative_state,
    score_collaborative,
)


class BPRMFModel:
    def __init__(self) -> None:
        self.state: CollaborativeState | None = None
        self.run_metadata: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return "bpr_mf"

    def fit(self, data: TrainingData) -> None:
        self.state = None
        self.run_metadata = {}
        pool = build_pair_pool(data)
        provider = EmbeddingProvider(
            len(data.user_indices), len(data.article_indices), config.BPR_DIM, config.SEED
        )
        result = train_bpr(
            provider,
            pool,
            epochs=config.BPR_EPOCHS,
            learning_rate=config.BPR_LEARNING_RATE,
            l2=config.BPR_L2,
            seed=config.SEED,
        )
        self.state = make_collaborative_state(data, result)
        self.run_metadata = {
            "device": result.device,
            "epochs": len(result.losses),
            "final_loss": result.losses[-1],
            "included_impressions": pool.included_impressions,
            "skipped_impressions": pool.skipped_impressions,
            "trained_users": int(result.trained_users.sum()),
            "trained_articles": int(result.trained_articles.sum()),
        }

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if self.state is None:
            raise DataError("BPR-MF score called before fit")
        return score_collaborative(self.state, inputs)
```

- [ ] **Step 5: Run BPR-MF tests and the full baseline**

Run:

```bash
.venv/bin/python -m unittest tests.test_bpr -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src/models/collaborative.py src/models/bpr_mf.py tests/test_bpr.py
.venv/bin/mypy src/models/collaborative.py src/models/bpr_mf.py tests/test_bpr.py
```

Expected: every command passes, repeated fits are deterministic, and the existing 20 tests remain green.

- [ ] **Step 6: Commit BPR-MF**

```bash
git add src/models/collaborative.py src/models/bpr_mf.py tests/test_bpr.py
git commit -m "feat: add BPR matrix factorization model"
```

---

### Task 4: Build exact LightGCN graph propagation and trained masks

**Files:**
- Create: `src/models/lightgcn.py`
- Create: `tests/test_lightgcn.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `PairPool`, `train_bpr`, `score_collaborative`.
- Produces:
  - `GraphData(adjacency, rows, columns, num_users, num_articles)`
  - `build_graph(data: TrainingData) -> GraphData`
  - `expand_masks(graph: GraphData, users: NDArray[np.bool_], articles: NDArray[np.bool_], layers: int) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]`
  - `LightGCNEmbeddings(num_users: int, num_articles: int, dimension: int, seed: int, graph: GraphData, layers: int)`
  - `LightGCNModel.name == "lightgcn"`, `fit`, `score`, and `run_metadata`.

- [ ] **Step 1: Write failing exact graph and propagation tests**

Create `tests/test_lightgcn.py`:

```python
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from src import config
from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError
from src.models.bpr import EmbeddingProvider
from src.models.lightgcn import (
    LightGCNEmbeddings,
    build_graph,
    expand_masks,
)


def graph_fixture() -> TrainingData:
    articles = {
        article: Article(article, "topic", article, "")
        for article in ("A", "B", "C", "D")
    }
    examples = (
        Example(
            RankingInput("I1", "U1", datetime(2019, 1, 1), ("A",), 1, ("B", "C")),
            Targets((1, 0)),
        ),
        Example(
            RankingInput("I2", "U2", datetime(2019, 1, 2), ("B",), 1, ("A", "C")),
            Targets((1, 0)),
        ),
    )
    return TrainingData(
        articles,
        examples,
        {"U1": 0, "U2": 1},
        {"A": 0, "B": 1, "C": 2, "D": 3},
    )


class LightGCNTests(unittest.TestCase):
    def test_graph_collapses_edges_and_normalizes_degrees(self) -> None:
        graph = build_graph(graph_fixture())
        dense = graph.adjacency.to_dense().numpy()
        expected = np.zeros((6, 6), dtype=np.float32)
        weight = 0.5
        for user, article in ((0, 2), (0, 3), (1, 2), (1, 3)):
            expected[user, article] = weight
            expected[article, user] = weight
        np.testing.assert_allclose(dense, expected, atol=1e-7)

    def test_one_layer_propagation_and_zero_layer_ablation(self) -> None:
        graph = build_graph(graph_fixture())
        zero = LightGCNEmbeddings(2, 4, 1, 4285, graph, layers=0)
        one = LightGCNEmbeddings(2, 4, 1, 4285, graph, layers=1)
        with torch.no_grad():
            values = torch.tensor([[1.0], [3.0], [2.0], [4.0], [8.0], [16.0]])
            for provider in (zero, one):
                provider.user_embeddings.weight[:] = values[:2]
                provider.article_embeddings.weight[:] = values[2:]
        raw = EmbeddingProvider(2, 4, 1, 4285)
        raw.load_state_dict(zero.state_dict(), strict=False)
        raw_users, raw_articles = raw()
        zero_users, zero_articles = zero()
        np.testing.assert_allclose(zero_users.detach().numpy(), raw_users.detach().numpy())
        np.testing.assert_allclose(zero_articles.detach().numpy(), raw_articles.detach().numpy())
        one_users, one_articles = one()
        np.testing.assert_allclose(one_users.detach().numpy().ravel(), [2.0, 3.0])
        np.testing.assert_allclose(
            one_articles.detach().numpy().ravel(), [2.0, 3.0, 4.0, 8.0]
        )

    def test_trained_masks_expand_exactly_two_hops(self) -> None:
        graph = build_graph(graph_fixture())
        users = np.asarray([True, False])
        articles = np.asarray([False, False, True, False])
        expanded_users, expanded_articles = expand_masks(graph, users, articles, layers=2)
        np.testing.assert_array_equal(expanded_users, [True, True])
        np.testing.assert_array_equal(expanded_articles, [True, True, True, False])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `.venv/bin/python -m unittest tests.test_lightgcn -v`  
Expected: `ImportError` because `src.models.lightgcn` does not exist.

- [ ] **Step 3: Implement graph construction, propagation, and structural mask expansion**

Create the first part of `src/models/lightgcn.py`:

```python
"""LightGCN graph construction, propagation, and recommender wrapper."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src import config
from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import EmbeddingProvider, build_pair_pool, train_bpr
from src.models.collaborative import (
    CollaborativeState,
    make_collaborative_state,
    score_collaborative,
)


@dataclass(frozen=True)
class GraphData:
    adjacency: Tensor
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    num_users: int
    num_articles: int


def build_graph(data: TrainingData) -> GraphData:
    num_users = len(data.user_indices)
    num_articles = len(data.article_indices)
    edges: set[tuple[int, int]] = set()
    for example in data.examples:
        try:
            user = data.user_indices[example.inputs.user_id]
            positives = [
                article
                for article, label in zip(
                    example.inputs.candidate_ids, example.targets.labels, strict=True
                )
                if label == 1
            ]
            for article in (*example.inputs.history, *positives):
                edges.add((user, data.article_indices[article]))
        except KeyError as error:
            raise DataError(f"LightGCN index mapping is missing {error.args[0]}") from error
    if not edges:
        raise DataError("LightGCN graph has no fitting interactions")
    undirected = [
        (left, num_users + right)
        for left, right in sorted(edges)
    ]
    undirected += [(right, left) for left, right in undirected]
    rows = np.asarray([row for row, _ in undirected], dtype=np.int64)
    columns = np.asarray([column for _, column in undirected], dtype=np.int64)
    degrees = np.bincount(rows, minlength=num_users + num_articles).astype(np.float64)
    values = 1.0 / np.sqrt(degrees[rows] * degrees[columns])
    if not np.isfinite(values).all():
        raise DataError("LightGCN graph normalization is non-finite")
    indices = torch.as_tensor(np.vstack((rows, columns)), dtype=torch.long)
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.as_tensor(values, dtype=torch.float32),
        (num_users + num_articles, num_users + num_articles),
    ).coalesce()
    return GraphData(adjacency, rows, columns, num_users, num_articles)


def expand_masks(
    graph: GraphData,
    users: NDArray[np.bool_],
    articles: NDArray[np.bool_],
    layers: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    active = np.concatenate((users, articles)).copy()
    for _ in range(layers):
        expanded = active.copy()
        expanded[graph.columns[active[graph.rows]]] = True
        active = expanded
    return active[: graph.num_users], active[graph.num_users :]


class LightGCNEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        num_users: int,
        num_articles: int,
        dimension: int,
        seed: int,
        graph: GraphData,
        layers: int,
    ) -> None:
        super().__init__(num_users, num_articles, dimension, seed)
        self.layers = layers
        self.graph = graph
        self.register_buffer("adjacency", graph.adjacency)

    def forward(self) -> tuple[Tensor, Tensor]:
        layer = torch.cat((self.user_embeddings.weight, self.article_embeddings.weight))
        layers = [layer]
        for _ in range(self.layers):
            layer = torch.sparse.mm(self.adjacency, layer)
            layers.append(layer)
        combined = torch.stack(layers).mean(dim=0)
        return combined[: self.graph.num_users], combined[self.graph.num_users :]

    def expand_trained_masks(
        self, users: NDArray[np.bool_], articles: NDArray[np.bool_]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        return expand_masks(self.graph, users, articles, self.layers)
```

- [ ] **Step 4: Run the exact graph tests and correct only verified arithmetic**

Run: `.venv/bin/python -m unittest tests.test_lightgcn -v`  
Expected: all three tests pass. If the hand-computed propagation expectation fails, inspect `adjacency.to_dense()` and recalculate the expected values before changing implementation; do not loosen tolerances beyond `1e-7`.

- [ ] **Step 5: Add failing LightGCN model, determinism, graph-boundary, and scoring tests**

Extend the `src.models.lightgcn` import with `LightGCNModel`. Add:

```python
def test_model_is_deterministic_and_finite(self) -> None:
    data = graph_fixture()
    with patch.multiple(config, BPR_EPOCHS=3, BPR_DIM=4, LIGHTGCN_LAYERS=2):
        model = LightGCNModel()
        model.fit(data)
        first = model.score(data.examples[0].inputs)
        model.fit(data)
        second = model.score(data.examples[0].inputs)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    self.assertTrue(np.isfinite(first).all())

def test_graph_uses_histories_and_positive_candidates_but_not_negatives(self) -> None:
    graph = build_graph(graph_fixture())
    dense = graph.adjacency.to_dense().numpy()
    self.assertGreater(dense[0, 2], 0.0)  # U1 history A
    self.assertGreater(dense[0, 3], 0.0)  # U1 clicked B
    self.assertEqual(dense[0, 4], 0.0)  # U1 displayed-negative C

def test_score_before_fit_fails(self) -> None:
    with self.assertRaisesRegex(DataError, "before fit"):
        LightGCNModel().score(graph_fixture().examples[0].inputs)

def test_disconnected_untrained_article_scores_zero(self) -> None:
    data = graph_fixture()
    with patch.multiple(config, BPR_EPOCHS=2, BPR_DIM=4):
        model = LightGCNModel()
        model.fit(data)
    inputs = RankingInput("E", "U1", datetime(2019, 1, 3), ("A",), 1, ("D",))
    np.testing.assert_array_equal(model.score(inputs), np.zeros(1))
```

- [ ] **Step 6: Implement `LightGCNModel`**

Append to `src/models/lightgcn.py`:

```python
class LightGCNModel:
    def __init__(self) -> None:
        self.state: CollaborativeState | None = None
        self.run_metadata: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return "lightgcn"

    def fit(self, data: TrainingData) -> None:
        self.state = None
        self.run_metadata = {}
        pool = build_pair_pool(data)
        graph = build_graph(data)
        provider = LightGCNEmbeddings(
            len(data.user_indices),
            len(data.article_indices),
            config.BPR_DIM,
            config.SEED,
            graph,
            config.LIGHTGCN_LAYERS,
        )
        result = train_bpr(
            provider,
            pool,
            epochs=config.BPR_EPOCHS,
            learning_rate=config.BPR_LEARNING_RATE,
            l2=config.BPR_L2,
            seed=config.SEED,
        )
        self.state = make_collaborative_state(data, result)
        self.run_metadata = {
            "device": result.device,
            "epochs": len(result.losses),
            "final_loss": result.losses[-1],
            "layers": config.LIGHTGCN_LAYERS,
            "unique_graph_edges": len(graph.rows) // 2,
            "included_impressions": pool.included_impressions,
            "skipped_impressions": pool.skipped_impressions,
            "trained_users": int(result.trained_users.sum()),
            "trained_articles": int(result.trained_articles.sum()),
        }

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if self.state is None:
            raise DataError("LightGCN score called before fit")
        return score_collaborative(self.state, inputs)
```

- [ ] **Step 7: Run all LightGCN and repository tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_lightgcn -v
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check src/models/lightgcn.py tests/test_lightgcn.py
.venv/bin/mypy src/models/lightgcn.py tests/test_lightgcn.py
```

Expected: every command passes.

- [ ] **Step 8: Commit LightGCN**

```bash
git add src/models/lightgcn.py tests/test_lightgcn.py
git commit -m "feat: add LightGCN propagation model"
```

---

### Task 5: Integrate both models and prove output compatibility

**Files:**
- Modify: `src/config.py`
- Modify: `src/run.py`
- Modify: `scripts/smoke_mind.py`
- Modify: `tests/test_infrastructure.py`
- Modify: `tests/test_bpr.py`

**Interfaces:**
- Consumes: `BPRMFModel`, `LightGCNModel`, and existing experiment/evaluation functions.
- Produces: model order `(PopularityModel, SemanticModel, BPRMFModel, LightGCNModel)`, fixed training metadata in `metrics.json`, and a smoke cohort with usable BPR pairs.

- [ ] **Step 1: Add failing registration and metadata tests**

Add imports for both new model classes to `tests/test_bpr.py`, then add:

```python
def test_collaborative_models_are_registered_after_existing_models(self) -> None:
    self.assertEqual(
        tuple(model_type().name for model_type in config.MODELS),
        ("popularity", "semantic", "bpr_mf", "lightgcn"),
    )
```

In `tests/test_infrastructure.py`, extend `test_evaluation_and_standard_run_outputs` after `metrics.json` is created:

```python
with (self.root / "run/metrics.json").open(encoding="utf-8") as handle:
    summary = json.load(handle)
self.assertEqual(summary["settings"]["bpr_dimension"], config.BPR_DIM)
self.assertEqual(summary["settings"]["lightgcn_layers"], config.LIGHTGCN_LAYERS)
self.assertEqual(summary["models"]["popularity"]["ndcg_at_5"], 1.0)
self.assertEqual(summary["training"]["popularity"], {})
```

Add `import json` at the top of that test file.

- [ ] **Step 2: Run tests and verify registration/shape failures**

Run: `.venv/bin/python -m unittest tests.test_bpr.BPRTests.test_collaborative_models_are_registered_after_existing_models tests.test_infrastructure.InfrastructureTests.test_evaluation_and_standard_run_outputs -v`  
Expected: registration assertion fails first; after registration, the old summary shape fails until `src/run.py` is updated.

- [ ] **Step 3: Register models and export fixed/per-model metadata**

In `src/config.py`, import the model classes and set:

```python
from src.models.bpr_mf import BPRMFModel
from src.models.lightgcn import LightGCNModel

MODELS: tuple[type[Model], ...] = (
    PopularityModel,
    SemanticModel,
    BPRMFModel,
    LightGCNModel,
)
```

In `src/run.py`, change the result type and model assignment:

```python
results: dict[str, dict[str, int | float]] = {}
training_metadata: dict[str, dict[str, object]] = {}
for model_type in config.MODELS:
    model = model_type()
    model.fit(data.training)
    metrics = evaluate_model(model, data.evaluation, output_dir / model.name)
    metadata = getattr(model, "run_metadata", None)
    results[model.name] = metrics
    training_metadata[model.name] = dict(metadata) if isinstance(metadata, Mapping) else {}
    print(f"{model.name}: {metrics}")
summary["models"] = results
summary["training"] = training_metadata
```

Import `Mapping` from `collections.abc` at the top of `src/run.py`.

After the existing `summary` initialization, extend its settings with:

```python
settings = summary["settings"]
if not isinstance(settings, dict):
    raise TypeError("Experiment settings must be a dictionary")
settings.update(
    {
        "bpr_dimension": config.BPR_DIM,
        "bpr_epochs": config.BPR_EPOCHS,
        "bpr_negatives": config.BPR_NEGATIVES,
        "bpr_learning_rate": config.BPR_LEARNING_RATE,
        "bpr_l2": config.BPR_L2,
        "bpr_init_std": config.BPR_INIT_STD,
        "lightgcn_layers": config.LIGHTGCN_LAYERS,
    }
)
```

- [ ] **Step 4: Make the smoke cohort valid for pairwise models**

In `scripts/smoke_mind.py`, import the new model classes. Replace the training selection with:

```python
training = tuple(
    example for example in data.training.examples if set(example.targets.labels) == {0, 1}
)[:2]
if len(training) != 2:
    raise DataError("Real-MIND smoke needs two mixed-label fitting impressions")
```

Import `DataError`, and patch:

```python
MODELS=(PopularityModel, SemanticModel, BPRMFModel, LightGCNModel),
BPR_EPOCHS=2,
BPR_DIM=8,
LIGHTGCN_LAYERS=1,
```

Update `test_mind_smoke_limits_examples_and_article_corpus` so its fixture supplies two mixed-label fitting rows and asserts `set(example.targets.labels) == {0, 1}` for every selected training example.

- [ ] **Step 5: Run focused integration tests**

Run:

```bash
.venv/bin/python -m unittest tests.test_bpr.BPRTests.test_collaborative_models_are_registered_after_existing_models -v
.venv/bin/python -m unittest tests.test_infrastructure.InfrastructureTests.test_evaluation_and_standard_run_outputs -v
.venv/bin/python -m unittest tests.test_infrastructure.InfrastructureTests.test_mind_smoke_limits_examples_and_article_corpus -v
```

Expected: all pass and the metadata schema is explicit.

- [ ] **Step 6: Run the complete quality gate**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check .
.venv/bin/mypy src tests
```

Expected: all tests, lint, and strict typing pass. Do not suppress a mypy error with a blanket ignore; use a narrow typed protocol or cast if `run_metadata` requires it.

- [ ] **Step 7: Commit integration**

```bash
git add src/config.py src/run.py scripts/smoke_mind.py tests/test_infrastructure.py tests/test_bpr.py
git commit -m "feat: integrate collaborative models into experiments"
```

---

### Task 6: Pass the real-MIND smoke and feasibility gate

**Files:**
- Modify only if a verified defect is found: `src/models/bpr.py`, `src/models/bpr_mf.py`, `src/models/lightgcn.py`, `scripts/smoke_mind.py`, matching test files.
- Create: `docs/marcus-model-handoff.md`

**Interfaces:**
- Consumes: installed dependencies, downloaded MIND training data, all four registered models.
- Produces: a successful tiny real-data run plus recorded device, pair counts, graph size, runtime, and memory feasibility evidence.

- [ ] **Step 1: Confirm only training data is present**

Run:

```bash
test -f data/raw/train/news.tsv
test -f data/raw/train/behaviors.tsv
test ! -e data/raw/validation/behaviors.tsv
```

Expected: all three commands exit `0`. If training files are absent, run `.venv/bin/python -m scripts.download_mind`. Do not download or inspect validation.

- [ ] **Step 2: Run the real-MIND smoke with timing**

Run:

```bash
/usr/bin/time -l .venv/bin/python -m scripts.smoke_mind 2>&1 | tee /tmp/cp4285-smoke.log
```

Expected: `Real-MIND smoke passed`, four model output directories, paired comparisons for six model pairs, finite metrics, and a recorded maximum resident set size.

- [ ] **Step 3: Audit smoke artifacts mechanically**

Run:

```bash
.venv/bin/python - <<'PY'
import csv
import json
from itertools import combinations
from pathlib import Path

root = Path("outputs/smoke/mind")
names = ("popularity", "semantic", "bpr_mf", "lightgcn")
with (root / "metrics.json").open() as handle:
    summary = json.load(handle)
assert tuple(summary["models"]) == names
reference = None
for name in names:
    with (root / name / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    identity = [(r["impression_id"], r["candidate_index"], r["article_id"], r["label"]) for r in rows]
    reference = identity if reference is None else reference
    assert identity == reference
with (root / "comparisons.json").open() as handle:
    comparisons = json.load(handle)
assert len(comparisons) == len(tuple(combinations(names, 2))) == 6
print("smoke artifact audit passed")
PY
```

Expected: `smoke artifact audit passed`.

- [ ] **Step 4: Record feasibility evidence without result claims**

Create the initial `docs/marcus-model-handoff.md` with the protocol bullets below, then append the observed smoke values directly from `outputs/smoke/mind/metrics.json` and `/tmp/cp4285-smoke.log` under the named fields. Do not write a field until its value has been observed:

```markdown
# Marcus model handoff

## Protocol

- Development only; supplied validation remained unopened.
- BPR pairs: each fitting click versus one displayed unclicked candidate from the same impression per epoch.
- LightGCN graph: distinct fitting history/click edges, symmetric degree normalization, two propagation layers.
- Shared settings: dimension 64, 20 epochs, Adam 0.001, sampled layer-0 L2 0.0001, seed 4285.
- Scoring: inner product with explicit trained-evidence masking and history fallback for untrained users.

## Smoke evidence

- Command: `/usr/bin/time -l .venv/bin/python -m scripts.smoke_mind`
- Status: passed on the tiny real-MIND cohort.
- Smoke metrics are functionality checks and are not reportable project results.

## Remaining gate

The full development run and artifact audit in Task 7 must pass before two-tower work starts.
```

Add `Device`, `BPR pair counts`, `LightGCN graph edges`, `Peak resident memory`, and `Elapsed time` bullets using the exact observed values before committing.

- [ ] **Step 5: Verify the handoff contains concrete smoke evidence**

Run:

```bash
rg -n '^## (Protocol|Smoke evidence|Remaining gate)$' docs/marcus-model-handoff.md
rg -n '^(\- Device|\- BPR pair counts|\- LightGCN graph edges|\- Peak resident memory|\- Elapsed time):' docs/marcus-model-handoff.md
```

Expected: all three headings and all five observed-value bullets are present. If the smoke or artifact audit failed, stop this plan, preserve the exact failing output, add a focused regression test in a reviewed plan amendment, and do not lower model settings based on smoke metrics.

- [ ] **Step 6: Commit the smoke evidence**

```bash
git add docs/marcus-model-handoff.md
git commit -m "test: verify collaborative models on MIND smoke"
```

Before committing, run `git diff --check` and verify `git status --short` lists no generated `data/` or `outputs/` files.

---

### Task 7: Run and audit the full development comparison

**Files:**
- Modify: `docs/marcus-model-handoff.md`
- Modify: `tasks.md`
- Generated but not committed unless the repository policy explicitly changes: `outputs/dev/**`

**Interfaces:**
- Consumes: the frozen Phase A code and development-only training split.
- Produces: four paired prediction exports, aggregate development metrics, six pairwise comparison records, method/runtime notes, and an evidence-backed core-gate decision.

- [ ] **Step 1: Record the exact pre-run revision and clean state**

Run:

```bash
git status --short
git rev-parse HEAD
```

Expected: no source changes and one exact commit hash to record under `## Development run` in the handoff.

- [ ] **Step 2: Run the full development experiment once**

Run:

```bash
/usr/bin/time -l .venv/bin/python -m src.run 2>&1 | tee /tmp/cp4285-development.log
.venv/bin/python -m src.analyze
.venv/bin/python -m src.beyond
```

Expected: all four models complete; `outputs/dev/metrics.json`, `outputs/dev/comparisons.json`, and `outputs/dev/beyond_accuracy.json` exist; every model has nonempty `predictions.csv` and `impression_metrics.csv`. This run may take materially longer than the smoke; do not interrupt it solely because there is no output between model completions.

- [ ] **Step 3: Run the exact paired-output audit**

Run:

```bash
.venv/bin/python - <<'PY'
import csv
import json
import math
from itertools import combinations
from pathlib import Path

root = Path("outputs/dev")
names = ("popularity", "semantic", "bpr_mf", "lightgcn")
with (root / "metrics.json").open() as handle:
    summary = json.load(handle)
assert tuple(summary["models"]) == names
reference = None
for name in names:
    with (root / name / "predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows, name
    identity = [(r["impression_id"], r["user_id"], r["candidate_index"], r["article_id"], r["label"]) for r in rows]
    assert reference is None or identity == reference, name
    reference = identity
    assert all(math.isfinite(float(row["score"])) for row in rows), name
    by_impression = {}
    for row in rows:
        by_impression.setdefault(row["impression_id"], []).append(int(row["rank"]))
    assert all(sorted(ranks) == list(range(1, len(ranks) + 1)) for ranks in by_impression.values()), name
with (root / "comparisons.json").open() as handle:
    comparisons = json.load(handle)
expected = {f"{right}_minus_{left}" for left, right in combinations(names, 2)}
assert set(comparisons) == expected
with (root / "beyond_accuracy.json").open() as handle:
    beyond = json.load(handle)
assert set(beyond) == set(names)
assert all(
    math.isfinite(float(value))
    for metrics in beyond.values()
    for value in metrics.values()
)
print(f"development audit passed: {len(reference)} candidate rows, {len(expected)} comparisons")
PY
```

Expected: `development audit passed` with nonzero candidate rows and six comparisons.

- [ ] **Step 4: Rerun deterministic collaborative fits on the same revision**

Write the first collaborative predictions to explicit temporary comparison names, rerun only the two collaborative models against the same loaded development data, and compare bytes:

```bash
cp outputs/dev/bpr_mf/predictions.csv /tmp/bpr_mf-first.csv
cp outputs/dev/lightgcn/predictions.csv /tmp/lightgcn-first.csv
.venv/bin/python - <<'PY'
from pathlib import Path
from unittest.mock import patch

from src import config
from src.data import load_development
from src.models.bpr_mf import BPRMFModel
from src.models.lightgcn import LightGCNModel
from src.run import run_experiment

with patch.object(config, "MODELS", (BPRMFModel, LightGCNModel)):
    run_experiment(load_development(), Path("/tmp/cp4285-collaborative-repeat"))
PY
cmp /tmp/bpr_mf-first.csv /tmp/cp4285-collaborative-repeat/bpr_mf/predictions.csv
cmp /tmp/lightgcn-first.csv /tmp/cp4285-collaborative-repeat/lightgcn/predictions.csv
```

Expected: both `cmp` commands exit `0`. If semantic re-encoding is expensive, its cache must be reused; do not remove it.

- [ ] **Step 5: Complete the method and result handoff with observed values**

Add the following sections to `docs/marcus-model-handoff.md`. Populate every named metric from `outputs/dev/metrics.json`, every cohort count from the exports, and runtime/memory from `/tmp/cp4285-development.log`; do not use brackets or synthetic example values:

```markdown
## Development run

- Revision
- Training impressions, evaluation users, and evaluation impressions
- Runtime and peak resident memory
- BPR-MF NDCG@5, MRR, final loss, trained users, and trained articles
- LightGCN NDCG@5, MRR, final loss, unique graph edges, trained users, and trained articles
- Pairwise differences: point Yugan to `outputs/dev/comparisons.json`; do not duplicate CIs manually

## Interpretation boundaries

- These are development results, not the held-out final evaluation.
- Click labels measure prediction of recorded clicks among displayed candidates, not satisfaction or causal preference.
- Untrained collaborative candidates receive zero and must be included in failure/subgroup analysis.
- BPR-MF versus LightGCN isolates graph propagation only under the shared fixed optimizer and sampler.

## AI-use record for Sood

- Codex inspected the shared repository, proposed and repeatedly audited the design and implementation plans, and drafted model code, tests, and verification fixes during execution.
- Marcus approved the architecture and is responsible for reviewing the implementation and deciding which results and claims enter the assessed deliverable.
- Before submission, replace this operational record with a declaration that states the review, edits, and experiment execution Marcus actually completed; do not claim work that did not occur.

## Downstream files

- `outputs/dev/<model>/predictions.csv`: paired per-candidate scores and labels
- `outputs/dev/<model>/impression_metrics.csv`: per-impression NDCG@5/MRR and history lengths
- `outputs/dev/metrics.json`: aggregate metrics, fixed settings, and training metadata
- `outputs/dev/comparisons.json`: all paired user-bootstrap comparisons
- `outputs/dev/beyond_accuracy.json`: diversity, novelty, and candidate-catalogue coverage
```

- [ ] **Step 6: Update task status from evidence**

In `tasks.md`, mark BPR-MF, LightGCN, and export/method-note items complete only if Steps 1–5 passed. Preserve the two-tower item as open and append a concise evidence note with the development command and handoff path. Do not mark final evaluation complete.

- [ ] **Step 7: Run the complete pre-handoff gate**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/ruff check .
.venv/bin/mypy src tests
git diff --check
rg -n -i 'TBD|TODO|insert value|replace me|fill in' docs/marcus-model-handoff.md
git status --short
```

Expected: tests/lint/types/diff pass; the red-flag scan returns no placeholder in the handoff; only the intended documentation/task files are modified; `data/` and `outputs/` remain ignored.

- [ ] **Step 8: Commit the core handoff**

```bash
git add docs/marcus-model-handoff.md tasks.md
git commit -m "docs: hand off collaborative development results"
```

- [ ] **Step 9: Declare the two-tower gate result**

Gate is **open** only if Tasks 1–7 all passed, the full run fits available memory/time, outputs are paired and deterministic, and no final validation was opened. Record `Two-tower gate: open` when all conditions pass. Otherwise record `Two-tower gate: closed` followed by the exact failing command and error on the next line. Commit only that file with `git add docs/marcus-model-handoff.md && git commit -m "docs: record two-tower gate"`.

---

## Core-plan completion audit

Before calling Phase A complete, verify every row has direct evidence:

| Requirement | Evidence |
|---|---|
| Shared impression-conditioned BPR pairs | `tests/test_bpr.py` exact pool/sampler tests and `src/models/bpr.py` |
| Same optimizer/settings | `src/config.py`, shared `train_bpr`, `outputs/dev/metrics.json` |
| BPR-MF contract and cold start | BPR tests plus `outputs/dev/bpr_mf/predictions.csv` |
| Exact LightGCN normalization/propagation | hand-computed tests in `tests/test_lightgcn.py` |
| Propagation-only ablation | zero-layer equivalence test and shared trainer/provider hierarchy |
| No random-vector exposure | trained-evidence mask tests for MF and exact graph-depth expansion |
| No evaluation-label leakage | `Model.score` interface, graph/pair builders using only `TrainingData`, leakage tests |
| Reproducibility | repeated-fit tests plus byte-identical development predictions |
| Four-model paired outputs | artifact audit across identical impression/candidate/label rows |
| Six pairwise comparisons | `outputs/dev/comparisons.json` key audit |
| Beyond-accuracy outputs | finite four-model `outputs/dev/beyond_accuracy.json` audit |
| No final-validation access | absence check and use of `src.run`, never `src.final` |
| Truthful handoff | observed values in `docs/marcus-model-handoff.md`; no final-result wording |
| Truthful AI-use record | exact Codex/Marcus roles recorded for Sood without claiming unperformed review |
| Task tracking | `tasks.md` statuses match passing evidence |

If any evidence is absent, contradictory, or indirect, Phase A remains incomplete and the two-tower plan stays closed.

---

## Downstream freeze boundary (not executed by this plan)

After the team has frozen the model set and analysis, Marcus—not the agent—reviews the implementation and authorizes the held-out run. Only then:

1. Obtain the untouched original MINDsmall development files required by the repository's final-evaluation workflow.
2. Record the frozen revision and verify the working tree is clean.
3. Run `python -m src.final` exactly once, followed by `python -m src.beyond_final` exactly once, without tuning or rerunning from the observed outcomes.
4. Audit that every frozen model has paired final outputs, that all expected pairwise comparisons and beyond-accuracy records are finite, and that the recorded revision matches the freeze.
5. Hand the evidence to Yugan and the truthful AI-use record to Sood; do not submit, upload, post, or message as Marcus.
6. Before Marcus uploads any assessed artifact, scan its visible content and metadata for his name and matric number and resolve any anonymisation failure.

If the two-tower gate remains closed, freeze and validate only the four core models. If it opens and the extension plan passes, freeze all five. Final outcomes must not be used to revise model code, settings, or claims selected from development evidence.
