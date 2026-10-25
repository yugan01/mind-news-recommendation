"""Shared deterministic BPR pairs, objective, and optimization."""

from dataclasses import dataclass
from typing import cast

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


def sample_pairs(
    pool: PairPool, epoch: int, seed: int, device: torch.device
) -> SampledPairs:
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
            self.article_embeddings.weight.normal_(
                0.0, config.BPR_INIT_STD, generator=generator
            )

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
    return cast(Tensor, ranking + l2 * penalty)


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
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        device = training_device()
        provider = provider.to(device)
        optimizer = torch.optim.Adam(
            provider.parameters(), lr=learning_rate, weight_decay=0.0
        )
        trained_users = np.zeros(provider.user_embeddings.num_embeddings, dtype=np.bool_)
        trained_articles = np.zeros(
            provider.article_embeddings.num_embeddings, dtype=np.bool_
        )
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
            loss.backward()  # type: ignore[no-untyped-call]
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
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
