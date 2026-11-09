"""Task-learned two-tower recommender over frozen text vectors."""

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from src import config
from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import (
    PairPool,
    SampledPairs,
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
    """Mean-pool frozen text vectors for each fitting impression's history."""
    features = np.zeros((len(data.examples), vectors.shape[1]), dtype=np.float32)
    for row, example in enumerate(data.examples):
        try:
            indices = [vector_indices[article] for article in example.inputs.history]
        except KeyError as error:
            raise DataError(
                f"Two-tower history is missing article {error.args[0]}"
            ) from error
        if indices:
            features[row] = vectors[indices].mean(axis=0)
    return features


class TwoTowerNetwork(nn.Module):
    """Separate user and article projection networks."""

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
        return cast(Tensor, self.user_tower(features))

    def article_vectors(self, features: Tensor) -> Tensor:
        return cast(Tensor, self.article_tower(features))


def _nonempty_history_pool(data: TrainingData, pool: PairPool) -> tuple[PairPool, int]:
    eligible_examples = np.asarray(
        [bool(example.inputs.history) for example in data.examples], dtype=np.bool_
    )
    keep = eligible_examples[pool.example_indices]
    skipped_examples = {
        int(index) for index in pool.example_indices[~keep]
    }
    if not keep.any():
        raise DataError("Two-tower training contains no usable nonempty-history pairs")
    positions = np.flatnonzero(keep)
    return (
        PairPool(
            pool.user_indices[keep],
            pool.positive_indices[keep],
            pool.example_indices[keep],
            tuple(pool.negative_options[int(index)] for index in positions),
            pool.included_impressions - len(skipped_examples),
            pool.skipped_impressions,
        ),
        len(skipped_examples),
    )


def two_tower_objective(
    network: TwoTowerNetwork,
    history_feature_table: Tensor,
    article_feature_table: Tensor,
    pairs: SampledPairs,
    l2: float,
) -> Tensor:
    """Project each feature table once, then gather the sampled training pairs."""
    users = network.user_vectors(history_feature_table)
    articles = network.article_vectors(article_feature_table)
    ranking = pairwise_bpr_loss(
        users[pairs.example_indices],
        articles[pairs.positive_indices],
        articles[pairs.negative_indices],
    )
    penalty = torch.stack(
        [parameter.square().mean() for parameter in network.parameters()]
    ).sum()
    return ranking + l2 * penalty


@dataclass(frozen=True)
class TwoTowerTrainingResult:
    network: TwoTowerNetwork
    losses: tuple[float, ...]
    device: str


def train_two_tower(
    network: TwoTowerNetwork,
    histories: NDArray[np.float32],
    articles: NDArray[np.float32],
    pool: PairPool,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    seed: int,
) -> TwoTowerTrainingResult:
    """Train deterministically with the shared impression-conditioned BPR pairs."""
    if epochs <= 0:
        raise ValueError("Two-tower epochs must be positive")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        device = training_device()
        network = network.to(device)
        history_tensor = torch.as_tensor(histories, dtype=torch.float32, device=device)
        article_tensor = torch.as_tensor(articles, dtype=torch.float32, device=device)
        optimizer = torch.optim.Adam(
            network.parameters(), lr=learning_rate, weight_decay=0.0
        )
        losses: list[float] = []
        for epoch in range(epochs):
            pairs = sample_pairs(pool, epoch, seed, device)
            optimizer.zero_grad(set_to_none=True)
            loss = two_tower_objective(
                network, history_tensor, article_tensor, pairs, l2
            )
            if not torch.isfinite(loss):
                raise DataError(f"Two-tower produced non-finite loss at epoch {epoch}")
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        network.eval()
        with torch.no_grad():
            projected_histories = network.user_vectors(history_tensor)
            projected_articles = network.article_vectors(article_tensor)
            if not torch.isfinite(projected_histories).all() or not torch.isfinite(
                projected_articles
            ).all():
                raise DataError("Two-tower produced non-finite embeddings")
        return TwoTowerTrainingResult(network.to("cpu"), tuple(losses), str(device))
    finally:
        torch.use_deterministic_algorithms(previous_determinism)


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
        original_pool = build_pair_pool(data)
        pool, skipped_empty_histories = _nonempty_history_pool(data, original_pool)
        article_features = np.empty((len(data.article_indices), 768), dtype=np.float32)
        for article, index in data.article_indices.items():
            try:
                article_features[index] = self.vectors[self.vector_indices[article]]
            except KeyError as error:
                raise DataError(
                    f"Two-tower candidates are missing article {error.args[0]}"
                ) from error
        histories = history_features(data, self.vectors, self.vector_indices)
        result = train_two_tower(
            TwoTowerNetwork(
                768,
                config.TWO_TOWER_HIDDEN,
                config.TWO_TOWER_OUTPUT,
                config.SEED,
            ),
            histories,
            article_features,
            pool,
            epochs=config.TWO_TOWER_EPOCHS,
            learning_rate=config.TWO_TOWER_LEARNING_RATE,
            l2=config.TWO_TOWER_L2,
            seed=config.SEED,
        )
        self.network = result.network
        self.run_metadata = {
            "device": result.device,
            "epochs": len(result.losses),
            "losses": result.losses,
            "final_loss": result.losses[-1],
            "pair_count": len(pool.user_indices),
            "included_impressions": pool.included_impressions,
            "skipped_impressions": pool.skipped_impressions,
            "skipped_empty_histories": skipped_empty_histories,
            "input_dimension": 768,
            "architecture": (
                f"separate 768->{config.TWO_TOWER_HIDDEN}"
                f"->{config.TWO_TOWER_OUTPUT} towers"
            ),
            "optimizer": "Adam",
            "seed": config.SEED,
            "learning_rate": config.TWO_TOWER_LEARNING_RATE,
            "l2": config.TWO_TOWER_L2,
            "frozen_encoder": config.SEMANTIC_ENCODER,
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
            raise DataError(
                f"Two-tower score is missing article {error.args[0]}"
            ) from error
        history_feature = torch.as_tensor(
            self.vectors[history].mean(axis=0)[None, :], dtype=torch.float32
        )
        candidate_features = torch.as_tensor(
            self.vectors[candidates], dtype=torch.float32
        )
        with torch.no_grad():
            user = self.network.user_vectors(history_feature)[0]
            articles = self.network.article_vectors(candidate_features)
            scores = (articles * user).sum(dim=1).numpy().astype(np.float64)
        if not np.isfinite(scores).all():
            raise DataError(f"Two-tower produced non-finite scores for {inputs.impression_id}")
        return scores
