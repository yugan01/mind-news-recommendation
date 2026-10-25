"""Shared training and label-free scoring contracts for every recommender."""

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Article:
    article_id: str
    category: str
    title: str
    abstract: str

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass(frozen=True)
class RankingInput:
    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    available_history_length: int
    candidate_ids: tuple[str, ...]


@dataclass(frozen=True)
class Targets:
    labels: tuple[int, ...]


@dataclass(frozen=True)
class Example:
    inputs: RankingInput
    targets: Targets


@dataclass(frozen=True)
class TrainingData:
    articles: Mapping[str, Article]
    examples: tuple[Example, ...]
    user_indices: Mapping[str, int]
    article_indices: Mapping[str, int]


class Model(Protocol):
    """Fit only on TrainingData; score without access to held-out targets.

    Scores must be finite, one-dimensional, and aligned with candidate_ids.
    Model-specific fitting algorithms and cold-start policies belong to each
    implementation and must be documented with the model.
    """

    @property
    def name(self) -> str: ...

    def fit(self, data: TrainingData) -> None: ...

    def score(self, inputs: RankingInput) -> NDArray[np.float64]: ...
