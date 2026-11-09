"""Rank by pre-evaluation training clicks; unseen articles receive zero."""

from collections import Counter

import numpy as np
from numpy.typing import NDArray

from src.contracts import RankingInput, TrainingData


class PopularityModel:
    @property
    def name(self) -> str:
        return "popularity"

    def __init__(self) -> None:
        self.clicks: Counter[str] = Counter()
        self.run_metadata: dict[str, object] = {"device": "not_applicable"}

    def fit(self, data: TrainingData) -> None:
        self.clicks.clear()
        self.run_metadata = {"device": "not_applicable"}
        for example in data.examples:
            self.clicks.update(
                article_id
                for article_id, label in zip(
                    example.inputs.candidate_ids, example.targets.labels, strict=True
                )
                if label == 1
            )

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        return np.asarray(
            [self.clicks[article_id] for article_id in inputs.candidate_ids], dtype=float
        )
