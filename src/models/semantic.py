"""Frozen article embeddings; unseen users need only their supplied history.

All history/candidate articles must be in the shared metadata corpus. Empty or
zero-vector histories receive tied zero scores, preserving candidate order.
"""

import numpy as np
from numpy.typing import NDArray

from src.contracts import RankingInput, TrainingData
from src.text_embeddings import encode_articles, load_article_vectors


class SemanticModel:
    @property
    def name(self) -> str:
        return "semantic"

    def __init__(self) -> None:
        self.article_indices: dict[str, int] = {}
        self.vectors: NDArray[np.float32] = np.empty((0, 768), dtype=np.float32)
        self.run_metadata: dict[str, object] = {}

    def fit(self, data: TrainingData) -> None:
        from src import config

        self.run_metadata = {}
        self.article_indices, self.vectors = load_article_vectors(
            data.articles, config.SEMANTIC_CACHE, encode_articles
        )
        self.run_metadata = {
            "device": "cpu",
            "frozen_encoder": config.SEMANTIC_ENCODER,
            "article_vectors": len(self.article_indices),
        }

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if not inputs.history:
            return np.zeros(len(inputs.candidate_ids), dtype=np.float64)
        history = [self.article_indices[article] for article in inputs.history]
        profile = self.vectors[history].mean(axis=0)
        norm = float(np.linalg.norm(profile))
        if norm == 0:
            return np.zeros(len(inputs.candidate_ids), dtype=np.float64)
        candidates = [self.article_indices[article] for article in inputs.candidate_ids]
        return np.asarray(self.vectors[candidates] @ (profile / norm), dtype=np.float64)
