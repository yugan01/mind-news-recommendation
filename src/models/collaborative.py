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
