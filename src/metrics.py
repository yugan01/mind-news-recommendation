"""Stable candidate-order tie handling and user-weighted ranking metrics."""

from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def ranking_order(scores: NDArray[np.float64]) -> NDArray[np.intp]:
    return np.argsort(-scores, kind="stable")


def ndcg_at_k(labels: Sequence[int], scores: NDArray[np.float64], k: int) -> float:
    labels_array = np.asarray(labels, dtype=np.int8)
    order = ranking_order(scores)[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(labels_array[order] * discounts))
    ideal = np.sort(labels_array)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(labels: Sequence[int], scores: NDArray[np.float64]) -> float:
    for rank, index in enumerate(ranking_order(scores), start=1):
        if labels[index] == 1:
            return 1.0 / rank
    return 0.0


def summarize_users(
    per_user: Mapping[str, Sequence[tuple[float, float]]],
) -> dict[str, int | float]:
    if not per_user:
        raise ValueError("Cannot summarize an empty evaluation")
    averages = np.asarray([np.mean(values, axis=0) for values in per_user.values()])
    return {
        "users": len(per_user),
        "impressions": sum(len(values) for values in per_user.values()),
        "ndcg_at_5": float(averages[:, 0].mean()),
        "mrr": float(averages[:, 1].mean()),
    }
