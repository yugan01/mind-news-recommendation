"""Ranking metrics with user-level aggregation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def ndcg_at_k(labels: Iterable[int], scores: Iterable[float], k: int = 5) -> float:
    labels_array = np.asarray(list(labels), dtype=np.int8)
    scores_array = np.asarray(list(scores), dtype=float)
    order = np.argsort(-scores_array, kind="stable")[:k]
    gains = labels_array[order]
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.sort(labels_array)[::-1][:k]
    idcg = float(np.sum(ideal * discounts[: len(ideal)]))
    return dcg / idcg if idcg else 0.0


def reciprocal_rank(labels: Iterable[int], scores: Iterable[float]) -> float:
    labels_array = np.asarray(list(labels), dtype=np.int8)
    scores_array = np.asarray(list(scores), dtype=float)
    for rank, index in enumerate(np.argsort(-scores_array, kind="stable"), start=1):
        if labels_array[index] == 1:
            return 1.0 / rank
    return 0.0


def aggregate_by_user(records: Iterable[dict[str, object]]) -> dict[str, float]:
    per_user: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ndcg_at_5": [], "mrr": []}
    )
    count = 0
    for record in records:
        user_id = str(record["user_id"])
        labels = record["labels"]
        scores = record["scores"]
        per_user[user_id]["ndcg_at_5"].append(ndcg_at_k(labels, scores, 5))
        per_user[user_id]["mrr"].append(reciprocal_rank(labels, scores))
        count += 1

    user_ndcg = [float(np.mean(values["ndcg_at_5"])) for values in per_user.values()]
    user_mrr = [float(np.mean(values["mrr"])) for values in per_user.values()]
    return {
        "users": len(per_user),
        "impressions": count,
        "ndcg_at_5": float(np.mean(user_ndcg)) if user_ndcg else math.nan,
        "mrr": float(np.mean(user_mrr)) if user_mrr else math.nan,
    }

