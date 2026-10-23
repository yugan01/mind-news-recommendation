"""Evaluate models on a fixed cohort without exposing targets to scoring."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from src.contracts import Example, Model
from src.data import DataError
from src.metrics import (
    ndcg_at_k,
    ranking_order,
    reciprocal_rank,
    summarize_users,
)


def evaluate_model(
    model: Model,
    examples: Sequence[Example],
    output_dir: Path,
) -> dict[str, int | float]:
    if not examples:
        raise DataError("Cannot evaluate an empty cohort")
    per_user: dict[str, list[tuple[float, float]]] = defaultdict(list)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (
        (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as predictions,
        (output_dir / "impression_metrics.csv").open("w", encoding="utf-8", newline="") as metrics,
    ):
        writer = csv.writer(predictions)
        metric_writer = csv.writer(metrics)
        writer.writerow(
            ["impression_id", "user_id", "candidate_index", "article_id", "score", "rank", "label"]
        )
        metric_writer.writerow(
            [
                "impression_id",
                "user_id",
                "timestamp",
                "available_history_length",
                "used_history_length",
                "ndcg_at_5",
                "mrr",
            ]
        )
        for example in examples:
            inputs = example.inputs
            scores = np.asarray(model.score(example.inputs), dtype=float)
            if (
                scores.shape != (len(example.inputs.candidate_ids),)
                or not np.isfinite(scores).all()
            ):
                raise DataError(
                    f"{model.name}: invalid scores for impression {inputs.impression_id}"
                )
            order = ranking_order(scores)
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            # Only the evaluator accesses held-out labels, after the model has returned scores.
            labels = example.targets.labels
            ndcg = ndcg_at_k(labels, scores, 5)
            mrr = reciprocal_rank(labels, scores)
            for index, article_id in enumerate(example.inputs.candidate_ids):
                writer.writerow(
                    [
                        inputs.impression_id,
                        inputs.user_id,
                        index,
                        article_id,
                        repr(float(scores[index])),
                        int(ranks[index]),
                        labels[index],
                    ]
                )
            metric_writer.writerow(
                [
                    inputs.impression_id,
                    inputs.user_id,
                    inputs.timestamp.isoformat(),
                    example.inputs.available_history_length,
                    len(example.inputs.history),
                    ndcg,
                    mrr,
                ]
            )
            per_user[inputs.user_id].append((ndcg, mrr))
    return summarize_users(per_user)
