"""Compare every pair of development models: python -m src.analyze."""

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src import config
from src.data import DataError
from src.metrics import ndcg_at_k, reciprocal_rank


@dataclass(frozen=True)
class Prediction:
    user_id: str
    article_ids: tuple[str, ...]
    labels: tuple[int, ...]
    scores: NDArray[np.float64]


def read_predictions(path: Path) -> dict[str, Prediction]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["impression_id"]].append(row)
    predictions: dict[str, Prediction] = {}
    for impression_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["candidate_index"]))
        if (
            [int(row["candidate_index"]) for row in rows] != list(range(len(rows)))
            or len({row["user_id"] for row in rows}) != 1
            or len({row["article_id"] for row in rows}) != len(rows)
        ):
            raise DataError(f"Invalid prediction group {impression_id}")
        scores = np.asarray([float(row["score"]) for row in rows])
        labels = tuple(int(row["label"]) for row in rows)
        if not np.isfinite(scores).all() or set(labels) != {0, 1}:
            raise DataError(f"Invalid scores or labels for {impression_id}")
        predictions[impression_id] = Prediction(
            rows[0]["user_id"],
            tuple(row["article_id"] for row in rows),
            labels,
            scores,
        )
    return predictions


def bootstrap_interval(differences: NDArray[np.float64]) -> tuple[float, float]:
    rng = np.random.default_rng(config.SEED)
    samples = np.empty(config.BOOTSTRAP_ITERATIONS)
    for index in range(config.BOOTSTRAP_ITERATIONS):
        samples[index] = rng.choice(differences, size=len(differences), replace=True).mean()
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def compare(
    baseline: dict[str, Prediction],
    challenger: dict[str, Prediction],
) -> dict[str, object]:
    if not baseline or baseline.keys() != challenger.keys():
        raise DataError("Models must have identical nonempty impression sets")
    paired: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for impression_id in sorted(baseline):
        left, right = baseline[impression_id], challenger[impression_id]
        if (left.user_id, left.article_ids, left.labels) != (
            right.user_id,
            right.article_ids,
            right.labels,
        ):
            raise DataError(f"Models disagree on user, candidates or labels for {impression_id}")
        paired[left.user_id].append(
            (
                ndcg_at_k(right.labels, right.scores, 5) - ndcg_at_k(left.labels, left.scores, 5),
                reciprocal_rank(right.labels, right.scores)
                - reciprocal_rank(left.labels, left.scores),
            )
        )
    differences = np.asarray([np.mean(paired[user], axis=0) for user in sorted(paired)])
    metrics: dict[str, object] = {}
    for index, metric in enumerate(("ndcg_at_5", "mrr")):
        values = differences[:, index]
        metrics[metric] = {
            "mean_paired_difference": float(values.mean()),
            "ci_95": bootstrap_interval(values),
            "users_improved_fraction": float(np.mean(values > 0)),
            "users_tied_fraction": float(np.mean(values == 0)),
        }
    return {"paired_users": len(paired), "paired_impressions": len(baseline), "metrics": metrics}


def analyze(output_dir: Path) -> None:
    names = [model().name for model in config.MODELS]
    results = {
        f"{challenger}_minus_{baseline}": compare(
            read_predictions(output_dir / baseline / "predictions.csv"),
            read_predictions(output_dir / challenger / "predictions.csv"),
        )
        for baseline, challenger in combinations(names, 2)
    }
    (output_dir / "comparisons.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Compared {len(results)} model pairs.")


if __name__ == "__main__":
    analyze(config.DEV_OUTPUT_DIR)
