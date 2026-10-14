"""Compare two prediction exports with paired user-level bootstrap intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

from .metrics import ndcg_at_k, reciprocal_rank


def user_metrics(path: Path) -> dict[str, dict[str, float]]:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for _, rows_iter in groupby(reader, key=lambda row: row["impression_id"]):
            rows = list(rows_iter)
            user_id = rows[0]["user_id"]
            labels = [int(row["label"]) for row in rows]
            scores = [float(row["score"]) for row in rows]
            totals[user_id][0] += ndcg_at_k(labels, scores, 5)
            totals[user_id][1] += reciprocal_rank(labels, scores)
            totals[user_id][2] += 1
    return {
        user_id: {
            "ndcg_at_5": values[0] / values[2],
            "mrr": values[1] / values[2],
        }
        for user_id, values in totals.items()
    }


def bootstrap_interval(
    differences: np.ndarray,
    iterations: int,
    seed: int,
    batch_size: int = 250,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, iterations, batch_size):
        size = min(batch_size, iterations - start)
        indices = rng.integers(0, len(differences), size=(size, len(differences)))
        estimates.append(differences[indices].mean(axis=1))
    samples = np.concatenate(estimates)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def compare(
    baseline: dict[str, dict[str, float]],
    challenger: dict[str, dict[str, float]],
    iterations: int,
    seed: int,
) -> dict[str, object]:
    users = sorted(set(baseline) & set(challenger))
    result: dict[str, object] = {"paired_users": len(users), "metrics": {}}
    for metric in ("ndcg_at_5", "mrr"):
        differences = np.asarray(
            [challenger[user][metric] - baseline[user][metric] for user in users]
        )
        low, high = bootstrap_interval(differences, iterations, seed)
        result["metrics"][metric] = {
            "mean_paired_difference": float(differences.mean()),
            "ci_95": [low, high],
            "users_improved_fraction": float(np.mean(differences > 0)),
            "users_tied_fraction": float(np.mean(differences == 0)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=4285)
    args = parser.parse_args()

    result = compare(
        user_metrics(args.baseline),
        user_metrics(args.challenger),
        args.iterations,
        args.seed,
    )
    result["configuration"] = {
        "baseline": str(args.baseline),
        "challenger": str(args.challenger),
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

