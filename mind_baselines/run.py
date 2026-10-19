"""Command-line runner for MIND-small baseline experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from collections import defaultdict

from .data import iter_eligible_impressions, sample_eligible_users
from .metrics import ndcg_at_k, reciprocal_rank
from .models import PopularityModel


def evaluate_model(
    name: str,
    model: object,
    behaviors_path: Path,
    output_dir: Path,
    min_history: int,
    selected_users: set[str] | None,
) -> dict:
    user_metrics: dict[str, list[tuple[float, float]]] = defaultdict(list)
    impression_count = 0
    output_path = output_dir / f"{name}_predictions.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["impression_id", "user_id", "article_id", "score", "rank", "label"])
        for impression in iter_eligible_impressions(
            behaviors_path,
            min_history=min_history,
            selected_users=selected_users,
        ):
            scores = np.asarray(model.score(impression), dtype=float)
            labels = [candidate.label for candidate in impression.candidates]
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(len(order), dtype=int)
            ranks[order] = np.arange(1, len(order) + 1)
            for candidate, score, rank in zip(impression.candidates, scores, ranks):
                writer.writerow(
                    [
                        impression.impression_id,
                        impression.user_id,
                        candidate.article_id,
                        f"{score:.10g}",
                        int(rank),
                        candidate.label,
                    ]
                )
            user_metrics[impression.user_id].append(
                (ndcg_at_k(labels, scores, 5), reciprocal_rank(labels, scores))
            )
            impression_count += 1
    user_ndcg = [float(np.mean([row[0] for row in rows])) for rows in user_metrics.values()]
    user_mrr = [float(np.mean([row[1] for row in rows])) for rows in user_metrics.values()]
    return {
        "users": len(user_metrics),
        "impressions": impression_count,
        "ndcg_at_5": float(np.mean(user_ndcg)),
        "mrr": float(np.mean(user_mrr)),
    }


def resolve_dataset_files(train_dir: Path, eval_dir: Path) -> tuple[Path, Path, Path, Path]:
    reczoo_news = train_dir / "news_corpus.tsv"
    if reczoo_news.exists():
        return reczoo_news, reczoo_news, train_dir / "train.csv", eval_dir / "valid.csv"
    return (
        train_dir / "news.tsv",
        eval_dir / "news.tsv",
        train_dir / "behaviors.tsv",
        eval_dir / "behaviors.tsv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--sample-users", type=int)
    parser.add_argument("--seed", type=int, default=4285)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, _, train_behaviors, eval_behaviors = resolve_dataset_files(
        args.train_dir, args.eval_dir
    )
    selected_users = None
    if args.sample_users is not None:
        selected_users = sample_eligible_users(
            eval_behaviors,
            args.sample_users,
            min_history=args.min_history,
            seed=args.seed,
        )
        if not selected_users:
            raise SystemExit("No eligible users found; check the paths and filters.")

    models = {
        "popularity": PopularityModel().fit(train_behaviors),
    }
    metrics = {
        name: evaluate_model(
            name,
            model,
            eval_behaviors,
            args.output_dir,
            args.min_history,
            selected_users,
        )
        for name, model in models.items()
    }
    metrics["configuration"] = {
        "min_history": args.min_history,
        "max_history": args.max_history,
        "sample_users": args.sample_users,
        "seed": args.seed,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

