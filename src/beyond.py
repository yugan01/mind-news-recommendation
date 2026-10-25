"""Beyond-accuracy metrics from development exports: python -m src.beyond."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from src import config
from src.analyze import Prediction, read_predictions
from src.data import DataError, ExperimentData, load_development
from src.metrics import ranking_order


def summarize(
    predictions: Mapping[str, Prediction],
    clicks: Counter[str],
    vectors: Mapping[str, NDArray[np.float32]],
    article_count: int,
) -> dict[str, int | float]:
    """Top-5 metrics; the coverage catalogue is the evaluation candidate union."""
    if not predictions:
        raise DataError("Beyond-accuracy metrics require nonempty predictions")
    catalogue = {article for row in predictions.values() for article in row.article_ids}
    recommended: set[str] = set()
    per_user: dict[str, list[tuple[float, float]]] = defaultdict(list)
    denominator = clicks.total() + article_count
    for row in predictions.values():
        top = [row.article_ids[index] for index in ranking_order(row.scores)[:5]]
        embeddings = np.asarray([vectors[article] for article in top], dtype=np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or np.any(norms == 0):
            raise DataError("Diversity requires finite, nonzero article embeddings")
        embeddings /= norms
        distances = 1 - np.clip(embeddings @ embeddings.T, -1, 1)
        pairs = distances[np.triu_indices(len(top), k=1)]
        diversity = float(pairs.mean()) if len(pairs) else 0.0
        novelty = float(np.mean([-np.log2((clicks[article] + 1) / denominator) for article in top]))
        per_user[row.user_id].append((diversity, novelty))
        recommended.update(top)
    averages = np.asarray([np.mean(rows, axis=0) for rows in per_user.values()]).mean(axis=0)
    return {
        "users": len(per_user),
        "impressions": len(predictions),
        "candidate_catalogue_size": len(catalogue),
        "recommended_articles": len(recommended),
        "intra_list_diversity_at_5": float(averages[0]),
        "novelty_bits_at_5": float(averages[1]),
        "catalogue_coverage_at_5": len(recommended) / len(catalogue),
    }


def analyze_beyond(data: ExperimentData, output_dir: Path) -> None:
    """Read existing exports/cache only; never load an encoder or fit a model."""
    if not config.SEMANTIC_CACHE.exists():
        raise DataError("Run the semantic model first to create its article vector cache")
    clicks: Counter[str] = Counter()
    for example in data.training.examples:
        clicks.update(
            article
            for article, label in zip(
                example.inputs.candidate_ids, example.targets.labels, strict=True
            )
            if label == 1
        )
    with np.load(config.SEMANTIC_CACHE, allow_pickle=False) as cache:
        vectors = dict(zip(cache["article_ids"].tolist(), cache["vectors"], strict=True))
        texts = dict(zip(cache["article_ids"].tolist(), cache["texts"].tolist(), strict=True))
    expected = {example.inputs.impression_id: example for example in data.evaluation}
    results: dict[str, dict[str, int | float]] = {}
    for model_type in config.MODELS:
        name = model_type().name
        predictions = read_predictions(output_dir / name / "predictions.csv")
        if predictions.keys() != expected.keys():
            raise DataError(f"{name}: predictions do not match the evaluation impression set")
        for impression, row in predictions.items():
            example = expected[impression]
            if (row.user_id, row.article_ids, row.labels) != (
                example.inputs.user_id,
                example.inputs.candidate_ids,
                example.targets.labels,
            ):
                raise DataError(
                    f"{name}: predictions disagree with evaluation example {impression}"
                )
            for article in row.article_ids:
                if article not in vectors or texts[article] != data.training.articles[article].text:
                    raise DataError(f"Missing or outdated cached embedding for {article}")
        results[name] = summarize(predictions, clicks, vectors, len(data.training.articles))
    (output_dir / "beyond_accuracy.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Beyond-accuracy metrics written for {len(results)} models.")


if __name__ == "__main__":
    analyze_beyond(load_development(), config.DEV_OUTPUT_DIR)
