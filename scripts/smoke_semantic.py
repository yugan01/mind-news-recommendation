"""Tiny real-encoder functionality check: python -m scripts.smoke_semantic.

Uses synthetic articles and impressions, never MIND validation. The resulting
metrics check the export path only; they are not experiment results.
"""

from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np

from src import config
from src.contracts import Article
from src.data import ExperimentData, Impression, make_example, make_training_data
from src.evaluate import evaluate_model
from src.models.semantic import SemanticModel
from src.run import run_experiment


def main() -> None:
    articles = {
        "SMOKE_A": Article(
            "SMOKE_A", "tech", "Electric car batteries", "New batteries extend electric car range."
        ),
        "SMOKE_B": Article(
            "SMOKE_B",
            "tech",
            "Charging electric vehicles",
            "A new network of car charging stations.",
        ),
        "SMOKE_C": Article(
            "SMOKE_C", "sport", "Football championship", "The football team wins the league final."
        ),
        "SMOKE_D": Article(
            "SMOKE_D",
            "sport",
            "Football transfers",
            "The club signs a new striker for next season.",
        ),
    }
    timestamp = datetime(2019, 11, 14)
    training = make_training_data(
        articles,
        [Impression("T1", "U1", timestamp, ("SMOKE_A",) * 10, ("SMOKE_B", "SMOKE_D"), (1, 0))],
    )
    evaluation = tuple(
        make_example(
            Impression(
                f"E{index}",
                user,
                timestamp + timedelta(days=1),
                (history,) * 10,
                ("SMOKE_B", "SMOKE_D"),
                labels,
            )
        )
        for index, (user, history, labels) in enumerate(
            [("U1", "SMOKE_A", (1, 0)), ("unseen-user", "SMOKE_C", (0, 1))]
        )
    )
    # Keep synthetic vectors separate from the real experiment's article cache.
    with patch.object(config, "SEMANTIC_CACHE", config.SMOKE_OUTPUT_DIR / "vectors.npz"):
        run_experiment(ExperimentData(training, evaluation), config.SMOKE_OUTPUT_DIR)
        with patch("src.models.semantic.encode_articles", side_effect=AssertionError("Cache miss")):
            model = SemanticModel()
            model.fit(training)
        assert model.vectors.shape == (4, 768)
        assert np.isfinite(model.vectors).all()
        np.testing.assert_allclose(np.linalg.norm(model.vectors, axis=1), 1, atol=1e-5)
        evaluate_model(model, evaluation, config.SMOKE_OUTPUT_DIR / "semantic_cached")
        original = config.SMOKE_OUTPUT_DIR / "semantic/predictions.csv"
        cached = config.SMOKE_OUTPUT_DIR / "semantic_cached/predictions.csv"
        assert original.read_bytes() == cached.read_bytes()
    print("Smoke check passed: four articles, two impressions, exports and cache reuse.")
    print("Synthetic smoke metrics are not reportable experiment results.")


if __name__ == "__main__":
    main()
