import json
import tempfile
import unittest
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src import config
from src.analyze import Prediction
from src.beyond import analyze_beyond, summarize
from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError, ExperimentData
from src.evaluate import evaluate_model
from src.models.popularity import PopularityModel


class BeyondTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vectors = {
            "A": np.array([1, 0], dtype=np.float32),
            "B": np.array([0, 1], dtype=np.float32),
            "C": np.array([-1, 0], dtype=np.float32),
        }

    def test_diversity_novelty_and_short_list(self) -> None:
        row = Prediction("U", ("A", "B", "C"), (1, 0, 0), np.array([3.0, 2.0, 1.0]))
        result = summarize({"I": row}, Counter({"A": 2}), self.vectors, 3)
        self.assertAlmostEqual(result["intra_list_diversity_at_5"], 4 / 3)
        self.assertAlmostEqual(
            result["novelty_bits_at_5"], (-np.log2(3 / 5) - 2 * np.log2(1 / 5)) / 3
        )
        self.assertEqual(result["catalogue_coverage_at_5"], 1)

    def test_user_weighting_and_candidate_catalogue_coverage(self) -> None:
        vectors = {str(index): self.vectors["A"] for index in range(6)}
        row = Prediction("U1", tuple(vectors), (1, 0, 0, 0, 0, 0), np.arange(6.0))
        result = summarize({"I": row}, Counter(), vectors, 6)
        self.assertAlmostEqual(result["catalogue_coverage_at_5"], 5 / 6)
        popular = Prediction("U1", ("A",), (1,), np.array([1.0]))
        rare = Prediction("U2", ("B",), (1,), np.array([1.0]))
        result = summarize(
            {"1": popular, "2": popular, "3": rare}, Counter({"A": 2}), self.vectors, 3
        )
        self.assertAlmostEqual(result["novelty_bits_at_5"], (-np.log2(3 / 5) - np.log2(1 / 5)) / 2)
        self.assertEqual(result["intra_list_diversity_at_5"], 0)

    def test_empty_predictions_and_invalid_embeddings(self) -> None:
        with self.assertRaises(DataError):
            summarize({}, Counter(), self.vectors, 3)
        row = Prediction("U", ("A",), (1,), np.array([1.0]))
        with self.assertRaises(DataError):
            summarize({"I": row}, Counter(), {"A": np.zeros(2, dtype=np.float32)}, 1)

    def test_export_pipeline_and_rejects_stale_examples_and_cache(self) -> None:
        articles = {key: Article(key, "tech", key, "") for key in self.vectors}
        example = Example(
            RankingInput("I", "U", datetime(2019, 11, 15), ("A",), 1, ("A", "B", "C")),
            Targets((1, 0, 0)),
        )
        training = TrainingData(articles, (example,), {"U": 0}, {"A": 0, "B": 1, "C": 2})
        data = ExperimentData(training, (example,))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            cache = output / "vectors.npz"
            np.savez(
                cache,
                article_ids=list(articles),
                texts=list(articles),
                vectors=list(self.vectors.values()),
            )
            model = PopularityModel()
            model.fit(training)
            evaluate_model(model, data.evaluation, output / model.name)
            with patch.multiple(config, SEMANTIC_CACHE=cache, MODELS=(PopularityModel,)):
                analyze_beyond(data, output)
                result = json.loads((output / "beyond_accuracy.json").read_text())
                self.assertEqual(result["popularity"]["impressions"], 1)
                self.assertEqual(
                    set(result["popularity"]),
                    {
                        "users",
                        "impressions",
                        "candidate_catalogue_size",
                        "recommended_articles",
                        "intra_list_diversity_at_5",
                        "novelty_bits_at_5",
                        "catalogue_coverage_at_5",
                    },
                )
                with self.assertRaises(DataError):
                    analyze_beyond(ExperimentData(training, ()), output)
                np.savez(cache, article_ids=["A"], texts=["outdated"], vectors=[self.vectors["A"]])
                with self.assertRaisesRegex(DataError, "outdated"):
                    analyze_beyond(data, output)


if __name__ == "__main__":
    unittest.main()
