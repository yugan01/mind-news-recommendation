import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from src import config
from src.contracts import Article, RankingInput, TrainingData
from src.models.semantic import SemanticModel


class SemanticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.cache = patch.object(config, "SEMANTIC_CACHE", Path(self.temp.name) / "vectors.npz")
        self.cache.start()
        self.articles = {
            "A": Article("A", "tech", "First title", "First abstract"),
            "B": Article("B", "sport", "Second title", ""),
            "C": Article("C", "tech", "Third title", "Third abstract"),
        }
        self.data = TrainingData(self.articles, (), {}, {"A": 0, "B": 1, "C": 2})
        self.vectors: NDArray[np.float32] = np.zeros((3, 768), dtype=np.float32)
        self.vectors[0, 0] = 1
        self.vectors[1, 1] = 1
        self.vectors[2, 0] = -1
        self.inputs = RankingInput(
            "I1", "unseen-user", datetime(2019, 11, 15), ("A", "A", "B"), 3, ("B", "C", "A")
        )

    def tearDown(self) -> None:
        self.cache.stop()
        self.temp.cleanup()

    def test_title_abstract_encoding_and_cosine_of_mean_history(self) -> None:
        model = SemanticModel()
        with patch("src.models.semantic.encode_articles", return_value=self.vectors) as encode:
            model.fit(self.data)
        encode.assert_called_once_with(
            ["First title First abstract", "Second title", "Third title Third abstract"]
        )
        np.testing.assert_allclose(
            model.score(self.inputs), np.array([1, -2, 2]) / np.sqrt(5), rtol=1e-6
        )
        self.assertEqual(model.score(self.inputs).dtype, np.float64)

    def test_cache_reused_without_encoder_and_refit_replaces_state(self) -> None:
        with patch("src.models.semantic.encode_articles", return_value=self.vectors) as encode:
            SemanticModel().fit(self.data)
            model = SemanticModel()
            model.fit(self.data)
            model.fit(self.data)
        encode.assert_called_once()
        self.assertEqual(model.article_indices, {"A": 0, "B": 1, "C": 2})
        np.testing.assert_array_equal(model.vectors, self.vectors)

    def test_only_new_or_changed_articles_are_encoded(self) -> None:
        model = SemanticModel()
        with patch("src.models.semantic.encode_articles", return_value=self.vectors):
            model.fit(self.data)
        articles = dict(self.articles)
        articles["B"] = replace(articles["B"], abstract="Updated")
        articles["D"] = Article("D", "tech", "New title", "New abstract")
        with patch("src.models.semantic.encode_articles", return_value=self.vectors[:2]) as encode:
            model.fit(replace(self.data, articles=articles))
        encode.assert_called_once_with(["Second title Updated", "New title New abstract"])
        np.testing.assert_array_equal(model.vectors[[0, 2]], self.vectors[[0, 2]])
        self.assertEqual(model.article_indices["D"], 3)

    def test_empty_or_cancelling_history_has_zero_scores(self) -> None:
        model = SemanticModel()
        with patch("src.models.semantic.encode_articles", return_value=self.vectors):
            model.fit(self.data)
        for history in ((), ("A", "C")):
            np.testing.assert_array_equal(
                model.score(replace(self.inputs, history=history)), np.zeros(3)
            )


if __name__ == "__main__":
    unittest.main()
