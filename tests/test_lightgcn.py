import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from src import config
from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError
from src.models.bpr import EmbeddingProvider
from src.models.lightgcn import (
    LightGCNEmbeddings,
    LightGCNModel,
    build_graph,
    expand_masks,
)


def graph_fixture() -> TrainingData:
    articles = {
        article: Article(article, "topic", article, "")
        for article in ("A", "B", "C", "D")
    }
    examples = (
        Example(
            RankingInput("I1", "U1", datetime(2019, 1, 1), ("A",), 1, ("B", "C")),
            Targets((1, 0)),
        ),
        Example(
            RankingInput("I2", "U2", datetime(2019, 1, 2), ("B",), 1, ("A", "C")),
            Targets((1, 0)),
        ),
    )
    return TrainingData(
        articles,
        examples,
        {"U1": 0, "U2": 1},
        {"A": 0, "B": 1, "C": 2, "D": 3},
    )


class LightGCNTests(unittest.TestCase):
    def test_graph_collapses_edges_and_normalizes_degrees(self) -> None:
        graph = build_graph(graph_fixture())
        dense = graph.adjacency.to_dense().numpy()
        expected = np.zeros((6, 6), dtype=np.float32)
        weight = 0.5
        for user, article in ((0, 2), (0, 3), (1, 2), (1, 3)):
            expected[user, article] = weight
            expected[article, user] = weight
        np.testing.assert_allclose(dense, expected, atol=1e-7)

    def test_one_layer_propagation_and_zero_layer_ablation(self) -> None:
        graph = build_graph(graph_fixture())
        zero = LightGCNEmbeddings(2, 4, 1, 4285, graph, layers=0)
        one = LightGCNEmbeddings(2, 4, 1, 4285, graph, layers=1)
        with torch.no_grad():
            values = torch.tensor([[1.0], [3.0], [2.0], [4.0], [8.0], [16.0]])
            for provider in (zero, one):
                provider.user_embeddings.weight[:] = values[:2]
                provider.article_embeddings.weight[:] = values[2:]
        raw = EmbeddingProvider(2, 4, 1, 4285)
        raw.load_state_dict(zero.state_dict(), strict=False)
        raw_users, raw_articles = raw()
        zero_users, zero_articles = zero()
        np.testing.assert_allclose(zero_users.detach().numpy(), raw_users.detach().numpy())
        np.testing.assert_allclose(zero_articles.detach().numpy(), raw_articles.detach().numpy())
        one_users, one_articles = one()
        np.testing.assert_allclose(one_users.detach().numpy().ravel(), [2.0, 3.0])
        np.testing.assert_allclose(
            one_articles.detach().numpy().ravel(), [2.0, 3.0, 4.0, 8.0]
        )

    def test_trained_masks_expand_exactly_two_hops(self) -> None:
        graph = build_graph(graph_fixture())
        users = np.asarray([True, False])
        articles = np.asarray([False, False, True, False])
        expanded_users, expanded_articles = expand_masks(graph, users, articles, layers=2)
        np.testing.assert_array_equal(expanded_users, [True, True])
        np.testing.assert_array_equal(expanded_articles, [True, True, True, False])

    def test_model_is_deterministic_and_finite(self) -> None:
        data = graph_fixture()
        with patch.multiple(config, BPR_EPOCHS=3, BPR_DIM=4, LIGHTGCN_LAYERS=2):
            model = LightGCNModel()
            model.fit(data)
            first = model.score(data.examples[0].inputs)
            model.fit(data)
            second = model.score(data.examples[0].inputs)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
        self.assertTrue(np.isfinite(first).all())

    def test_graph_uses_histories_and_positive_candidates_but_not_negatives(self) -> None:
        graph = build_graph(graph_fixture())
        dense = graph.adjacency.to_dense().numpy()
        self.assertGreater(dense[0, 2], 0.0)  # U1 history A
        self.assertGreater(dense[0, 3], 0.0)  # U1 clicked B
        self.assertEqual(dense[0, 4], 0.0)  # U1 displayed-negative C

    def test_score_before_fit_fails(self) -> None:
        with self.assertRaisesRegex(DataError, "before fit"):
            LightGCNModel().score(graph_fixture().examples[0].inputs)

    def test_disconnected_untrained_article_scores_zero(self) -> None:
        data = graph_fixture()
        with patch.multiple(config, BPR_EPOCHS=2, BPR_DIM=4):
            model = LightGCNModel()
            model.fit(data)
        inputs = RankingInput("E", "U1", datetime(2019, 1, 3), ("A",), 1, ("D",))
        np.testing.assert_array_equal(model.score(inputs), np.zeros(1))


if __name__ == "__main__":
    unittest.main()
