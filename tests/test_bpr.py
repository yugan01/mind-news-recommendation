import unittest
from dataclasses import replace
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from src import config
from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError
from src.models.bpr import (
    EmbeddingProvider,
    bpr_objective,
    build_pair_pool,
    sample_pairs,
    train_bpr,
)
from src.models.bpr_mf import BPRMFModel


def make_training_data() -> TrainingData:
    articles = {
        article: Article(article, "topic", f"Title {article}", "")
        for article in ("A", "B", "C", "D")
    }
    rows = (
        Example(
            RankingInput("I1", "U1", datetime(2019, 11, 1), ("A",), 1, ("B", "C", "D")),
            Targets((1, 0, 0)),
        ),
        Example(
            RankingInput("I2", "U2", datetime(2019, 11, 2), ("B",), 1, ("A", "D")),
            Targets((1, 1)),
        ),
    )
    return TrainingData(
        articles,
        rows,
        {"U1": 0, "U2": 1},
        {article: index for index, article in enumerate(articles)},
    )


class BPRTests(unittest.TestCase):
    def test_collaborative_models_are_registered_after_existing_models(self) -> None:
        self.assertEqual(
            tuple(model_type().name for model_type in config.MODELS),
            ("popularity", "semantic", "bpr_mf", "lightgcn", "two_tower"),
        )

    def test_fixed_training_settings(self) -> None:
        self.assertEqual(config.BPR_DIM, 64)
        self.assertEqual(config.BPR_EPOCHS, 20)
        self.assertEqual(config.BPR_NEGATIVES, 1)
        self.assertEqual(config.BPR_LEARNING_RATE, 0.001)
        self.assertEqual(config.BPR_L2, 0.0001)
        self.assertEqual(config.BPR_INIT_STD, 0.1)
        self.assertEqual(config.LIGHTGCN_LAYERS, 2)

    def test_pair_pool_skips_one_class_impressions(self) -> None:
        pool = build_pair_pool(make_training_data())
        np.testing.assert_array_equal(pool.user_indices, [0])
        np.testing.assert_array_equal(pool.positive_indices, [1])
        np.testing.assert_array_equal(pool.example_indices, [0])
        np.testing.assert_array_equal(pool.negative_options[0], [2, 3])
        self.assertEqual((pool.included_impressions, pool.skipped_impressions), (1, 1))

    def test_sampling_is_epoch_deterministic_and_stays_in_impression(self) -> None:
        pool = build_pair_pool(make_training_data())
        first = sample_pairs(pool, epoch=0, seed=4285, device=torch.device("cpu"))
        repeat = sample_pairs(pool, epoch=0, seed=4285, device=torch.device("cpu"))
        next_epoch = sample_pairs(pool, epoch=1, seed=4285, device=torch.device("cpu"))
        self.assertTrue(torch.equal(first.negative_indices, repeat.negative_indices))
        self.assertFalse(torch.equal(first.negative_indices, next_epoch.negative_indices))
        self.assertIn(int(first.negative_indices[0]), (2, 3))
        self.assertEqual(int(first.example_indices[0]), 0)

    def test_pair_pool_rejects_no_usable_pairs(self) -> None:
        data = make_training_data()
        with self.assertRaisesRegex(DataError, "no usable BPR pairs"):
            build_pair_pool(
                TrainingData(
                    data.articles,
                    data.examples[1:],
                    data.user_indices,
                    data.article_indices,
                )
            )

    def test_pair_pool_rejects_noncontiguous_indices(self) -> None:
        data = make_training_data()
        broken = TrainingData(
            data.articles,
            data.examples,
            data.user_indices,
            {"A": 0, "B": 2, "C": 3, "D": 4},
        )
        with self.assertRaisesRegex(DataError, "contiguous article indices"):
            build_pair_pool(broken)

    def test_exact_bpr_objective(self) -> None:
        provider = EmbeddingProvider(1, 2, 1, seed=4285)
        with torch.no_grad():
            provider.user_embeddings.weight[:] = torch.tensor([[1.0]])
            provider.article_embeddings.weight[:] = torch.tensor([[2.0], [0.0]])
        pairs = sample_pairs(
            build_pair_pool(
                TrainingData(
                    {"P": Article("P", "x", "P", ""), "N": Article("N", "x", "N", "")},
                    (
                        Example(
                            RankingInput(
                                "I", "U", datetime(2019, 1, 1), (), 0, ("P", "N")
                            ),
                            Targets((1, 0)),
                        ),
                    ),
                    {"U": 0},
                    {"P": 0, "N": 1},
                )
            ),
            epoch=0,
            seed=4285,
            device=torch.device("cpu"),
        )
        loss = bpr_objective(provider, pairs, l2=0.0)
        self.assertAlmostEqual(float(loss.detach()), float(np.log1p(np.exp(-2.0))), places=6)

    def test_shared_training_increases_positive_margin(self) -> None:
        data = make_training_data()
        provider = EmbeddingProvider(2, 4, 4, seed=4285)
        pool = build_pair_pool(data)
        with torch.no_grad():
            before_users, before_articles = provider()
            before = float(
                np.mean(
                    [
                        float((before_users[0] * before_articles[1]).sum())
                        - float((before_users[0] * before_articles[index]).sum())
                        for index in (2, 3)
                    ]
                )
            )
        result = train_bpr(
            provider, pool, epochs=1, learning_rate=0.01, l2=0.0, seed=4285
        )
        after = float(
            np.mean(
                [
                    result.user_vectors[0] @ result.article_vectors[1]
                    - result.user_vectors[0] @ result.article_vectors[index]
                    for index in (2, 3)
                ]
            )
        )
        self.assertGreater(after, before)
        self.assertEqual(len(result.losses), 1)
        self.assertTrue(np.isfinite(result.losses).all())

    def test_exact_sampled_training_masks(self) -> None:
        data = make_training_data()
        result = train_bpr(
            EmbeddingProvider(2, 4, 4, seed=4285),
            build_pair_pool(data),
            epochs=1,
            learning_rate=0.01,
            l2=0.0,
            seed=4285,
        )
        np.testing.assert_array_equal(result.trained_users, [True, False])
        np.testing.assert_array_equal(result.trained_articles, [False, True, False, True])

    def test_shared_training_rejects_nonfinite_loss(self) -> None:
        data = make_training_data()
        provider = EmbeddingProvider(2, 4, 4, seed=4285)
        previous_determinism = torch.are_deterministic_algorithms_enabled()
        with (
            patch(
                "src.models.bpr.bpr_objective",
                return_value=torch.tensor(float("nan"), requires_grad=True),
            ),
            self.assertRaisesRegex(DataError, "non-finite loss at epoch 0"),
        ):
            train_bpr(
                provider,
                build_pair_pool(data),
                epochs=1,
                learning_rate=0.01,
                l2=0.0,
                seed=4285,
            )
        self.assertEqual(
            torch.are_deterministic_algorithms_enabled(), previous_determinism
        )

    def test_bpr_mf_fit_score_cold_start_and_refit(self) -> None:
        data = make_training_data()
        with patch.multiple(config, BPR_EPOCHS=3, BPR_DIM=4):
            model = BPRMFModel()
            model.fit(data)
            first = model.score(data.examples[0].inputs)
            model.fit(data)
            second = model.score(data.examples[0].inputs)
        np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
        self.assertEqual(first.shape, (3,))
        self.assertTrue(np.isfinite(first).all())
        unseen = replace(data.examples[0].inputs, user_id="NEW", history=("B",))
        self.assertTrue(np.isfinite(model.score(unseen)).all())
        empty = replace(unseen, history=())
        np.testing.assert_array_equal(model.score(empty), np.zeros(3))

    def test_bpr_mf_masks_untrained_candidates(self) -> None:
        data = make_training_data()
        with patch.multiple(config, BPR_EPOCHS=1, BPR_DIM=4):
            model = BPRMFModel()
            model.fit(data)
        inputs = replace(data.examples[0].inputs, candidate_ids=("A",))
        np.testing.assert_array_equal(model.score(inputs), np.zeros(1))
        self.assertIsNotNone(model.state)
        assert model.state is not None
        cold = replace(
            data.examples[0].inputs,
            user_id="NEW",
            history=("A", "B"),
            candidate_ids=("B",),
        )
        expected = float(model.state.article_vectors[1] @ model.state.article_vectors[1])
        np.testing.assert_allclose(model.score(cold), [expected], rtol=0, atol=1e-7)

    def test_bpr_mf_rejects_score_before_fit(self) -> None:
        with self.assertRaisesRegex(DataError, "before fit"):
            BPRMFModel().score(make_training_data().examples[0].inputs)


if __name__ == "__main__":
    unittest.main()
