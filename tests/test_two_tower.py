import tempfile
import unittest
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from src import config
from src.contracts import Article, Example, RankingInput, Targets, TrainingData
from src.data import DataError
from src.models.bpr import build_pair_pool, sample_pairs
from src.models.two_tower import (
    TwoTowerModel,
    TwoTowerNetwork,
    history_features,
    train_two_tower,
    two_tower_objective,
)


def tower_fixture(*, include_empty: bool = False) -> tuple[TrainingData, NDArray[np.float32]]:
    articles = {
        article: Article(article, "topic", article, "")
        for article in ("A", "B", "C", "D")
    }
    examples = [
        Example(
            RankingInput("I1", "U1", datetime(2019, 1, 1), ("A", "B"), 2, ("B", "C")),
            Targets((1, 0)),
        )
    ]
    if include_empty:
        examples.append(
            Example(
                RankingInput("I2", "U2", datetime(2019, 1, 2), (), 0, ("A", "D")),
                Targets((1, 0)),
            )
        )
    data = TrainingData(
        articles,
        tuple(examples),
        {"U1": 0, "U2": 1},
        {"A": 0, "B": 1, "C": 2, "D": 3},
    )
    vectors = np.zeros((4, 768), dtype=np.float32)
    for index in range(4):
        vectors[index, index] = 1.0
    return data, vectors


class TwoTowerTests(unittest.TestCase):
    def test_two_tower_is_registered_last(self) -> None:
        self.assertEqual(tuple(model().name for model in config.MODELS)[-1], "two_tower")

    def test_fixed_settings_and_exact_separate_architecture(self) -> None:
        self.assertEqual(config.TWO_TOWER_HIDDEN, 64)
        self.assertEqual(config.TWO_TOWER_OUTPUT, 64)
        self.assertEqual(config.TWO_TOWER_EPOCHS, 20)
        self.assertEqual(config.TWO_TOWER_LEARNING_RATE, 0.001)
        self.assertEqual(config.TWO_TOWER_L2, 0.0001)
        network = TwoTowerNetwork(768, 64, 64, seed=4285)
        self.assertIsNot(network.user_tower, network.article_tower)
        for tower in (network.user_tower, network.article_tower):
            self.assertEqual([type(layer) for layer in tower], [nn.Linear, nn.ReLU, nn.Linear])
            first = cast(nn.Linear, tower[0])
            second = cast(nn.Linear, tower[2])
            self.assertEqual((first.in_features, first.out_features), (768, 64))
            self.assertEqual((second.in_features, second.out_features), (64, 64))

    def test_history_features_follow_each_example_and_empty_is_zero(self) -> None:
        data, vectors = tower_fixture()
        changed = replace(data.examples[0], inputs=replace(data.examples[0].inputs, history=("C",)))
        empty = replace(data.examples[0], inputs=replace(data.examples[0].inputs, history=()))
        data = replace(data, examples=(data.examples[0], changed, empty))
        features = history_features(data, vectors, {"A": 0, "B": 1, "C": 2, "D": 3})
        np.testing.assert_array_equal(features[0, :4], [0.5, 0.5, 0.0, 0.0])
        np.testing.assert_array_equal(features[1, :4], [0.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(features[2], np.zeros(768))

    def test_objective_projects_once_and_duplicate_gradients_accumulate(self) -> None:
        data, vectors = tower_fixture()
        histories = torch.as_tensor(history_features(data, vectors, data.article_indices))
        articles = torch.as_tensor(vectors)
        network = TwoTowerNetwork(768, 4, 4, seed=4285)
        with torch.no_grad():
            for tower in (network.user_tower, network.article_tower):
                first = cast(nn.Linear, tower[0])
                second = cast(nn.Linear, tower[2])
                first.weight.zero_()
                first.weight[:, :4] = torch.eye(4)
                first.bias.zero_()
                second.weight.copy_(torch.eye(4))
                second.bias.zero_()
        pairs = sample_pairs(build_pair_pool(data), 0, 4285, torch.device("cpu"))
        duplicated = replace(
            pairs,
            positive_indices=torch.cat((pairs.positive_indices, pairs.positive_indices)),
            negative_indices=torch.cat((pairs.negative_indices, pairs.negative_indices)),
            example_indices=torch.cat((pairs.example_indices, pairs.example_indices)),
            user_indices=torch.cat((pairs.user_indices, pairs.user_indices)),
        )
        calls = {"user": 0, "article": 0}
        user_hook = network.user_tower.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("user", calls["user"] + 1)
        )
        article_hook = network.article_tower.register_forward_hook(
            lambda _module, _inputs, _output: calls.__setitem__("article", calls["article"] + 1)
        )
        loss = two_tower_objective(network, histories, articles, duplicated, l2=0.0)
        loss.backward()  # type: ignore[no-untyped-call]
        user_hook.remove()
        article_hook.remove()
        self.assertEqual(calls, {"user": 1, "article": 1})
        duplicated_gradients = [
            parameter.grad.detach().clone()
            for parameter in network.parameters()
            if parameter.grad is not None
        ]
        user_gradient = cast(nn.Linear, network.user_tower[0]).weight.grad
        article_gradient = cast(nn.Linear, network.article_tower[0]).weight.grad
        self.assertIsNotNone(user_gradient)
        self.assertIsNotNone(article_gradient)
        assert user_gradient is not None
        assert article_gradient is not None
        self.assertGreater(float(user_gradient.abs().sum()), 0.0)
        self.assertGreater(float(article_gradient.abs().sum()), 0.0)
        network.zero_grad(set_to_none=True)
        two_tower_objective(
            network, histories, articles, pairs, l2=0.0
        ).backward()  # type: ignore[no-untyped-call]
        single_gradients = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.grad is not None
        ]
        self.assertEqual(len(duplicated_gradients), len(single_gradients))
        for duplicated_gradient, single_gradient in zip(
            duplicated_gradients, single_gradients, strict=True
        ):
            torch.testing.assert_close(duplicated_gradient, single_gradient)

    def test_single_update_improves_sampled_positive_negative_margin(self) -> None:
        data, vectors = tower_fixture()
        histories = torch.as_tensor(history_features(data, vectors, data.article_indices))
        articles = torch.as_tensor(vectors)
        network = TwoTowerNetwork(768, 4, 4, seed=4285)
        pairs = sample_pairs(build_pair_pool(data), 0, 4285, torch.device("cpu"))

        def margin() -> float:
            with torch.no_grad():
                users = network.user_vectors(histories)
                items = network.article_vectors(articles)
                return float(
                    (users[pairs.example_indices] * items[pairs.positive_indices]).sum()
                    - (users[pairs.example_indices] * items[pairs.negative_indices]).sum()
                )

        before = margin()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
        optimizer.zero_grad(set_to_none=True)
        two_tower_objective(
            network, histories, articles, pairs, l2=0.0
        ).backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        self.assertGreater(margin(), before)

    def test_fit_is_deterministic_skips_empty_histories_and_records_metadata(self) -> None:
        data, vectors = tower_fixture(include_empty=True)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        with (
            patch.object(config, "SEMANTIC_CACHE", Path(temp.name) / "vectors.npz"),
            patch.object(config, "TWO_TOWER_EPOCHS", 3),
            patch("src.models.two_tower.encode_articles", return_value=vectors),
        ):
            left = TwoTowerModel()
            right = TwoTowerModel()
            left.fit(data)
            left_scores = left.score(data.examples[0].inputs)
            right.fit(data)
            right_scores = right.score(data.examples[0].inputs)
        np.testing.assert_allclose(left_scores, right_scores, rtol=0, atol=1e-7)
        self.assertEqual(left.run_metadata["skipped_empty_histories"], 1)
        self.assertEqual(left.run_metadata["included_impressions"], 1)
        self.assertEqual(left.run_metadata["input_dimension"], 768)
        self.assertEqual(left.run_metadata["architecture"], "separate 768->64->64 towers")
        self.assertEqual(left.run_metadata["optimizer"], "Adam")
        self.assertEqual(left.run_metadata["frozen_encoder"], config.SEMANTIC_ENCODER)
        self.assertEqual(left.run_metadata["seed"], config.SEED)
        self.assertEqual(left.run_metadata["learning_rate"], config.TWO_TOWER_LEARNING_RATE)
        self.assertEqual(left.run_metadata["l2"], config.TWO_TOWER_L2)
        self.assertEqual(left.run_metadata["epochs"], 3)
        self.assertEqual(left.run_metadata["pair_count"], 1)
        self.assertTrue(np.isfinite(cast(float, left.run_metadata["final_loss"])))

    def test_true_interaction_cold_start_and_empty_history_scores(self) -> None:
        data, vectors = tower_fixture()
        model = TwoTowerModel()
        model.vector_indices = dict(data.article_indices)
        model.vectors = vectors

        class IdentityTower(nn.Module):
            def user_vectors(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, :4]

            def article_vectors(self, features: torch.Tensor) -> torch.Tensor:
                return features[:, :4]

        model.network = cast(TwoTowerNetwork, IdentityTower())
        unseen = replace(
            data.examples[0].inputs,
            user_id="NEW",
            history=("D",),
            candidate_ids=("D", "B"),
        )
        np.testing.assert_array_equal(model.score(unseen), [1.0, 0.0])
        np.testing.assert_array_equal(model.score(replace(unseen, history=())), np.zeros(2))

    def test_fit_rejects_only_empty_history_pairs(self) -> None:
        data, vectors = tower_fixture(include_empty=True)
        data = replace(data, examples=(data.examples[1],))
        with (
            patch.object(config, "SEMANTIC_CACHE", Path("unused-cache.npz")),
            patch(
                "src.models.two_tower.load_article_vectors",
                return_value=(dict(data.article_indices), vectors),
            ),
            self.assertRaisesRegex(DataError, "no usable nonempty-history pairs"),
        ):
            TwoTowerModel().fit(data)

    def test_nonfinite_failure_restores_determinism_state(self) -> None:
        data, vectors = tower_fixture()
        histories = history_features(data, vectors, data.article_indices)
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(not previous)
        expected = not previous
        try:
            with (
                patch(
                    "src.models.two_tower.two_tower_objective",
                    return_value=torch.tensor(float("nan"), requires_grad=True),
                ),
                self.assertRaisesRegex(DataError, "non-finite loss at epoch 0"),
            ):
                train_two_tower(
                    TwoTowerNetwork(768, 4, 4, seed=4285),
                    histories,
                    vectors,
                    build_pair_pool(data),
                    epochs=1,
                    learning_rate=0.01,
                    l2=0.0,
                    seed=4285,
                )
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), expected)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_success_restores_determinism_state_and_score_before_fit_fails(self) -> None:
        data, vectors = tower_fixture()
        histories = history_features(data, vectors, data.article_indices)
        previous = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(not previous)
        expected = not previous
        try:
            train_two_tower(
                TwoTowerNetwork(768, 4, 4, seed=4285),
                histories,
                vectors,
                build_pair_pool(data),
                epochs=1,
                learning_rate=0.01,
                l2=0.0,
                seed=4285,
            )
            self.assertEqual(torch.are_deterministic_algorithms_enabled(), expected)
        finally:
            torch.use_deterministic_algorithms(previous)
        with self.assertRaisesRegex(DataError, "before fit"):
            TwoTowerModel().score(data.examples[0].inputs)


if __name__ == "__main__":
    unittest.main()
