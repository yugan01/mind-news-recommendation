import csv
import json
import math
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.typing import NDArray

from scripts.smoke_mind import load_smoke
from src import config
from src.analyze import Prediction, bootstrap_interval, compare, read_predictions
from src.contracts import Model, RankingInput, Targets
from src.data import (
    DataError,
    Impression,
    chronological_split,
    is_eligible,
    load_articles,
    load_development,
    make_example,
    read_impressions,
    select_dev_cohort,
)
from src.evaluate import evaluate_model
from src.final import load_final
from src.metrics import ndcg_at_k, reciprocal_rank, summarize_users
from src.models.bpr_mf import BPRMFModel
from src.models.lightgcn import LightGCNModel
from src.models.popularity import PopularityModel
from src.run import run_experiment


class InspectingModel(PopularityModel):
    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if hasattr(inputs, "labels") or hasattr(inputs, "targets"):
            raise AssertionError("Evaluation labels reached the model")
        if len(inputs.history) > config.MAX_HISTORY:
            raise AssertionError("History was not capped")
        return super().score(inputs)


class InfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.train = self.root / "train"
        self.train.mkdir()
        self.news = (
            'A\ttech\tx\tA title\t"Unmatched quote\turl\t[]\t[]\n'
            "B\tsport\tx\tB title\t\turl\t[]\t[]\n"
            "C\ttech\tx\tC title\tC abstract\turl\t[]\t[]\n"
        )
        (self.train / "news.tsv").write_text(self.news, encoding="utf-8")
        self.rows = [
            Impression(
                str(index),
                f"U{index % 3}",
                datetime(2019, 11, 11) + timedelta(hours=index),
                ("A", "B") * 12 if index else (),
                ("B", "C"),
                (0, 1),
            )
            for index in range(10)
        ]
        self.write_behaviors(self.train, self.rows)
        self.configuration = patch.multiple(
            config,
            TRAIN_DIR=self.train,
            VALIDATION_DIR=self.root / "validation",
            BOOTSTRAP_ITERATIONS=100,
        )
        self.configuration.start()

    def tearDown(self) -> None:
        self.configuration.stop()
        self.temp.cleanup()

    def write_behaviors(self, directory: Path, rows: list[Impression]) -> None:
        with (directory / "behaviors.tsv").open("w", encoding="utf-8") as handle:
            for row in rows:
                candidates = " ".join(
                    f"{article}-{label}"
                    for article, label in zip(row.candidate_ids, row.labels, strict=True)
                )
                timestamp = row.timestamp.strftime("%m/%d/%Y %I:%M:%S %p")
                handle.write(
                    f"{row.impression_id}\t{row.user_id}\t{timestamp}"
                    f"\t{' '.join(row.history)}\t{candidates}\n"
                )

    def test_original_tsv_metadata_and_quotes(self) -> None:
        articles = load_articles(self.train / "news.tsv")
        self.assertEqual(articles["A"].abstract, '"Unmatched quote')
        self.assertEqual(articles["A"].category, "tech")
        self.assertEqual(articles["B"].text, "B title")
        self.assertEqual(list(read_impressions(self.train / "behaviors.tsv")), self.rows)

    def test_invalid_data_is_not_silently_skipped(self) -> None:
        with (self.train / "news.tsv").open("a", encoding="utf-8") as handle:
            handle.write(self.news.splitlines()[0] + "\n")
        with self.assertRaisesRegex(DataError, "duplicate article"):
            load_articles(self.train / "news.tsv")
        (self.train / "news.tsv").write_text(self.news, encoding="utf-8")
        self.write_behaviors(
            self.train, [replace(self.rows[0], history=("missing",)), *self.rows[1:]]
        )
        with self.assertRaisesRegex(DataError, "missing articles"):
            load_development()
        self.write_behaviors(self.train, [replace(self.rows[0], labels=(2, 0))])
        with self.assertRaises(DataError):
            list(read_impressions(self.train / "behaviors.tsv"))

    def test_chronological_split_keeps_ties_together(self) -> None:
        rows = self.rows.copy()
        rows[7] = replace(rows[7], timestamp=rows[8].timestamp)
        training, evaluation = chronological_split(rows)
        self.assertEqual(len(training), 7)
        self.assertLess(
            max(row.timestamp for row in training), min(row.timestamp for row in evaluation)
        )
        with self.assertRaises(DataError):
            chronological_split([rows[8], replace(rows[9], timestamp=rows[8].timestamp)])

    def test_eligibility_uses_full_history_before_truncation(self) -> None:
        self.assertTrue(is_eligible(self.rows[1]))
        example = make_example(self.rows[1])
        self.assertEqual(example.inputs.available_history_length, 24)
        self.assertEqual(example.inputs.history, self.rows[1].history[-20:])
        self.assertFalse(is_eligible(replace(self.rows[1], labels=(1, 1))))
        self.assertFalse(is_eligible(replace(self.rows[1], history=("A",))))

    def test_sampling_is_by_user_and_reproducible(self) -> None:
        with patch.object(config, "DEV_USERS", 1):
            left = select_dev_cohort(self.rows)
            right = select_dev_cohort(list(reversed(self.rows)))
        self.assertEqual({row.impression_id for row in left}, {row.impression_id for row in right})
        self.assertEqual(len({row.user_id for row in left}), 1)
        self.assertEqual(
            left,
            [row for row in self.rows if row.user_id == left[0].user_id and is_eligible(row)],
        )

    def test_development_retains_training_rows_and_never_opens_validation(self) -> None:
        self.assertFalse(config.VALIDATION_DIR.exists())
        data = load_development()
        self.assertEqual(len(data.training.examples), 8)
        self.assertEqual(data.training.examples[0].inputs.history, ())
        self.assertEqual(len(data.evaluation), 2)
        self.assertEqual(list(data.training.article_indices), ["A", "B", "C"])
        self.assertEqual(data.evaluation, load_development().evaluation)

    def test_popularity_follows_contract_and_refit_resets_counts(self) -> None:
        data = load_development()
        model: Model = InspectingModel()
        for _ in range(2):
            model.fit(data.training)
            np.testing.assert_array_equal(model.score(data.evaluation[0].inputs), [0.0, 8.0])
        unseen = replace(data.evaluation[0].inputs, candidate_ids=("unseen",))
        np.testing.assert_array_equal(model.score(unseen), [0.0])

    def test_mind_smoke_limits_examples_and_article_corpus(self) -> None:
        development = load_development()
        development = replace(
            development,
            training=replace(
                development.training,
                articles={
                    **development.training.articles,
                    "unused": replace(development.training.articles["A"], article_id="unused"),
                },
            ),
        )
        with patch("scripts.smoke_mind.load_development", return_value=development):
            data = load_smoke()
        self.assertEqual(len(data.training.examples), 2)
        self.assertTrue(
            all(set(example.targets.labels) == {0, 1} for example in data.training.examples)
        )
        self.assertEqual(data.evaluation, development.evaluation[:2])
        expected = {
            article
            for example in (*data.training.examples, *data.evaluation)
            for article in (*example.inputs.history, *example.inputs.candidate_ids)
        }
        self.assertEqual(set(data.training.articles), expected)
        self.assertEqual(set(data.training.article_indices), expected)
        self.assertEqual(
            set(data.training.user_indices),
            {example.inputs.user_id for example in data.training.examples},
        )

    def test_evaluation_and_standard_run_outputs(self) -> None:
        data = load_development()
        model = InspectingModel()
        model.fit(data.training)
        output = self.root / "inspect"
        metrics = evaluate_model(model, data.evaluation, output)
        self.assertEqual(metrics["ndcg_at_5"], 1.0)
        self.assertEqual(set(read_predictions(output / "predictions.csv")), {"8", "9"})
        with patch.object(config, "MODELS", (InspectingModel,)):
            run_experiment(data, self.root / "run")
        self.assertTrue((self.root / "run/metrics.json").exists())
        with (self.root / "run/metrics.json").open(encoding="utf-8") as handle:
            summary = json.load(handle)
        self.assertEqual(summary["settings"]["bpr_dimension"], config.BPR_DIM)
        self.assertEqual(summary["settings"]["bpr_optimizer"], "Adam")
        self.assertEqual(summary["settings"]["lightgcn_layers"], config.LIGHTGCN_LAYERS)
        self.assertEqual(
            summary["settings"]["lightgcn_layer_combination"], "uniform_mean"
        )
        self.assertEqual(summary["models"]["popularity"]["ndcg_at_5"], 1.0)
        self.assertEqual(summary["training"]["popularity"], {})
        with self.assertRaisesRegex(DataError, "existing output"):
            run_experiment(data, self.root / "run")
        with patch.object(model, "score", return_value=np.array([np.nan, 0.0])):
            with self.assertRaisesRegex(DataError, "invalid scores"):
                evaluate_model(model, data.evaluation, self.root / "nan")
        with patch.object(model, "score", return_value=np.array([1.0])):
            with self.assertRaisesRegex(DataError, "invalid scores"):
                evaluate_model(model, data.evaluation, self.root / "wrong-length")

    def test_collaborative_training_metadata_is_exported(self) -> None:
        data = load_development()
        output = self.root / "collaborative"
        with (
            patch.object(config, "MODELS", (BPRMFModel, LightGCNModel)),
            patch.multiple(config, BPR_EPOCHS=1, BPR_DIM=4, LIGHTGCN_LAYERS=1),
        ):
            run_experiment(data, output)
        with (output / "metrics.json").open(encoding="utf-8") as handle:
            training = json.load(handle)["training"]
        for name in ("bpr_mf", "lightgcn"):
            metadata = training[name]
            self.assertEqual(metadata["epochs"], 1)
            self.assertTrue(math.isfinite(metadata["final_loss"]))
            self.assertGreater(metadata["trained_users"], 0)
            self.assertGreater(metadata["trained_articles"], 0)
            self.assertIn(metadata["device"], ("cpu", "mps", "cuda"))
        self.assertGreater(training["lightgcn"]["unique_graph_edges"], 0)

    def test_held_out_label_mutation_cannot_change_collaborative_fit(self) -> None:
        data = load_development()
        mutated = replace(
            data,
            evaluation=tuple(
                replace(
                    example,
                    targets=Targets(tuple(1 - label for label in example.targets.labels)),
                )
                for example in data.evaluation
            ),
        )
        left = self.root / "held-out-left"
        right = self.root / "held-out-right"
        with (
            patch.object(config, "MODELS", (BPRMFModel, LightGCNModel)),
            patch.multiple(config, BPR_EPOCHS=1, BPR_DIM=4, LIGHTGCN_LAYERS=1),
        ):
            run_experiment(data, left)
            run_experiment(mutated, right)
        with (left / "metrics.json").open(encoding="utf-8") as handle:
            left_training = json.load(handle)["training"]
        with (right / "metrics.json").open(encoding="utf-8") as handle:
            right_training = json.load(handle)["training"]
        self.assertEqual(left_training, right_training)
        for name in ("bpr_mf", "lightgcn"):
            with (left / name / "predictions.csv").open(encoding="utf-8") as handle:
                left_scores = [row["score"] for row in csv.DictReader(handle)]
            with (right / name / "predictions.csv").open(encoding="utf-8") as handle:
                right_scores = [row["score"] for row in csv.DictReader(handle)]
            self.assertEqual(left_scores, right_scores)

    def test_metrics_and_equal_user_weight(self) -> None:
        self.assertEqual(ndcg_at_k((1, 0), np.array([1.0, 0.0]), 5), 1.0)
        self.assertEqual(reciprocal_rank((0, 1), np.array([1.0, 1.0])), 0.5)
        metrics = summarize_users({"U1": [(1.0, 1.0)] * 10, "U2": [(0.0, 0.0)]})
        self.assertEqual(metrics["ndcg_at_5"], 0.5)
        self.assertEqual(bootstrap_interval(np.array([0.25, 0.25])), (0.25, 0.25))

    def test_comparisons_reject_different_examples(self) -> None:
        base = Prediction("U1", ("B", "C"), (0, 1), np.array([0.0, 1.0]))
        alternatives = [
            {},
            {"other": base},
            {"1": replace(base, user_id="U2")},
            {"1": replace(base, article_ids=("C", "B"))},
            {"1": replace(base, labels=(1, 0))},
        ]
        for right in alternatives:
            with self.subTest(right=right), self.assertRaises(DataError):
                compare({"1": base}, right)
        result = compare({"1": base}, {"1": replace(base, scores=-base.scores)})
        self.assertEqual(result["paired_users"], 1)

    def test_final_uses_full_training_and_all_eligible_validation_users(self) -> None:
        config.VALIDATION_DIR.mkdir()
        (config.VALIDATION_DIR / "news.tsv").write_text(
            self.news + "D\ttech\tx\tD title\t\turl\t[]\t[]\n",
            encoding="utf-8",
        )
        rows = [
            replace(
                self.rows[8],
                impression_id="V1",
                timestamp=datetime(2019, 11, 12),
                candidate_ids=("C", "D"),
            ),
            replace(
                self.rows[9],
                impression_id="V2",
                user_id="unseen",
                timestamp=datetime(2019, 11, 12, 1),
            ),
        ]
        self.write_behaviors(config.VALIDATION_DIR, rows)
        with patch.object(config, "DEV_USERS", 1):
            data = load_final()
        self.assertEqual(len(data.training.examples), 10)
        self.assertEqual(len(data.evaluation), 2)
        self.assertNotIn("unseen", data.training.user_indices)
        self.assertIn("D", data.training.articles)


if __name__ == "__main__":
    unittest.main()
