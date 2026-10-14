from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mind_baselines.data import eligible_impressions, load_articles
from mind_baselines.metrics import aggregate_by_user, ndcg_at_k, reciprocal_rank
from mind_baselines.models import PopularityModel, TfidfModel
from mind_baselines.analyze import bootstrap_interval


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.news = self.root / "news.tsv"
        self.behaviors = self.root / "behaviors.tsv"
        self.news.write_text(
            "N1\ttech\telectric\tElectric cars improve\tBattery range increases\turl\t[]\t[]\n"
            "N2\tsport\tfootball\tFootball final\tTeam wins the match\turl\t[]\t[]\n"
            "N3\ttech\tbattery\tNew battery\tElectric vehicle technology\turl\t[]\t[]\n",
            encoding="utf-8",
        )
        history = " ".join(["N1"] * 10)
        self.behaviors.write_text(
            f"1\tU1\t11/15/2019 9:00:00 AM\t{history}\tN3-1 N2-0\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_parser_and_filter(self) -> None:
        articles = load_articles(self.news)
        impressions = eligible_impressions(self.behaviors)
        self.assertEqual(set(articles), {"N1", "N2", "N3"})
        self.assertEqual(len(impressions), 1)
        self.assertEqual(impressions[0].candidates[0].label, 1)

    def test_metrics(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([1, 0], [0.9, 0.1]), 1.0)
        self.assertAlmostEqual(reciprocal_rank([0, 1], [0.9, 0.1]), 0.5)
        summary = aggregate_by_user(
            [{"user_id": "U1", "labels": [1, 0], "scores": [1.0, 0.0]}]
        )
        self.assertEqual(summary["users"], 1)
        self.assertAlmostEqual(summary["ndcg_at_5"], 1.0)
        low, high = bootstrap_interval(
            __import__("numpy").asarray([0.1, 0.2, 0.3]), iterations=1000, seed=1
        )
        self.assertGreater(low, 0.0)
        self.assertGreaterEqual(high, low)

    def test_models_prefer_clicked_article(self) -> None:
        articles = load_articles(self.news)
        impression = eligible_impressions(self.behaviors)[0]
        popularity = PopularityModel().fit(self.behaviors)
        self.assertGreater(popularity.score(impression)[0], popularity.score(impression)[1])
        tfidf = TfidfModel(max_features=100).fit(articles, articles)
        scores = tfidf.score(impression)
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()

