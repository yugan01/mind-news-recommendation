"""Popularity and TF-IDF recommendation baselines."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from .data import Article, Impression, iter_impressions


class PopularityModel:
    def __init__(self) -> None:
        self.clicks: Counter[str] = Counter()

    def fit(self, behaviors_path: Path) -> "PopularityModel":
        if behaviors_path.suffix.lower() == ".csv":
            with behaviors_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row["click"] == "1":
                        self.clicks[row["news_id"]] += 1
            return self
        for impression in iter_impressions(behaviors_path):
            self.clicks.update(
                candidate.article_id
                for candidate in impression.candidates
                if candidate.label == 1
            )
        return self

    def score(self, impression: Impression) -> np.ndarray:
        return np.asarray(
            [self.clicks[candidate.article_id] for candidate in impression.candidates],
            dtype=float,
        )


class TfidfModel:
    def __init__(self, max_history: int = 20, max_features: int = 100_000) -> None:
        self.max_history = max_history
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self.article_ids: list[str] = []
        self.index: dict[str, int] = {}
        self.matrix: csr_matrix | None = None

    def fit(
        self,
        training_articles: dict[str, Article],
        all_articles: dict[str, Article],
    ) -> "TfidfModel":
        training_texts = [article.text for article in training_articles.values()]
        self.vectorizer.fit(training_texts)
        self.article_ids = list(all_articles)
        self.index = {article_id: index for index, article_id in enumerate(self.article_ids)}
        self.matrix = self.vectorizer.transform(
            [all_articles[article_id].text for article_id in self.article_ids]
        ).tocsr()
        return self

    def score(self, impression: Impression) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("Model must be fitted before scoring")
        history_indices = [
            self.index[article_id]
            for article_id in impression.history[-self.max_history :]
            if article_id in self.index
        ]
        candidate_indices = [
            self.index.get(candidate.article_id) for candidate in impression.candidates
        ]
        if not history_indices:
            return np.zeros(len(candidate_indices), dtype=float)

        user_vector = self.matrix[history_indices].mean(axis=0)
        user_norm = float(np.linalg.norm(user_vector))
        scores = np.zeros(len(candidate_indices), dtype=float)
        known_positions = [i for i, index in enumerate(candidate_indices) if index is not None]
        if not known_positions or user_norm == 0.0:
            return scores

        known_indices = [candidate_indices[i] for i in known_positions]
        candidate_matrix = self.matrix[known_indices]
        similarities = np.asarray(candidate_matrix @ user_vector.T).ravel()
        candidate_norms = np.sqrt(candidate_matrix.multiply(candidate_matrix).sum(axis=1)).A1
        denominator = candidate_norms * user_norm
        similarities = np.divide(
            similarities,
            denominator,
            out=np.zeros_like(similarities),
            where=denominator != 0,
        )
        scores[known_positions] = similarities
        return scores

