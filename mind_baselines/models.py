"""Popularity recommendation baseline."""

from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path

import numpy as np

from .data import Impression, iter_impressions


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
