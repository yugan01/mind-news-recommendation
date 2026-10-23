"""Strict original-MIND readers and shared impression preprocessing."""

import csv
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Mapping

from src import config
from src.contracts import Article, Example, RankingInput, Targets, TrainingData


class DataError(ValueError):
    """Input data violates the experiment's schema or integrity requirements."""


@dataclass(frozen=True)
class Impression:
    impression_id: str
    user_id: str
    timestamp: datetime
    history: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentData:
    training: TrainingData
    evaluation: tuple[Example, ...]


def load_articles(path: Path) -> dict[str, Article]:
    articles: dict[str, Article] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        # Original MIND TSV fields are not CSV-quoted; prose may contain unmatched quotes.
        for line, row in enumerate(
            csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), start=1
        ):
            if len(row) != 8 or not row[0] or not row[1] or not row[3].strip():
                raise DataError(
                    f"{path}:{line}: expected eight fields, article ID, category and title"
                )
            article = Article(row[0], row[1], row[3], row[4])
            if article.article_id in articles:
                raise DataError(f"{path}:{line}: duplicate article {article.article_id}")
            articles[article.article_id] = article
    if not articles:
        raise DataError(f"{path}: empty article corpus")
    return articles


def read_impressions(path: Path) -> Iterator[Impression]:
    seen: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for line, row in enumerate(
            csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE), start=1
        ):
            if len(row) != 5 or not row[0] or not row[1]:
                raise DataError(f"{path}:{line}: expected five fields and nonempty IDs")
            if row[0] in seen:
                raise DataError(f"{path}:{line}: duplicate impression {row[0]}")
            seen.add(row[0])
            try:
                timestamp = datetime.strptime(row[2], "%m/%d/%Y %I:%M:%S %p")
                pairs = [token.rsplit("-", 1) for token in row[4].split()]
                candidate_ids = tuple(pair[0] for pair in pairs)
                labels = tuple(int(pair[1]) for pair in pairs)
            except (ValueError, IndexError) as error:
                raise DataError(f"{path}:{line}: invalid timestamp or candidate label") from error
            if (
                not candidate_ids
                or any(not article_id for article_id in candidate_ids)
                or len(set(candidate_ids)) != len(candidate_ids)
                or any(label not in (0, 1) for label in labels)
            ):
                raise DataError(f"{path}:{line}: empty/duplicate candidates or nonbinary labels")
            yield Impression(
                row[0], row[1], timestamp, tuple(row[3].split()), candidate_ids, labels
            )


def validate_references(impressions: list[Impression], articles: Mapping[str, Article]) -> None:
    for impression in impressions:
        missing = (set(impression.history) | set(impression.candidate_ids)) - articles.keys()
        if missing:
            raise DataError(
                f"Impression {impression.impression_id}: missing articles {sorted(missing)}"
            )


def is_eligible(impression: Impression) -> bool:
    return len(impression.history) >= config.MIN_HISTORY and set(impression.labels) == {0, 1}


def make_example(impression: Impression) -> Example:
    return Example(
        RankingInput(
            impression.impression_id,
            impression.user_id,
            impression.timestamp,
            impression.history[-config.MAX_HISTORY :],
            len(impression.history),
            impression.candidate_ids,
        ),
        Targets(impression.labels),
    )


def chronological_split(
    impressions: list[Impression],
) -> tuple[list[Impression], list[Impression]]:
    if len(impressions) < 2:
        raise DataError("At least two impressions are required for a chronological split")
    ordered = sorted(impressions, key=lambda row: (row.timestamp, row.impression_id))
    cutoff = ordered[math.floor(len(ordered) * (1 - config.DEV_FRACTION))].timestamp
    training = [row for row in ordered if row.timestamp < cutoff]
    evaluation = [row for row in ordered if row.timestamp >= cutoff]
    if not training:
        raise DataError("Timestamp ties leave no chronological training partition")
    return training, evaluation


def select_dev_cohort(impressions: list[Impression]) -> list[Impression]:
    eligible = [row for row in impressions if is_eligible(row)]
    users = sorted({row.user_id for row in eligible})
    if not users:
        raise DataError("No eligible development users")
    selected = set(random.Random(config.SEED).sample(users, min(config.DEV_USERS, len(users))))
    return [row for row in eligible if row.user_id in selected]


def make_training_data(articles: dict[str, Article], impressions: list[Impression]) -> TrainingData:
    # Include users only from fitting interactions; held-out users remain explicitly unseen.
    return TrainingData(
        articles,
        tuple(make_example(row) for row in impressions),
        {user: index for index, user in enumerate(sorted({row.user_id for row in impressions}))},
        {article: index for index, article in enumerate(sorted(articles))},
    )


def load_development() -> ExperimentData:
    """Use only supplied training data; never open validation during development."""
    articles = load_articles(config.TRAIN_DIR / "news.tsv")
    impressions = list(read_impressions(config.TRAIN_DIR / "behaviors.tsv"))
    validate_references(impressions, articles)
    training, held_out = chronological_split(impressions)
    return ExperimentData(
        make_training_data(articles, training),
        tuple(make_example(row) for row in select_dev_cohort(held_out)),
    )
