"""Streaming readers for the MIND news and behavior files."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from itertools import chain, groupby
from pathlib import Path
from typing import Iterator, Iterable


@dataclass(frozen=True)
class Article:
    article_id: str
    text: str


@dataclass(frozen=True)
class Candidate:
    article_id: str
    label: int


@dataclass(frozen=True)
class Impression:
    impression_id: str
    user_id: str
    history: tuple[str, ...]
    candidates: tuple[Candidate, ...]


def load_articles(path: Path) -> dict[str, Article]:
    articles: dict[str, Article] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        first = next(reader, None)
        if first is None:
            return articles
        if first[0] in {"news_id", "article_id"}:
            header = first
            article_col = header.index(first[0])
            title_col = header.index("title")
            abstract_col = header.index("abstract")
            rows: Iterable[list[str]] = reader
        else:
            article_col, title_col, abstract_col = 0, 3, 4
            rows = chain([first], reader)
        for row in rows:
            if len(row) <= max(article_col, title_col, abstract_col):
                continue
            article_id = row[article_col]
            title = row[title_col]
            abstract = row[abstract_col]
            articles[article_id] = Article(article_id, f"{title} {abstract}".strip())
    return articles


def parse_candidate(token: str) -> Candidate:
    article_id, label = token.rsplit("-", 1)
    return Candidate(article_id=article_id, label=int(label))


def iter_impressions(path: Path) -> Iterator[Impression]:
    if path.suffix.lower() == ".csv":
        yield from iter_reczoo_impressions(path)
        return
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 5:
                continue
            history = tuple(row[3].split()) if row[3] else ()
            candidates = tuple(parse_candidate(token) for token in row[4].split())
            yield Impression(row[0], row[1], history, candidates)


def iter_reczoo_impressions(path: Path) -> Iterator[Impression]:
    """Reconstruct impressions from RecZoo's one-candidate-per-row files."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for impression_id, rows_iter in groupby(reader, key=lambda row: row["imp_id"]):
            rows = list(rows_iter)
            first = rows[0]
            history = tuple(filter(None, first.get("news_his", "").split("^")))
            candidates = tuple(
                Candidate(row["news_id"], int(row["click"])) for row in rows
            )
            yield Impression(impression_id, first["user_id"], history, candidates)


def is_eligible(impression: Impression, min_history: int) -> bool:
    return (
        len(impression.history) >= min_history
        and any(candidate.label == 1 for candidate in impression.candidates)
        and any(candidate.label == 0 for candidate in impression.candidates)
    )


def iter_eligible_impressions(
    path: Path,
    min_history: int = 10,
    selected_users: set[str] | None = None,
) -> Iterator[Impression]:
    for impression in iter_impressions(path):
        if selected_users is not None and impression.user_id not in selected_users:
            continue
        if is_eligible(impression, min_history):
            yield impression


def sample_eligible_users(
    path: Path,
    count: int,
    min_history: int = 10,
    seed: int = 4285,
) -> set[str]:
    users = {
        impression.user_id
        for impression in iter_eligible_impressions(path, min_history=min_history)
    }
    if count >= len(users):
        return users
    return set(random.Random(seed).sample(sorted(users), count))


def eligible_impressions(
    path: Path,
    min_history: int = 10,
    sample_users: int | None = None,
    seed: int = 4285,
) -> list[Impression]:
    rows = list(iter_eligible_impressions(path, min_history=min_history))
    if sample_users is None:
        return rows

    users = sorted({row.user_id for row in rows})
    if sample_users >= len(users):
        return rows

    selected = set(random.Random(seed).sample(users, sample_users))
    return [row for row in rows if row.user_id in selected]

