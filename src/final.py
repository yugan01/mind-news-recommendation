"""Final evaluation only: run once after the team freezes model/preprocessing decisions."""

from src import config
from src.analyze import analyze
from src.data import (
    DataError,
    ExperimentData,
    is_eligible,
    load_articles,
    make_example,
    make_training_data,
    read_impressions,
    validate_references,
)
from src.run import run_experiment


def load_final() -> ExperimentData:
    articles = load_articles(config.TRAIN_DIR / "news.tsv")
    training = list(read_impressions(config.TRAIN_DIR / "behaviors.tsv"))
    validate_references(training, articles)
    for article_id, article in load_articles(config.VALIDATION_DIR / "news.tsv").items():
        if article_id in articles and article != articles[article_id]:
            raise DataError(f"Conflicting training/validation metadata for {article_id}")
        articles[article_id] = article
    validation = list(read_impressions(config.VALIDATION_DIR / "behaviors.tsv"))
    validate_references(validation, articles)
    if not training or not validation:
        raise DataError("Training and final validation must be nonempty")
    if max(row.timestamp for row in training) >= min(row.timestamp for row in validation):
        raise DataError("Final validation must be strictly later than training")
    evaluation = tuple(make_example(row) for row in validation if is_eligible(row))
    return ExperimentData(make_training_data(articles, training), evaluation)


def main() -> None:
    run_experiment(load_final(), config.FINAL_OUTPUT_DIR)
    analyze(config.FINAL_OUTPUT_DIR)


if __name__ == "__main__":
    main()
