"""Real-MIND functionality check: python -m scripts.smoke_mind.

Uses two training and two development impressions, with only their referenced
articles. This is not a model comparison experiment; never opens validation.
"""

from unittest.mock import patch

from src import config
from src.analyze import analyze
from src.beyond import analyze_beyond
from src.contracts import TrainingData
from src.data import DataError, ExperimentData, load_development
from src.models.bpr_mf import BPRMFModel
from src.models.lightgcn import LightGCNModel
from src.models.popularity import PopularityModel
from src.models.semantic import SemanticModel
from src.models.two_tower import TwoTowerModel
from src.run import run_experiment


def load_smoke() -> ExperimentData:
    data = load_development()
    training = tuple(
        example for example in data.training.examples if set(example.targets.labels) == {0, 1}
    )[:2]
    if len(training) != 2:
        raise DataError("Real-MIND smoke needs two mixed-label fitting impressions")
    evaluation = data.evaluation[:2]
    article_ids = sorted(
        {
            article
            for example in (*training, *evaluation)
            for article in (*example.inputs.history, *example.inputs.candidate_ids)
        }
    )
    users = sorted({example.inputs.user_id for example in training})
    return ExperimentData(
        TrainingData(
            {article: data.training.articles[article] for article in article_ids},
            training,
            {user: index for index, user in enumerate(users)},
            {article: index for index, article in enumerate(article_ids)},
        ),
        evaluation,
    )


def main() -> None:
    data = load_smoke()
    output = config.SMOKE_OUTPUT_DIR / "mind"
    # Isolate smoke caches/exports and keep future trained models out of this tiny check.
    with patch.multiple(
        config,
        SEMANTIC_CACHE=output / "vectors.npz",
        MODELS=(
            PopularityModel,
            SemanticModel,
            BPRMFModel,
            LightGCNModel,
            TwoTowerModel,
        ),
        BPR_EPOCHS=2,
        BPR_DIM=8,
        LIGHTGCN_LAYERS=1,
        TWO_TOWER_EPOCHS=2,
    ):
        run_experiment(data, output)
        analyze(output)
        analyze_beyond(data, output)
    print(f"Real-MIND smoke passed: {len(data.training.articles)} articles, two dev impressions.")
    print("Smoke metrics are not reportable experiment results; validation was not opened.")


if __name__ == "__main__":
    main()
