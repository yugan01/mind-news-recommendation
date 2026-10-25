"""Run the fixed development experiment: python -m src.run."""

import json
from pathlib import Path

from src import config
from src.data import ExperimentData, load_development
from src.evaluate import evaluate_model


def run_experiment(data: ExperimentData, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "settings": {
            "dev_fraction": config.DEV_FRACTION,
            "min_history": config.MIN_HISTORY,
            "max_history": config.MAX_HISTORY,
            "dev_users": config.DEV_USERS,
            "seed": config.SEED,
        },
        "training_impressions": len(data.training.examples),
        "models": {},
    }
    results: dict[str, dict[str, int | float]] = {}
    for model_type in config.MODELS:
        model = model_type()
        model.fit(data.training)
        metrics = evaluate_model(model, data.evaluation, output_dir / model.name)
        results[model.name] = metrics
        print(f"{model.name}: {metrics}")
    summary["models"] = results
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    run_experiment(load_development(), config.DEV_OUTPUT_DIR)


if __name__ == "__main__":
    main()
