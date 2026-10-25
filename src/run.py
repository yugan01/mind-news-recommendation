"""Run the fixed development experiment: python -m src.run."""

import json
from collections.abc import Mapping
from pathlib import Path

from src import config
from src.data import DataError, ExperimentData, load_development
from src.evaluate import evaluate_model


def run_experiment(data: ExperimentData, output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise DataError(f"Refusing to mix fresh results with existing output: {output_dir}")
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
    settings = summary["settings"]
    if not isinstance(settings, dict):
        raise TypeError("Experiment settings must be a dictionary")
    settings.update(
        {
            "bpr_dimension": config.BPR_DIM,
            "bpr_epochs": config.BPR_EPOCHS,
            "bpr_negatives": config.BPR_NEGATIVES,
            "bpr_learning_rate": config.BPR_LEARNING_RATE,
            "bpr_l2": config.BPR_L2,
            "bpr_init_std": config.BPR_INIT_STD,
            "bpr_optimizer": "Adam",
            "lightgcn_layers": config.LIGHTGCN_LAYERS,
            "lightgcn_layer_combination": "uniform_mean",
            "two_tower_input_dimension": 768,
            "two_tower_hidden": config.TWO_TOWER_HIDDEN,
            "two_tower_output": config.TWO_TOWER_OUTPUT,
            "two_tower_epochs": config.TWO_TOWER_EPOCHS,
            "two_tower_learning_rate": config.TWO_TOWER_LEARNING_RATE,
            "two_tower_l2": config.TWO_TOWER_L2,
            "two_tower_optimizer": "Adam",
            "two_tower_frozen_encoder": config.SEMANTIC_ENCODER,
        }
    )
    results: dict[str, dict[str, int | float]] = {}
    training_metadata: dict[str, dict[str, object]] = {}
    for model_type in config.MODELS:
        model = model_type()
        model.fit(data.training)
        metrics = evaluate_model(model, data.evaluation, output_dir / model.name)
        metadata = getattr(model, "run_metadata", None)
        results[model.name] = metrics
        training_metadata[model.name] = (
            dict(metadata) if isinstance(metadata, Mapping) else {}
        )
        print(f"{model.name}: {metrics}")
    summary["models"] = results
    summary["training"] = training_metadata
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    run_experiment(load_development(), config.DEV_OUTPUT_DIR)


if __name__ == "__main__":
    main()
