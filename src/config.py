"""The team's fixed experiment settings. Change these here, not per model."""

from pathlib import Path

from src.contracts import Model
from src.models.popularity import PopularityModel

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/train"
VALIDATION_DIR = ROOT / "data/raw/validation"
DEV_OUTPUT_DIR = ROOT / "outputs/dev"
FINAL_OUTPUT_DIR = ROOT / "outputs/final"

DEV_FRACTION = 0.2
MIN_HISTORY = 10
MAX_HISTORY = 20
DEV_USERS = 5000
SEED = 4285
BOOTSTRAP_ITERATIONS = 10000

# Add future models to this fixed list when they are ready for comparison.
MODELS: tuple[type[Model], ...] = (PopularityModel,)
