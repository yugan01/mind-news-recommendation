"""The team's fixed experiment settings. Change these here, not per model."""

from pathlib import Path

from src.contracts import Model
from src.models.popularity import PopularityModel
from src.models.semantic import SemanticModel

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

SEMANTIC_ENCODER = "google/embeddinggemma-2"
SEMANTIC_CACHE = ROOT / "outputs/cache/embeddinggemma-2.npz"
SEMANTIC_BATCH_SIZE = 8
SEMANTIC_CPU_THREADS = 2

# Add future models to this fixed list when they are ready for comparison.
MODELS: tuple[type[Model], ...] = (PopularityModel, SemanticModel)
