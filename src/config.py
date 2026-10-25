"""The team's fixed experiment settings. Change these here, not per model."""

from pathlib import Path

from src.contracts import Model
from src.models.bpr_mf import BPRMFModel
from src.models.lightgcn import LightGCNModel
from src.models.popularity import PopularityModel
from src.models.semantic import SemanticModel

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data/raw/train"
VALIDATION_DIR = ROOT / "data/raw/validation"
DEV_OUTPUT_DIR = ROOT / "outputs/dev"
FINAL_OUTPUT_DIR = ROOT / "outputs/final"
SMOKE_OUTPUT_DIR = ROOT / "outputs/smoke"

DEV_FRACTION = 0.2
MIN_HISTORY = 10
MAX_HISTORY = 20
DEV_USERS = 5000
SEED = 4285
BOOTSTRAP_ITERATIONS = 10000

BPR_DIM = 64
BPR_EPOCHS = 20
BPR_NEGATIVES = 1
BPR_LEARNING_RATE = 0.001
BPR_L2 = 0.0001
BPR_INIT_STD = 0.1
LIGHTGCN_LAYERS = 2

SEMANTIC_ENCODER = "google/embeddinggemma-2"
SEMANTIC_CACHE = ROOT / "outputs/cache/embeddinggemma-2.npz"
SEMANTIC_BATCH_SIZE = 8
SEMANTIC_CPU_THREADS = 2

MODELS: tuple[type[Model], ...] = (
    PopularityModel,
    SemanticModel,
    BPRMFModel,
    LightGCNModel,
)
