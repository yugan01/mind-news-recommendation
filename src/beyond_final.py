"""Post-process existing final predictions: python -m src.beyond_final."""

from src import config
from src.beyond import analyze_beyond
from src.final import load_final

if __name__ == "__main__":
    analyze_beyond(load_final(), config.FINAL_OUTPUT_DIR)
