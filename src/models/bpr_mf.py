"""Matrix factorization trained with the shared BPR objective."""

from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from src import config
from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import EmbeddingProvider, build_pair_pool, train_bpr
from src.models.collaborative import (
    CollaborativeState,
    make_collaborative_state,
    score_collaborative,
)


class BPRMFModel:
    def __init__(self) -> None:
        self.state: CollaborativeState | None = None
        self.run_metadata: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return "bpr_mf"

    def fit(self, data: TrainingData) -> None:
        self.state = None
        self.run_metadata = {}
        pool = build_pair_pool(data)
        provider = EmbeddingProvider(
            len(data.user_indices), len(data.article_indices), config.BPR_DIM, config.SEED
        )
        result = train_bpr(
            provider,
            pool,
            epochs=config.BPR_EPOCHS,
            learning_rate=config.BPR_LEARNING_RATE,
            l2=config.BPR_L2,
            seed=config.SEED,
        )
        self.state = make_collaborative_state(data, result)
        self.run_metadata = {
            "device": result.device,
            "epochs": len(result.losses),
            "final_loss": result.losses[-1],
            "included_impressions": pool.included_impressions,
            "skipped_impressions": pool.skipped_impressions,
            "trained_users": int(result.trained_users.sum()),
            "trained_articles": int(result.trained_articles.sum()),
        }

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if self.state is None:
            raise DataError("BPR-MF score called before fit")
        return score_collaborative(self.state, inputs)
