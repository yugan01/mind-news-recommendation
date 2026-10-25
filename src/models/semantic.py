"""Frozen article embeddings; unseen users need only their supplied history.

All history/candidate articles must be in the shared metadata corpus. Empty or
zero-vector histories receive tied zero scores, preserving candidate order.
"""

import numpy as np
from numpy.typing import NDArray

from src.contracts import RankingInput, TrainingData


def encode_articles(texts: list[str]) -> NDArray[np.float32]:
    # Import only on a cache miss, so cached runs do not load the encoder.
    import torch
    from sentence_transformers import SentenceTransformer

    from src import config

    torch.set_num_threads(config.SEMANTIC_CPU_THREADS)
    encoder = SentenceTransformer(
        config.SEMANTIC_ENCODER,
        device="cpu",
        config_kwargs={"vision_config": None, "audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    return np.asarray(
        encoder.encode(
            texts,
            prompt_name="SentenceSimilarity",
            batch_size=config.SEMANTIC_BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )


class SemanticModel:
    @property
    def name(self) -> str:
        return "semantic"

    def __init__(self) -> None:
        self.article_indices: dict[str, int] = {}
        self.vectors: NDArray[np.float32] = np.empty((0, 768), dtype=np.float32)

    def fit(self, data: TrainingData) -> None:
        from src import config

        article_ids = sorted(data.articles)
        texts = [data.articles[article_id].text for article_id in article_ids]
        self.article_indices = {article_id: index for index, article_id in enumerate(article_ids)}
        self.vectors = np.empty((len(article_ids), 768), dtype=np.float32)
        cached: dict[str, int] = {}
        cached_texts: list[str] = []
        cached_vectors: NDArray[np.float32] = np.empty((0, 768), dtype=np.float32)
        if config.SEMANTIC_CACHE.exists():
            with np.load(config.SEMANTIC_CACHE, allow_pickle=False) as cache:
                cached = {article: index for index, article in enumerate(cache["article_ids"])}
                cached_texts = cache["texts"].tolist()
                cached_vectors = cache["vectors"]
        missing: list[int] = []
        for index, article_id in enumerate(article_ids):
            cached_index = cached.get(article_id)
            if cached_index is not None and cached_texts[cached_index] == texts[index]:
                self.vectors[index] = cached_vectors[cached_index]
            else:
                missing.append(index)
        if missing:
            self.vectors[missing] = encode_articles([texts[index] for index in missing])
            config.SEMANTIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                config.SEMANTIC_CACHE,
                article_ids=np.asarray(article_ids),
                texts=np.asarray(texts),
                vectors=self.vectors,
            )

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if not inputs.history:
            return np.zeros(len(inputs.candidate_ids), dtype=np.float64)
        history = [self.article_indices[article] for article in inputs.history]
        profile = self.vectors[history].mean(axis=0)
        norm = float(np.linalg.norm(profile))
        if norm == 0:
            return np.zeros(len(inputs.candidate_ids), dtype=np.float64)
        candidates = [self.article_indices[article] for article in inputs.candidate_ids]
        return np.asarray(self.vectors[candidates] @ (profile / norm), dtype=np.float64)
