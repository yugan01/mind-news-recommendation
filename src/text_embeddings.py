"""Frozen text encoding and validated article-vector cache."""

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from numpy.typing import NDArray

from src.contracts import Article
from src.data import DataError

Encoder = Callable[[list[str]], NDArray[np.float32]]


def encode_articles(texts: list[str]) -> NDArray[np.float32]:
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


def load_article_vectors(
    articles: Mapping[str, Article], cache_path: Path, encoder: Encoder
) -> tuple[dict[str, int], NDArray[np.float32]]:
    article_ids = sorted(articles)
    texts = [articles[article_id].text for article_id in article_ids]
    article_indices = {article_id: index for index, article_id in enumerate(article_ids)}
    vectors = np.empty((len(article_ids), 768), dtype=np.float32)
    cached: dict[str, int] = {}
    cached_texts: list[str] = []
    cached_vectors: NDArray[np.float32] = np.empty((0, 768), dtype=np.float32)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            if set(cache.files) != {"article_ids", "texts", "vectors"}:
                raise DataError(f"Invalid semantic cache fields: {cache_path}")
            cached = {article: index for index, article in enumerate(cache["article_ids"])}
            cached_texts = cache["texts"].tolist()
            cached_vectors = np.asarray(cache["vectors"], dtype=np.float32)
            if (
                cached_vectors.shape != (len(cached), 768)
                or len(cached_texts) != len(cached)
                or not np.isfinite(cached_vectors).all()
            ):
                raise DataError(f"Invalid semantic cache shape: {cache_path}")
    missing: list[int] = []
    for index, article_id in enumerate(article_ids):
        cached_index = cached.get(article_id)
        if cached_index is not None and cached_texts[cached_index] == texts[index]:
            vectors[index] = cached_vectors[cached_index]
        else:
            missing.append(index)
    if missing:
        encoded = np.asarray(encoder([texts[index] for index in missing]), dtype=np.float32)
        if encoded.shape != (len(missing), 768) or not np.isfinite(encoded).all():
            raise DataError("Text encoder returned invalid article vectors")
        vectors[missing] = encoded
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            article_ids=np.asarray(article_ids),
            texts=np.asarray(texts),
            vectors=vectors,
        )
    return article_indices, vectors
