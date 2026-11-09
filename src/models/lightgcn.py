"""LightGCN graph construction, propagation, and recommender wrapper."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from src import config
from src.contracts import RankingInput, TrainingData
from src.data import DataError
from src.models.bpr import EmbeddingProvider, build_pair_pool, train_bpr
from src.models.collaborative import (
    CollaborativeState,
    make_collaborative_state,
    score_collaborative,
)


@dataclass(frozen=True)
class GraphData:
    adjacency: Tensor
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    num_users: int
    num_articles: int


def build_graph(data: TrainingData) -> GraphData:
    num_users = len(data.user_indices)
    num_articles = len(data.article_indices)
    edges: set[tuple[int, int]] = set()
    for example in data.examples:
        try:
            user = data.user_indices[example.inputs.user_id]
            positives = [
                article
                for article, label in zip(
                    example.inputs.candidate_ids, example.targets.labels, strict=True
                )
                if label == 1
            ]
            for article in (*example.inputs.history, *positives):
                edges.add((user, data.article_indices[article]))
        except KeyError as error:
            raise DataError(
                f"LightGCN index mapping is missing {error.args[0]}"
            ) from error
    if not edges:
        raise DataError("LightGCN graph has no fitting interactions")
    undirected = [(left, num_users + right) for left, right in sorted(edges)]
    undirected += [(right, left) for left, right in undirected]
    rows = np.asarray([row for row, _ in undirected], dtype=np.int64)
    columns = np.asarray([column for _, column in undirected], dtype=np.int64)
    degrees = np.bincount(rows, minlength=num_users + num_articles).astype(np.float64)
    values = 1.0 / np.sqrt(degrees[rows] * degrees[columns])
    if not np.isfinite(values).all():
        raise DataError("LightGCN graph normalization is non-finite")
    indices = torch.as_tensor(np.vstack((rows, columns)), dtype=torch.long)
    adjacency = torch.sparse_coo_tensor(
        indices,
        torch.as_tensor(values, dtype=torch.float32),
        (num_users + num_articles, num_users + num_articles),
        check_invariants=True,
    ).coalesce()
    return GraphData(adjacency, rows, columns, num_users, num_articles)


def expand_masks(
    graph: GraphData,
    users: NDArray[np.bool_],
    articles: NDArray[np.bool_],
    layers: int,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    active = np.concatenate((users, articles)).copy()
    for _ in range(layers):
        expanded = active.copy()
        expanded[graph.columns[active[graph.rows]]] = True
        active = expanded
    return active[: graph.num_users], active[graph.num_users :]


class LightGCNEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        num_users: int,
        num_articles: int,
        dimension: int,
        seed: int,
        graph: GraphData,
        layers: int,
    ) -> None:
        super().__init__(num_users, num_articles, dimension, seed)
        self.layers = layers
        self.graph = graph
        self.register_buffer("adjacency", graph.adjacency)

    def forward(self) -> tuple[Tensor, Tensor]:
        layer = torch.cat((self.user_embeddings.weight, self.article_embeddings.weight))
        layers = [layer]
        for _ in range(self.layers):
            layer = torch.sparse.mm(self.adjacency, layer)
            layers.append(layer)
        combined = torch.stack(layers).mean(dim=0)
        return combined[: self.graph.num_users], combined[self.graph.num_users :]

    def expand_trained_masks(
        self, users: NDArray[np.bool_], articles: NDArray[np.bool_]
    ) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
        return expand_masks(self.graph, users, articles, self.layers)


class LightGCNModel:
    def __init__(self) -> None:
        self.state: CollaborativeState | None = None
        self.run_metadata: Mapping[str, object] = {}

    @property
    def name(self) -> str:
        return "lightgcn"

    def fit(self, data: TrainingData) -> None:
        self.state = None
        self.run_metadata = {}
        pool = build_pair_pool(data)
        graph = build_graph(data)
        provider = LightGCNEmbeddings(
            len(data.user_indices),
            len(data.article_indices),
            config.BPR_DIM,
            config.SEED,
            graph,
            config.LIGHTGCN_LAYERS,
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
            "layers": config.LIGHTGCN_LAYERS,
            "unique_graph_edges": len(graph.rows) // 2,
            "included_impressions": pool.included_impressions,
            "skipped_impressions": pool.skipped_impressions,
            "trained_users": int(result.trained_users.sum()),
            "trained_articles": int(result.trained_articles.sum()),
        }

    def score(self, inputs: RankingInput) -> NDArray[np.float64]:
        if self.state is None:
            raise DataError("LightGCN score called before fit")
        return score_collaborative(self.state, inputs)
