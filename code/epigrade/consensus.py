from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import faiss
except ImportError as exc:
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None


def l2_normalize(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, eps)


@dataclass(frozen=True)
class ConsensusResult:
    indices: np.ndarray
    similarities: np.ndarray
    consistency: np.ndarray
    weights: np.ndarray
    protein_prior: np.ndarray
    disagreement: np.ndarray
    mean_similarity: np.ndarray


class ReciprocalProteinMemory:

    def __init__(
        self,
        temperature: float = 1.0,
        candidates: int = 20,
        reference_degree: int = 20,
        consistency_floor: float = 0.05,
        use_consistency: bool = True,
        use_spread_weights: bool = True,
    ) -> None:
        if faiss is None:
            raise ImportError("faiss-cpu is required.") from _FAISS_IMPORT_ERROR
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)
        self.candidates = int(candidates)
        self.reference_degree = int(reference_degree)
        self.consistency_floor = float(consistency_floor)
        self.use_consistency = bool(use_consistency)
        self.use_spread_weights = bool(use_spread_weights)
        self.features: np.ndarray | None = None
        self.proteins: np.ndarray | None = None
        self.index = None
        self.reference_graph: np.ndarray | None = None

    def fit(self, rna_latents: np.ndarray, proteins: np.ndarray) -> None:
        features = l2_normalize(rna_latents)
        proteins = np.asarray(proteins, dtype=np.float32)
        if features.shape[0] != proteins.shape[0]:
            raise ValueError("Latents and proteins require the same row count.")
        self.features = np.ascontiguousarray(features)
        self.proteins = np.ascontiguousarray(proteins)
        self.index = faiss.IndexFlatIP(features.shape[1])
        self.index.add(self.features)

        if self.use_consistency:
            degree = min(self.reference_degree + 1, self.features.shape[0])
            _, graph = self.index.search(self.features, degree)
            self.reference_graph = graph[:, 1:].astype(np.int32)
        else:
            self.reference_graph = None

    def _consistency(self, candidate_indices: np.ndarray, block: int = 512) -> np.ndarray:
        if self.reference_graph is None:
            return np.ones_like(candidate_indices, dtype=np.float32)
        n_queries, n_candidates = candidate_indices.shape
        degree = self.reference_graph.shape[1]
        out = np.empty((n_queries, n_candidates), dtype=np.float32)
        for start in range(0, n_queries, block):
            stop = min(start + block, n_queries)
            candidates = candidate_indices[start:stop]
            neighbours = self.reference_graph[candidates]
            shared = (neighbours[..., None] == candidates[:, None, None, :]).any(-1)
            out[start:stop] = shared.sum(-1) / float(degree)
        return out

    def retrieve(
        self,
        query_latents: np.ndarray,
        neighbors: int,
        self_indices: np.ndarray | None = None,
    ) -> ConsensusResult:
        if self.index is None or self.features is None or self.proteins is None:
            raise RuntimeError("Call fit before retrieve.")
        query = np.ascontiguousarray(l2_normalize(query_latents))
        n_bank = self.features.shape[0]
        n_candidates = min(max(self.candidates, neighbors + 1), n_bank)
        if neighbors >= n_bank:
            raise ValueError("neighbors must be smaller than the memory size.")

        similarities, indices = self.index.search(query, n_candidates)
        if self_indices is not None:
            self_indices = np.asarray(self_indices, dtype=np.int64)
            similarities = np.where(indices == self_indices[:, None], -np.inf, similarities)

        consistency = self._consistency(indices)
        gated = similarities * (self.consistency_floor
                                + (1.0 - self.consistency_floor) * consistency)
        order = np.argsort(-gated, axis=1)[:, :neighbors]
        rows = np.arange(query.shape[0])[:, None]
        indices = indices[rows, order]
        similarities = similarities[rows, order]
        consistency = consistency[rows, order]

        if self.use_spread_weights:
            low = similarities[:, -1:][...]
            spread = np.maximum(similarities[:, :1] - low, 1e-6)
            logits = (similarities - low) / spread / self.temperature
        else:
            logits = similarities / self.temperature
        logits = np.where(np.isfinite(logits), logits, -np.inf)
        logits = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(logits).astype(np.float32)
        if self.use_consistency:
            weights = weights * (self.consistency_floor + consistency).astype(np.float32)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

        neighbour_protein = self.proteins[indices]
        prior = np.einsum("nk,nkp->np", weights, neighbour_protein).astype(np.float32)
        deviation = neighbour_protein - prior[:, None, :]
        disagreement = np.sqrt(
            np.einsum("nk,nkp->np", weights, deviation ** 2)
        ).astype(np.float32)
        finite = np.where(np.isfinite(similarities), similarities, 0.0)
        mean_similarity = np.sum(weights * finite, axis=1, keepdims=True).astype(np.float32)

        return ConsensusResult(
            indices=indices.astype(np.int64),
            similarities=finite.astype(np.float32),
            consistency=consistency.astype(np.float32),
            weights=weights,
            protein_prior=prior,
            disagreement=disagreement,
            mean_similarity=mean_similarity,
        )
