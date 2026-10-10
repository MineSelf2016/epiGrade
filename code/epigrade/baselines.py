from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _l2_normalise(matrix: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, eps)


def _highly_variable(reference: np.ndarray, n_features: int) -> np.ndarray:
    variance = np.asarray(reference, dtype=np.float64).var(axis=0)
    n_features = min(n_features, variance.size)
    return np.argsort(variance)[::-1][:n_features]


def global_mean_predictor(reference_protein: np.ndarray, n_query: int) -> np.ndarray:
    mean = np.asarray(reference_protein, dtype=np.float32).mean(axis=0, keepdims=True)
    return np.repeat(mean, n_query, axis=0)


def expression_knn_transfer(
    reference_rna: np.ndarray,
    reference_protein: np.ndarray,
    query_rna: np.ndarray,
    neighbors: int = 5,
    space: str = "pca",
    n_components: int = 50,
    n_features: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    reference_rna = np.asarray(reference_rna, dtype=np.float32)
    query_rna = np.asarray(query_rna, dtype=np.float32)
    features = _highly_variable(reference_rna, n_features)
    reference_space = reference_rna[:, features]
    query_space = query_rna[:, features]

    if space == "pca":
        n_components = min(n_components, reference_space.shape[1], reference_space.shape[0] - 1)
        model = PCA(n_components=n_components, random_state=seed)
        reference_space = model.fit_transform(reference_space)
        query_space = model.transform(query_space)
    elif space != "expression":
        raise ValueError("space must be 'pca' or 'expression'.")

    reference_space = _l2_normalise(reference_space)
    query_space = _l2_normalise(query_space)

    finder = NearestNeighbors(n_neighbors=neighbors, metric="euclidean").fit(reference_space)
    distances, indices = finder.kneighbors(query_space)
    weights = 1.0 / np.maximum(distances, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    neighbour_protein = np.asarray(reference_protein, dtype=np.float32)[indices]
    return np.einsum("nk,nkp->np", weights.astype(np.float32), neighbour_protein)


def seurat_anchor_transfer(
    reference_rna: np.ndarray,
    reference_protein: np.ndarray,
    query_rna: np.ndarray,
    n_components: int = 30,
    n_features: int = 2000,
    k_anchor: int = 5,
    k_score: int = 30,
    k_weight: int = 50,
    seed: int = 42,
) -> np.ndarray:
    reference_rna = np.asarray(reference_rna, dtype=np.float32)
    query_rna = np.asarray(query_rna, dtype=np.float32)
    reference_protein = np.asarray(reference_protein, dtype=np.float32)

    features = _highly_variable(reference_rna, n_features)
    n_components = min(n_components, len(features), reference_rna.shape[0] - 1)
    model = PCA(n_components=n_components, random_state=seed)
    reference_space = _l2_normalise(model.fit_transform(reference_rna[:, features]))
    query_space = _l2_normalise(model.transform(query_rna[:, features]))

    k_anchor = min(k_anchor, reference_space.shape[0], query_space.shape[0])
    ref_finder = NearestNeighbors(n_neighbors=k_anchor).fit(reference_space)
    query_finder = NearestNeighbors(n_neighbors=k_anchor).fit(query_space)
    _, query_to_reference = ref_finder.kneighbors(query_space)
    _, reference_to_query = query_finder.kneighbors(reference_space)

    reciprocal = [set(reference_to_query[r].tolist()) for r in range(reference_space.shape[0])]
    anchor_query: list[int] = []
    anchor_reference: list[int] = []
    for q in range(query_space.shape[0]):
        for r in query_to_reference[q]:
            if q in reciprocal[r]:
                anchor_query.append(q)
                anchor_reference.append(int(r))
    if not anchor_query:
        return reference_protein[query_to_reference[:, 0]]

    anchor_query = np.asarray(anchor_query)
    anchor_reference = np.asarray(anchor_reference)

    k_score = min(k_score, reference_space.shape[0], query_space.shape[0])
    _, reference_neighbourhood = NearestNeighbors(n_neighbors=k_score).fit(
        reference_space).kneighbors(reference_space[anchor_reference])
    _, query_neighbourhood = NearestNeighbors(n_neighbors=k_score).fit(
        query_space).kneighbors(query_space[anchor_query])
    scores = np.empty(anchor_query.size, dtype=np.float32)
    for index in range(anchor_query.size):
        shared = np.intersect1d(
            query_to_reference[query_neighbourhood[index]].ravel(),
            reference_neighbourhood[index],
            assume_unique=False,
        )
        scores[index] = shared.size / float(k_score)
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = np.ones_like(scores)

    anchor_space = query_space[anchor_query]
    k_weight = min(k_weight, anchor_space.shape[0])
    distances, neighbours = NearestNeighbors(n_neighbors=k_weight).fit(
        anchor_space).kneighbors(query_space)
    furthest = np.maximum(distances[:, -1:], 1e-8)
    kernel = np.exp(-((distances / furthest) ** 2) / 0.5).astype(np.float32)
    kernel *= scores[neighbours]
    total = kernel.sum(axis=1, keepdims=True)
    kernel = np.divide(kernel, total, out=np.full_like(kernel, 1.0 / k_weight),
                       where=total > 1e-12)
    return np.einsum("nk,nkp->np", kernel,
                     reference_protein[anchor_reference[neighbours]])
