from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

_MMD_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)


def _correlation_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_centered = a - a.mean(axis=1, keepdims=True)
    b_centered = b - b.mean(axis=1, keepdims=True)
    numerator = np.sum(a_centered * b_centered, axis=1)
    denominator = np.sqrt(np.sum(a_centered**2, axis=1) * np.sum(b_centered**2, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                     where=denominator > 1e-12)


def _correlation_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _correlation_rows(a.T, b.T)


def protein_pearson(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _correlation_columns(np.asarray(prediction, np.float32),
                                np.asarray(target, np.float32))


def cell_pearson(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return _correlation_rows(np.asarray(prediction, np.float32),
                             np.asarray(target, np.float32))


def protein_pearson_mean(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")
    return float(protein_pearson(prediction, target).mean())


def protein_spearman(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    ranked_prediction = np.apply_along_axis(rankdata, 0, prediction)
    ranked_target = np.apply_along_axis(rankdata, 0, target)
    return _correlation_columns(ranked_prediction, ranked_target)


def _correlation_matrix(array: np.ndarray, row_variables: bool) -> np.ndarray:
    values = array if row_variables else array.T
    return np.nan_to_num(np.corrcoef(values), nan=0.0, posinf=0.0, neginf=0.0)


def correlation_matrix_distance(a: np.ndarray, b: np.ndarray) -> float:
    numerator = float(np.sum(a * b))
    denominator = float(np.linalg.norm(a, "fro") * np.linalg.norm(b, "fro"))
    return 1.0 - numerator / max(denominator, 1e-12)


def central_moment_discrepancy(prediction: np.ndarray, target: np.ndarray,
                               order: int = 5) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    prediction_mean = prediction.mean(axis=0)
    target_mean = target.mean(axis=0)
    discrepancy = np.abs(prediction_mean - target_mean)
    prediction_centered = prediction - prediction_mean
    target_centered = target - target_mean
    for moment in range(2, order + 1):
        discrepancy += np.abs((prediction_centered**moment).mean(axis=0)
                              - (target_centered**moment).mean(axis=0))
    return float(discrepancy.mean())


def _median_bandwidth(x: np.ndarray, y: np.ndarray, rng) -> float:
    pooled = np.vstack([x, y])
    if pooled.shape[0] > 1000:
        pooled = pooled[rng.choice(pooled.shape[0], 1000, replace=False)]
    squared = np.maximum(
        np.sum(pooled**2, axis=1, keepdims=True)
        + np.sum(pooled**2, axis=1, keepdims=True).T
        - 2.0 * (pooled @ pooled.T),
        0.0,
    )
    upper = squared[np.triu_indices_from(squared, k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    return max(median, 1e-8)


def rbf_mmd(prediction: np.ndarray, target: np.ndarray, max_samples: int = 2000,
            seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape[0] > max_samples:
        prediction = prediction[rng.choice(prediction.shape[0], max_samples, replace=False)]
    if target.shape[0] > max_samples:
        target = target[rng.choice(target.shape[0], max_samples, replace=False)]

    bandwidth = _median_bandwidth(prediction, target, rng)

    def squared_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(
            np.sum(a**2, axis=1, keepdims=True)
            + np.sum(b**2, axis=1, keepdims=True).T
            - 2.0 * (a @ b.T),
            0.0,
        )

    d_xx = squared_distances(prediction, prediction)
    d_yy = squared_distances(target, target)
    d_xy = squared_distances(prediction, target)

    total = 0.0
    for scale in _MMD_SCALES:
        gamma = 1.0 / (scale * bandwidth)
        total += (np.exp(-gamma * d_xx).mean()
                  + np.exp(-gamma * d_yy).mean()
                  - 2.0 * np.exp(-gamma * d_xy).mean())
    return float(total / len(_MMD_SCALES))


def normalised_wasserstein(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    out = np.empty(prediction.shape[1], dtype=np.float64)
    for index in range(prediction.shape[1]):
        column = target[:, index]
        spread = column.std()
        distance = wasserstein_distance(prediction[:, index], column)
        out[index] = distance / max(spread, 1e-8)
    return out


def evaluate(prediction: np.ndarray, target: np.ndarray, cell_matrix_limit: int = 5000,
             seed: int = 42) -> dict[str, object]:
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes must match.")

    pcc_protein = protein_pearson(prediction, target)
    pcc_cell = cell_pearson(prediction, target)
    scc_protein = protein_spearman(prediction, target)

    protein_corr_prediction = _correlation_matrix(prediction, row_variables=False)
    protein_corr_target = _correlation_matrix(target, row_variables=False)

    rng = np.random.default_rng(seed)
    if prediction.shape[0] > cell_matrix_limit:
        selected = rng.choice(prediction.shape[0], cell_matrix_limit, replace=False)
        prediction_cells, target_cells = prediction[selected], target[selected]
    else:
        prediction_cells, target_cells = prediction, target
    cell_corr_prediction = _correlation_matrix(prediction_cells, row_variables=True)
    cell_corr_target = _correlation_matrix(target_cells, row_variables=True)

    nw1 = normalised_wasserstein(prediction, target)
    return {
        "pcc_protein": pcc_protein.astype(np.float32),
        "pcc_cell": pcc_cell.astype(np.float32),
        "scc_protein": scc_protein.astype(np.float32),
        "pcc_protein_mean": float(pcc_protein.mean()),
        "pcc_cell_mean": float(pcc_cell.mean()),
        "scc_protein_mean": float(scc_protein.mean()),
        "cmd_protein": correlation_matrix_distance(protein_corr_prediction,
                                                   protein_corr_target),
        "cmd_cell": correlation_matrix_distance(cell_corr_prediction, cell_corr_target),
        "cmod_protein": central_moment_discrepancy(prediction, target, order=5),
        "mmd": rbf_mmd(prediction, target, seed=seed),
        "nw1": nw1.astype(np.float32),
        "nw1_mean": float(nw1.mean()),
        "mse": float(np.mean((prediction - target) ** 2)),
        "rmse": float(np.sqrt(np.mean((prediction - target) ** 2))),
    }
