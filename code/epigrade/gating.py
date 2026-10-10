from __future__ import annotations

import numpy as np

__all__ = ["otsu_threshold", "quadrant_fractions"]


def otsu_threshold(values, bins: int = 256) -> float:
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite values to threshold")
    counts, edges = np.histogram(values, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight_low = np.cumsum(counts)
    weight_high = weight_low[-1] - weight_low
    total = np.cumsum(counts * centres)
    mean_low = total / np.maximum(weight_low, 1)
    mean_high = (total[-1] - total) / np.maximum(weight_high, 1)
    between = weight_low * weight_high * (mean_low - mean_high) ** 2
    optimal = between >= between.max() * (1.0 - 1e-9)
    return float(np.median(centres[optimal]))


def quadrant_fractions(x, y, threshold_x: float, threshold_y: float) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    high_x, high_y = x > threshold_x, y > threshold_y
    return {"++": float(np.mean(high_x & high_y)),
            "+-": float(np.mean(high_x & ~high_y)),
            "-+": float(np.mean(~high_x & high_y)),
            "--": float(np.mean(~high_x & ~high_y))}
