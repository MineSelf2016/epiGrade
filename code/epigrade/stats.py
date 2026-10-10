from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TestResult:
    comparison: str
    mean_difference: float
    statistic: float
    pvalue: float
    degrees_of_freedom: int
    method: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def paired_ttest(differences, comparison: str = "") -> TestResult:
    differences = np.asarray(differences, dtype=np.float64)
    statistic, pvalue = stats.ttest_1samp(differences, 0.0)
    return TestResult(comparison, float(differences.mean()), float(statistic),
                      float(pvalue), differences.size - 1, "paired t-test")


def corrected_resampled_ttest(differences, n_train: int, n_test: int,
                              comparison: str = "") -> TestResult:
    differences = np.asarray(differences, dtype=np.float64)
    k = differences.size
    if k < 2:
        raise ValueError("At least two folds are required.")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("Training and test sizes must be positive.")
    variance = differences.var(ddof=1)
    denominator = np.sqrt(max((1.0 / k) + (n_test / n_train), 1e-12) * variance)
    if denominator <= 0:
        statistic = np.inf if differences.mean() > 0 else -np.inf
        pvalue = 0.0
    else:
        statistic = float(differences.mean() / denominator)
        pvalue = float(2.0 * stats.t.sf(abs(statistic), df=k - 1))
    return TestResult(comparison, float(differences.mean()), float(statistic), pvalue,
                      k - 1, "Nadeau-Bengio corrected resampled t-test")


def holm_bonferroni(results: list[TestResult]) -> list[dict[str, object]]:
    order = np.argsort([result.pvalue for result in results])
    family_size = len(results)
    adjusted: list[float] = [0.0] * family_size
    running = 0.0
    for rank, index in enumerate(order):
        value = (family_size - rank) * results[index].pvalue
        running = max(running, min(value, 1.0))
        adjusted[index] = running
    return [{**result.as_dict(), "pvalue_holm": adjusted[index],
             "significant_0.05": adjusted[index] < 0.05}
            for index, result in enumerate(results)]


def cluster_bootstrap_ci(prediction: np.ndarray, target: np.ndarray, groups,
                         metric, n_boot: int = 2000, alpha: float = 0.05,
                         seed: int = 42) -> dict[str, float]:
    groups = np.asarray(groups)
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError(
            "A cluster bootstrap needs at least two groups; report the point "
            "estimate instead when the dataset has a single donor."
        )
    index_by_group = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    observed = float(metric(prediction, target))
    samples = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_by_group[group] for group in chosen])
        samples[draw] = metric(prediction[rows], target[rows])
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"estimate": observed, "ci_lower": float(lower), "ci_upper": float(upper),
             "n_groups": int(unique.size), "n_boot": int(n_boot),
             "method": "cluster bootstrap over groups"}


def cell_bootstrap_ci(prediction: np.ndarray, target: np.ndarray, metric,
                      n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = prediction.shape[0]
    samples = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        rows = rng.integers(0, n, size=n)
        samples[draw] = metric(prediction[rows], target[rows])
    lower, upper = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {"estimate": float(metric(prediction, target)), "ci_lower": float(lower),
            "ci_upper": float(upper), "n_boot": int(n_boot),
            "method": "cell-level bootstrap (pseudo-replicated)"}
