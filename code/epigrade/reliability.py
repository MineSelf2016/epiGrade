from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURES = ("max_gene_corr", "top_gene_corr", "celltype_ratio", "detection",
            "rel_disagreement")


@dataclass(frozen=True)
class PanelReliability:
    proteins: list[str]
    score: np.ndarray
    features: dict[str, np.ndarray]
    recommended: np.ndarray
    threshold: float

    def report(self) -> list[str]:
        return [p for p, keep in zip(self.proteins, self.recommended) if keep]

    def as_table(self):
        import pandas as pd
        return pd.DataFrame({"protein": self.proteins, "reliability": self.score,
                             "recommended": self.recommended, **self.features})


def atlas_features(rna_normalized: np.ndarray, protein_normalized: np.ndarray,
                   cell_types: np.ndarray | None = None,
                   disagreement: np.ndarray | None = None,
                   max_cells: int = 20000, max_genes: int = 2000,
                   seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rna = np.asarray(rna_normalized, dtype=np.float32)
    protein = np.asarray(protein_normalized, dtype=np.float32)

    variance = rna.var(axis=0)
    genes = np.argsort(variance)[::-1][:min(max_genes, rna.shape[1])]
    rows = (rng.choice(rna.shape[0], max_cells, replace=False)
            if rna.shape[0] > max_cells else np.arange(rna.shape[0]))
    rna_block, protein_block = rna[np.ix_(rows, genes)], protein[rows]

    rna_c = rna_block - rna_block.mean(0)
    protein_c = protein_block - protein_block.mean(0)
    correlations = np.abs(
        (rna_c.T @ protein_c)
        / (np.maximum(np.sqrt((rna_c ** 2).sum(0)), 1e-8)[:, None]
           * np.maximum(np.sqrt((protein_c ** 2).sum(0)), 1e-8)[None, :])
    )

    features = {
        "max_gene_corr": correlations.max(axis=0),
        "top_gene_corr": np.sort(correlations, axis=0)[-10:].mean(axis=0),
        "detection": (protein > 0).mean(axis=0),
    }

    if cell_types is not None:
        labels = np.asarray(cell_types)
        total = protein.var(axis=0) + 1e-8
        grand = protein.mean(axis=0)
        between = np.zeros(protein.shape[1])
        for level in np.unique(labels):
            mask = labels == level
            if mask.sum() >= 20:
                between += mask.mean() * (protein[mask].mean(axis=0) - grand) ** 2
        features["celltype_ratio"] = between / total

    if disagreement is not None:
        spread = np.maximum(protein.std(axis=0), 1e-8)
        features["rel_disagreement"] = np.asarray(disagreement) / spread
    return features


class ReliabilityEstimator:

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.columns: list[str] = []
        self._centre: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._weights: np.ndarray | None = None
        self._intercept: float = 0.0

    @staticmethod
    def _matrix(features: dict[str, np.ndarray], columns) -> np.ndarray:
        return np.column_stack([np.asarray(features[c], dtype=np.float64)
                                for c in columns])

    def fit(self, features: dict[str, np.ndarray], observed_pcc: np.ndarray
            ) -> "ReliabilityEstimator":
        self.columns = [c for c in FEATURES if c in features]
        if not self.columns:
            raise ValueError("No usable features were supplied.")
        x = self._matrix(features, self.columns)
        y = np.asarray(observed_pcc, dtype=np.float64)
        self._centre, self._scale = x.mean(0), np.maximum(x.std(0), 1e-8)
        z = (x - self._centre) / self._scale
        gram = z.T @ z + self.alpha * np.eye(z.shape[1])
        self._weights = np.linalg.solve(gram, z.T @ (y - y.mean()))
        self._intercept = float(y.mean())
        return self

    def predict(self, features: dict[str, np.ndarray]) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Call fit before predict.")
        missing = [c for c in self.columns if c not in features]
        if missing:
            raise ValueError(f"Missing features at prediction time: {missing}")
        z = (self._matrix(features, self.columns) - self._centre) / self._scale
        return np.clip(z @ self._weights + self._intercept, 0.0, 1.0)

    def assess(self, proteins, features: dict[str, np.ndarray],
               threshold: float = 0.5) -> PanelReliability:
        score = self.predict(features)
        return PanelReliability(proteins=list(proteins), score=score,
                                features={c: np.asarray(features[c])
                                          for c in self.columns},
                                recommended=score >= threshold,
                                threshold=float(threshold))


def rank_only(features: dict[str, np.ndarray], key: str = "rel_disagreement"
              ) -> np.ndarray:
    if key not in features:
        raise KeyError(f"{key} was not computed.")
    values = np.asarray(features[key], dtype=np.float64)
    order = values.argsort().argsort() / max(len(values) - 1, 1)
    return 1.0 - order if key == "rel_disagreement" else order
