from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PreprocessingConfig:
    normalization_target_sum: float
    genes_per_protein: int
    gene_cap: int


@dataclass(frozen=True)
class AlignmentConfig:
    hidden_dim: int
    latent_dim: int
    projection_dim: int
    depth: int
    dropout: float
    contrastive_weight: float
    rna_zinb_weight: float
    protein_zinb_weight: float
    temperature_init: float
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip: float


@dataclass(frozen=True)
class RetrievalConfig:
    index: str
    neighbors: int
    temperature: float
    exact_cosine: bool
    candidates: int
    reference_degree: int
    consistency_floor: float
    use_consistency: bool
    use_spread_weights: bool


@dataclass(frozen=True)
class CorrectionConfig:
    condition_mode: str
    hidden_dim: int
    depth: int
    time_dim: int
    dropout: float
    mask_probability: float
    mask_disagreement_scale: float
    clean_weight: float
    source_refresh_interval: int
    training_group_exclusion: bool
    finetune_rna_encoder: bool
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    gradient_clip: float
    validation_interval: int
    nfe: int
    standardize_protein: bool = False
    rna_skip: bool = False
    pcc_weight: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    folds: int
    seed: int
    validation_fraction_of_reference: float
    protein_obsm_key: str


@dataclass(frozen=True)
class EpiGradeConfig:
    method: str
    preprocessing: PreprocessingConfig
    alignment: AlignmentConfig
    retrieval: RetrievalConfig
    correction: CorrectionConfig
    experiment: ExperimentConfig


_EXPECTED: dict[str, dict[str, Any]] = {
    "preprocessing": {"normalization_target_sum": 10000.0, "genes_per_protein": 100,
                      "gene_cap": 1000},
    "alignment": {"hidden_dim": 256, "latent_dim": 128, "projection_dim": 64, "depth": 2,
                  "contrastive_weight": 10.0, "epochs": 20, "batch_size": 256},
    "retrieval": {"index": "IndexFlatIP", "neighbors": 5, "candidates": 20,
                  "reference_degree": 20, "use_consistency": False,
                  "use_spread_weights": False},
    "correction": {"condition_mode": "disagreement", "hidden_dim": 512, "depth": 4,
                   "mask_probability": 0.2, "mask_disagreement_scale": 1.0,
                   "clean_weight": 1.0, "source_refresh_interval": 10, "epochs": 50,
                   "training_group_exclusion": False,
                   "batch_size": 1024, "nfe": 1},
    "experiment": {"folds": 5},
}


def _validate(raw: dict[str, Any]) -> None:
    if raw.get("method") != "EpiGrade":
        raise ValueError("The configuration must use method=EpiGrade.")
    for section, expected_values in _EXPECTED.items():
        actual = raw.get(section, {})
        for key, expected in expected_values.items():
            value = actual.get(key)
            if value != expected:
                raise ValueError(
                    f"Configuration mismatch: {section}.{key}={value!r}; "
                    f"expected {expected!r}."
                )


def _select(raw: dict[str, Any], keys) -> dict[str, Any]:
    return {key: raw[key] for key in keys if key in raw}


def load_config(path: str | Path, strict: bool = False) -> EpiGradeConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if strict:
        _validate(raw)
    field_names = lambda cls: [f for f in cls.__dataclass_fields__]
    return EpiGradeConfig(
        method=raw["method"],
        preprocessing=PreprocessingConfig(**_select(raw["preprocessing"],
                                                    field_names(PreprocessingConfig))),
        alignment=AlignmentConfig(**_select(raw["alignment"], field_names(AlignmentConfig))),
        retrieval=RetrievalConfig(**_select(raw["retrieval"], field_names(RetrievalConfig))),
        correction=CorrectionConfig(**_select(raw["correction"],
                                              field_names(CorrectionConfig))),
        experiment=ExperimentConfig(**_select(raw["experiment"],
                                              field_names(ExperimentConfig))),
    )
