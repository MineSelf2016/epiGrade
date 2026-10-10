from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .aligner import CrossModalAligner
from .config import EpiGradeConfig
from .consensus import ConsensusResult, ReciprocalProteinMemory
from .metrics import protein_pearson_mean
from .objectives import flow_matching_objective, masked_residual_objective


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = [torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in arrays]
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())


def build_memory(config: EpiGradeConfig) -> ReciprocalProteinMemory:
    return ReciprocalProteinMemory(
        temperature=config.retrieval.temperature,
        candidates=config.retrieval.candidates,
        reference_degree=config.retrieval.reference_degree,
        consistency_floor=config.retrieval.consistency_floor,
        use_consistency=config.retrieval.use_consistency,
        use_spread_weights=config.retrieval.use_spread_weights,
    )


@torch.no_grad()
def encode_rna(model: CrossModalAligner, rna: np.ndarray, device: torch.device,
               batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(rna), batch_size):
        batch = torch.from_numpy(rna[start:start + batch_size]).to(device)
        out.append(model.encode_rna(batch).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_alignment(model: CrossModalAligner, rna_normalized, rna_counts,
                    protein_normalized, protein_counts, config: EpiGradeConfig,
                    device: torch.device, log: Callable[[str], None] = print):
    stage = config.alignment
    loader = _loader(rna_normalized, rna_counts, protein_normalized, protein_counts,
                     batch_size=stage.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage.learning_rate,
                                  weight_decay=stage.weight_decay)
    history = []
    for epoch in range(1, stage.epochs + 1):
        model.train()
        sums = {"total": 0.0, "contrastive": 0.0, "rna_zinb": 0.0, "protein_zinb": 0.0}
        batches = 0
        for rna_norm, rna_raw, prot_norm, prot_raw in loader:
            losses = model.objective(
                rna_norm.to(device), rna_raw.to(device), prot_norm.to(device),
                prot_raw.to(device), stage.contrastive_weight, stage.rna_zinb_weight,
                stage.protein_zinb_weight)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), stage.gradient_clip)
            optimizer.step()
            for key in sums:
                sums[key] += float(losses[key].detach())
            batches += 1
        row = {"epoch": float(epoch), **{k: v / batches for k, v in sums.items()}}
        history.append(row)
        log(f"align epoch={epoch:02d}/{stage.epochs} loss={row['total']:.5f} "
            f"contrastive={row['contrastive']:.5f}")
    return history


def retrieve_training_priors(aligner: CrossModalAligner, fit_rna: np.ndarray,
                             fit_protein: np.ndarray, config: EpiGradeConfig,
                             device: torch.device, groups: np.ndarray | None = None):
    latent = encode_rna(aligner, fit_rna, device)
    n_cells = len(fit_rna)

    if groups is None or not config.correction.training_group_exclusion:
        memory = build_memory(config)
        memory.fit(latent, fit_protein)
        return memory.retrieve(latent, config.retrieval.neighbors,
                               self_indices=np.arange(n_cells, dtype=np.int64))

    groups = np.asarray(groups)
    unique = np.unique(groups)
    if unique.size < 2:
        memory = build_memory(config)
        memory.fit(latent, fit_protein)
        return memory.retrieve(latent, config.retrieval.neighbors,
                               self_indices=np.arange(n_cells, dtype=np.int64))

    neighbors = config.retrieval.neighbors
    n_proteins = fit_protein.shape[1]
    indices = np.empty((n_cells, neighbors), dtype=np.int64)
    similarities = np.empty((n_cells, neighbors), dtype=np.float32)
    consistency = np.empty((n_cells, neighbors), dtype=np.float32)
    weights = np.empty((n_cells, neighbors), dtype=np.float32)
    prior = np.empty((n_cells, n_proteins), dtype=np.float32)
    disagreement = np.empty((n_cells, n_proteins), dtype=np.float32)
    mean_similarity = np.empty((n_cells, 1), dtype=np.float32)

    for group in unique:
        held = groups == group
        bank = np.flatnonzero(~held)
        query = np.flatnonzero(held)
        memory = build_memory(config)
        memory.fit(latent[bank], fit_protein[bank])
        result = memory.retrieve(latent[query], neighbors)
        indices[query] = bank[result.indices]
        similarities[query] = result.similarities
        consistency[query] = result.consistency
        weights[query] = result.weights
        prior[query] = result.protein_prior
        disagreement[query] = result.disagreement
        mean_similarity[query] = result.mean_similarity

    return ConsensusResult(indices=indices, similarities=similarities,
                           consistency=consistency, weights=weights,
                           protein_prior=prior, disagreement=disagreement,
                           mean_similarity=mean_similarity)


def neighbourhood_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Neighbour index arrays must have the same shape.")
    out = np.empty(a.shape[0], dtype=np.float64)
    for row in range(a.shape[0]):
        first, second = set(a[row].tolist()), set(b[row].tolist())
        out[row] = len(first & second) / max(len(first | second), 1)
    return float(out.mean())


def corrector_input(aligner: CrossModalAligner, corrector, rna: torch.Tensor) -> torch.Tensor:
    latent = aligner.encode_rna(rna)
    if getattr(corrector, "rna_skip_dim", 0):
        latent = torch.cat([latent, rna], dim=-1)
    return latent


@torch.no_grad()
def predict(corrector, aligner: CrossModalAligner, rna: np.ndarray, prior: np.ndarray,
            uncertainty: np.ndarray, device: torch.device, batch_size: int,
            flow_steps: int | None = None) -> np.ndarray:
    corrector.eval()
    aligner.eval()
    out = []
    for start in range(0, len(rna), batch_size):
        stop = start + batch_size
        latent = corrector_input(aligner, corrector,
                                 torch.from_numpy(rna[start:stop]).to(device))
        source = torch.from_numpy(prior[start:stop]).to(device)
        extra = torch.from_numpy(uncertainty[start:stop]).to(device)
        if flow_steps is None:
            out.append(corrector.predict(source, latent, extra).cpu().numpy())
        else:
            out.append(corrector.predict(source, latent, extra, steps=flow_steps).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_correction(aligner: CrossModalAligner, corrector, fit_rna, fit_protein,
                     validation_rna, validation_protein, initial_prior,
                     initial_uncertainty, config: EpiGradeConfig, device: torch.device,
                     log: Callable[[str], None] = print, use_flow: bool = False,
                     mask_disagreement_scale: float | None = None,
                     mask_probability: float | None = None,
                     fit_groups: np.ndarray | None = None):
    stage = config.correction
    scale = stage.mask_disagreement_scale if mask_disagreement_scale is None \
        else mask_disagreement_scale
    rate = stage.mask_probability if mask_probability is None else mask_probability

    for parameter in aligner.parameters():
        parameter.requires_grad = False
    for parameter in aligner.rna_encoder.parameters():
        parameter.requires_grad = stage.finetune_rna_encoder

    trainable = list(corrector.parameters()) + (
        list(aligner.rna_encoder.parameters()) if stage.finetune_rna_encoder else [])
    optimizer = torch.optim.AdamW(trainable, lr=stage.learning_rate,
                                  weight_decay=stage.weight_decay)

    prior = np.asarray(initial_prior, dtype=np.float32)
    uncertainty = np.asarray(initial_uncertainty, dtype=np.float32)
    loader = _loader(fit_rna, prior, uncertainty, fit_protein,
                     batch_size=stage.batch_size, shuffle=True)

    best_pcc, best_epoch = -np.inf, 0
    best_corrector = deepcopy(corrector.state_dict())
    best_encoder = deepcopy(aligner.rna_encoder.state_dict())
    initial_neighbours, history = None, []

    for epoch in range(1, stage.epochs + 1):
        if (stage.source_refresh_interval and epoch > 1
                and (epoch - 1) % stage.source_refresh_interval == 0):
            refreshed = retrieve_training_priors(aligner, fit_rna, fit_protein, config,
                                                 device, groups=fit_groups)
            if initial_neighbours is None:
                initial_neighbours = refreshed.indices.copy()
            prior, uncertainty = refreshed.protein_prior, refreshed.disagreement
            loader = _loader(fit_rna, prior, uncertainty, fit_protein,
                             batch_size=stage.batch_size, shuffle=True)

        corrector.train()
        if stage.finetune_rna_encoder:
            aligner.rna_encoder.train()
        totals = {"total": 0.0, "masked": 0.0, "clean": 0.0, "rate": 0.0}
        batches = 0
        for rna, source, extra, target in loader:
            rna, source = rna.to(device), source.to(device)
            extra, target = extra.to(device), target.to(device)
            latent = corrector_input(aligner, corrector, rna)
            if use_flow:
                width = getattr(corrector, "condition_dim", None)
                condition = extra[:, :1] if width == 1 and extra.shape[1] > 1 else extra
                losses = flow_matching_objective(
                    corrector, source, target, latent, condition,
                    rate, stage.clean_weight)
            else:
                losses = masked_residual_objective(
                    corrector, source, target, latent, extra, rate, scale,
                    stage.clean_weight, stage.pcc_weight)
            optimizer.zero_grad(set_to_none=True)
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, stage.gradient_clip)
            optimizer.step()
            totals["total"] += float(losses.total.detach())
            totals["masked"] += float(losses.masked.detach())
            totals["clean"] += float(losses.clean.detach())
            totals["rate"] += losses.mean_mask_rate
            batches += 1

        row = {"epoch": float(epoch),
               **{f"train_{k}": v / batches for k, v in totals.items()}}

        if epoch % stage.validation_interval == 0:
            memory = build_memory(config)
            memory.fit(encode_rna(aligner, fit_rna, device), fit_protein)
            result = memory.retrieve(encode_rna(aligner, validation_rna, device),
                                     config.retrieval.neighbors)
            width = getattr(corrector, "condition_dim", None)
            extra = result.mean_similarity if width == 1 else result.disagreement
            prediction = predict(corrector, aligner, validation_rna, result.protein_prior,
                                 extra, device, stage.batch_size,
                                 flow_steps=stage.nfe if use_flow else None)
            validation_pcc = protein_pearson_mean(prediction, validation_protein)
            row["validation_pcc_protein"] = validation_pcc
            if validation_pcc > best_pcc:
                best_pcc, best_epoch = validation_pcc, epoch
                best_corrector = deepcopy(corrector.state_dict())
                best_encoder = deepcopy(aligner.rna_encoder.state_dict())
            log(f"correct epoch={epoch:02d}/{stage.epochs} "
                f"loss={row['train_total']:.5f} val_PCC-P={validation_pcc:.4f}")
        history.append(row)

    corrector.load_state_dict(best_corrector)
    aligner.rna_encoder.load_state_dict(best_encoder)

    final = retrieve_training_priors(aligner, fit_rna, fit_protein, config, device,
                                     groups=fit_groups)
    drift = {
        "source_refresh_interval": int(stage.source_refresh_interval),
        "neighbour_jaccard_start_to_end": (
            neighbourhood_jaccard(initial_neighbours, final.indices)
            if initial_neighbours is not None else None),
        "mean_prior_shift_l2": float(np.linalg.norm(
            final.protein_prior - np.asarray(initial_prior, np.float32), axis=1).mean()),
        "mean_consistency": float(final.consistency.mean()),
    }
    return history, {"best_epoch": best_epoch,
                     "best_validation_pcc_protein": float(best_pcc),
                     "corrector_state": best_corrector, "rna_encoder_state": best_encoder,
                     "geometry_drift": drift}


def save_json(path: str | Path, payload: object) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, ensure_ascii=False)


def config_to_dict(config: EpiGradeConfig) -> dict[str, object]:
    return asdict(config)
