from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.aligner import CrossModalAligner
from epigrade.config import load_config
from epigrade.consensus import ReciprocalProteinMemory
from epigrade.corrector import DisagreementGuidedCorrector, FlowCorrector
from epigrade.data import load_h5ad_dataset, load_or_create_splits, prepare_fold_data
from epigrade.metrics import evaluate, protein_pearson
from epigrade.reliability import ReliabilityEstimator, atlas_features
from epigrade.training import (
    build_memory, config_to_dict, encode_rna, predict, retrieve_training_priors,
    save_json, set_seed, train_alignment, train_correction,
)

VARIANTS = ("full", "prior_only", "with_consistency", "with_spread_weights",
            "with_consensus", "scalar_confidence", "uniform_mask",
            "flow_parameterisation", "flow_with_disagreement", "legacy",
            "donor_aware", "donor_aware_flow", "no_refresh")


def configure(config, variant: str):
    retrieval = config.retrieval
    use_flow = variant in ("flow_parameterisation", "flow_with_disagreement",
                           "legacy", "donor_aware_flow")
    scalar = variant in ("scalar_confidence", "flow_parameterisation", "legacy")
    if variant in ("with_consistency", "with_consensus"):
        retrieval = replace(retrieval, use_consistency=True)
    if variant in ("with_spread_weights", "with_consensus"):
        retrieval = replace(retrieval, use_spread_weights=True)
    scale = 0.0 if variant in ("uniform_mask", "legacy") else \
        config.correction.mask_disagreement_scale
    correction = config.correction
    if variant in ("donor_aware", "donor_aware_flow"):
        correction = replace(correction, training_group_exclusion=True)
    if variant == "no_refresh":
        correction = replace(correction, source_refresh_interval=0)
    if variant == "donor_aware_flow":
        scalar = True
    return (replace(config, retrieval=retrieval, correction=correction),
            use_flow, scalar, scale)


def apply_overrides(config, assignments: list[str]):
    for assignment in assignments:
        target, _, raw = assignment.partition("=")
        section, _, field = target.partition(".")
        if not (section and field and raw):
            raise SystemExit(f"--set expects SECTION.FIELD=VALUE, got {assignment!r}")
        block = getattr(config, section, None)
        if block is None or not hasattr(block, field):
            raise SystemExit(f"--set: no config field {section}.{field}")
        current = getattr(block, field)
        caster = type(current) if current is not None else str
        value = (raw.lower() in ("1", "true", "yes") if isinstance(current, bool)
                 else caster(raw))
        config = replace(config, **{section: replace(block, **{field: value})})
        print(f"override {section}.{field} = {value}", flush=True)
    return config


def match_condition(corrector, extra: np.ndarray) -> np.ndarray:
    width = getattr(corrector, "condition_dim", None)
    return extra[:, :1] if width == 1 and extra.shape[1] > 1 else extra


def uncertainty_channel(result, scalar: bool, n_proteins: int) -> np.ndarray:
    if scalar:
        return np.repeat(result.mean_similarity, n_proteins, axis=1).astype(np.float32)
    return result.disagreement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", default="full", choices=VARIANTS)
    parser.add_argument("--split-mode", default="donor", choices=("cell", "donor"))
    parser.add_argument("--group-key", default="donor")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                        help="Override one config field, for sweeps.")
    parser.add_argument("--tune", action="store_true",
                        help="Score on the validation split and never read the "
                             "test split, so a sweep cannot select on test.")
    parser.add_argument("--lodo", action="store_true",
                        help="Donor-grouped: hold out one donor per fold "
                             "instead of packing donors into a fixed count.")
    parser.add_argument("--folds", default="")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "epigrade.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config, use_flow, scalar, mask_scale = configure(base_config, args.variant)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    dataset = load_h5ad_dataset(args.data, config.experiment.protein_obsm_key)
    n_cells, n_proteins = dataset.protein_counts.shape

    groups = None
    folds = config.experiment.folds
    if args.split_mode == "donor":
        if dataset.obs is None or args.group_key not in dataset.obs:
            raise SystemExit(f"--split-mode donor needs adata.obs['{args.group_key}'].")
        groups = dataset.obs[args.group_key].astype(str).to_numpy()
        n_groups = int(np.unique(groups).size)
        if n_groups < 2:
            raise SystemExit(f"{args.dataset_name} has too few {args.group_key} levels.")
        folds = n_groups if args.lodo else min(config.experiment.folds, n_groups)
        if folds != config.experiment.folds:
            print(f"donor-grouped: {folds} folds, one {args.group_key} held out "
                  f"per fold", flush=True)

    config = apply_overrides(config, args.set)
    splits = load_or_create_splits(
        args.splits, n_cells, folds,
        config.experiment.seed,
        config.experiment.validation_fraction_of_reference, groups=groups)
    save_json(args.output / "splits.json", splits)
    save_json(args.output / "config.json", config_to_dict(config))
    save_json(args.output / "run_settings.json", {
        "dataset": args.dataset_name, "variant": args.variant,
        "split_mode": args.split_mode, "group_key": args.group_key,
        "use_flow": use_flow, "scalar_confidence": scalar,
        "mask_disagreement_scale": mask_scale,
        "training_group_exclusion": config.correction.training_group_exclusion,
        "n_cells": int(n_cells), "n_proteins": int(n_proteins),
        "protein_names": list(dataset.protein_names)})

    requested = (sorted({int(v) for v in args.folds.split(",") if v.strip()})
                 if args.folds else list(range(len(splits))))
    partial = len(requested) < len(splits)
    pooled_prediction = np.empty_like(dataset.protein_counts, dtype=np.float32)
    pooled_target = np.empty_like(dataset.protein_counts, dtype=np.float32)
    pooled_prior = np.empty_like(dataset.protein_counts, dtype=np.float32)
    summaries: list[dict[str, object]] = []

    for fold_id, split in enumerate(splits):
        if fold_id not in requested:
            continue
        started = time.time()
        print(f"\n===== {args.dataset_name} [{args.variant}/{args.split_mode}] "
              f"fold {fold_id + 1}/{len(splits)} =====", flush=True)
        seed = config.experiment.seed + fold_id + args.seed_offset
        set_seed(seed)

        fit = np.asarray(split["fit"], dtype=np.int64)
        validation = np.asarray(split["validation"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)

        prepared = prepare_fold_data(
            dataset, fit, config.preprocessing.normalization_target_sum,
            config.preprocessing.genes_per_protein, config.preprocessing.gene_cap)
        fold_dir = args.output / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        save_json(fold_dir / "selected_genes.json", prepared.selected_genes)

        aligner = CrossModalAligner(
            rna_dim=prepared.rna_normalized.shape[1], protein_dim=n_proteins,
            hidden_dim=config.alignment.hidden_dim, latent_dim=config.alignment.latent_dim,
            projection_dim=config.alignment.projection_dim, depth=config.alignment.depth,
            dropout=config.alignment.dropout,
            temperature_init=config.alignment.temperature_init).to(device)
        train_alignment(aligner, prepared.rna_normalized[fit], prepared.rna_counts[fit],
                        prepared.protein_normalized[fit], prepared.protein_counts[fit],
                        config, device)

        if config.correction.standardize_protein:
            protein_mean = prepared.protein_normalized[fit].mean(axis=0)
            protein_scale = np.maximum(prepared.protein_normalized[fit].std(axis=0), 1e-3)
        else:
            protein_mean = np.zeros(n_proteins, dtype=np.float32)
            protein_scale = np.ones(n_proteins, dtype=np.float32)
        protein_space = ((prepared.protein_normalized - protein_mean)
                         / protein_scale).astype(np.float32)

        fit_protein = protein_space[fit]
        fit_groups = groups[fit] if groups is not None else None
        fit_result = retrieve_training_priors(aligner, prepared.rna_normalized[fit],
                                              fit_protein, config, device,
                                              groups=fit_groups)
        fit_extra = uncertainty_channel(fit_result, scalar, n_proteins)

        checkpoint: dict[str, object] = {}
        corrector = None
        if args.variant != "prior_only":
            if use_flow:
                corrector = FlowCorrector(
                    protein_dim=n_proteins, rna_latent_dim=config.alignment.latent_dim,
                    hidden_dim=config.correction.hidden_dim, depth=config.correction.depth,
                    time_dim=config.correction.time_dim,
                    dropout=config.correction.dropout,
                    condition_dim=1 if scalar else n_proteins).to(device)
            else:
                corrector = DisagreementGuidedCorrector(
                    protein_dim=n_proteins, rna_latent_dim=config.alignment.latent_dim,
                    hidden_dim=config.correction.hidden_dim, depth=config.correction.depth,
                    dropout=config.correction.dropout,
                    rna_skip_dim=(prepared.rna_normalized.shape[1]
                                  if config.correction.rna_skip else 0)).to(device)
                if config.correction.rna_skip:
                    corrector.set_gene_scaling(prepared.rna_normalized[fit])
            _, checkpoint = train_correction(
                aligner, corrector, prepared.rna_normalized[fit], fit_protein,
                prepared.rna_normalized[validation], protein_space[validation],
                fit_result.protein_prior, fit_extra, config, device,
                use_flow=use_flow, mask_disagreement_scale=mask_scale,
                fit_groups=fit_groups)

        reference = np.concatenate([fit, validation])
        reference_protein = prepared.protein_normalized[reference]
        memory = build_memory(config)
        memory.fit(encode_rna(aligner, prepared.rna_normalized[reference], device),
                   protein_space[reference])
        result = memory.retrieve(
            encode_rna(aligner, prepared.rna_normalized[test], device),
            config.retrieval.neighbors)
        test_extra = uncertainty_channel(result, scalar, n_proteins)
        target = prepared.protein_normalized[test]

        if args.variant == "prior_only":
            prediction = result.protein_prior
        else:
            prediction = predict(corrector, aligner, prepared.rna_normalized[test],
                                 result.protein_prior,
                                 match_condition(corrector, test_extra), device,
                                 config.correction.batch_size,
                                 flow_steps=config.correction.nfe if use_flow else None)
        prediction = prediction * protein_scale + protein_mean
        result = replace(result,
                         protein_prior=result.protein_prior * protein_scale + protein_mean,
                         disagreement=result.disagreement * protein_scale)

        reference_types = (dataset.obs["celltype_l1"].astype(str).to_numpy()[reference]
                           if dataset.obs is not None
                           and "celltype_l1" in dataset.obs else None)
        features = atlas_features(prepared.rna_normalized[reference],
                                  reference_protein, cell_types=reference_types,
                                  disagreement=result.disagreement.mean(axis=0))

        if args.tune:
            summaries.append({
                "fold": fold_id,
                "validation_pcc_protein": checkpoint.get(
                    "best_validation_pcc_protein"),
                "best_epoch": checkpoint.get("best_epoch"),
                "n_validation": int(validation.size),
                "minutes": round((time.time() - started) / 60.0, 2),
            })
            save_json(fold_dir / "tuning_record.json", summaries[-1])
            print(f"fold={fold_id} validation PCC-P="
                  f"{summaries[-1]['validation_pcc_protein']}", flush=True)
            continue

        metrics = evaluate(prediction, target)
        prior_metrics = evaluate(result.protein_prior, target)
        np.savez_compressed(
            fold_dir / "outputs.npz", prediction=prediction, target=target,
            retrieval_prior=result.protein_prior, disagreement=result.disagreement,
            consistency=result.consistency, retrieval_similarity=result.mean_similarity,
            retrieval_indices=result.indices, test_indices=test,
            protein_names=np.asarray(prepared.protein_names, dtype=object))
        save_json(fold_dir / "test_metrics.json", metrics)
        save_json(fold_dir / "prior_metrics.json", prior_metrics)
        save_json(fold_dir / "reliability_features.json",
                  {"protein": list(prepared.protein_names),
                   **{k: v.tolist() for k, v in features.items()},
                   "observed_pcc": protein_pearson(prediction, target).tolist()})

        pooled_prediction[test] = prediction
        pooled_target[test] = target
        pooled_prior[test] = result.protein_prior

        summaries.append({
            "fold": fold_id, "seed": seed, "n_fit": int(fit.size),
            "n_validation": int(validation.size), "n_test": int(test.size),
            "selected_gene_count": len(prepared.selected_genes),
            "test_pcc_protein": metrics["pcc_protein_mean"],
            "test_pcc_cell": metrics["pcc_cell_mean"],
            "test_cmd_protein": metrics["cmd_protein"], "test_mmd": metrics["mmd"],
            "test_nw1": metrics["nw1_mean"], "test_rmse": metrics["rmse"],
            "prior_pcc_protein": prior_metrics["pcc_protein_mean"],
            "mean_consistency": float(result.consistency.mean()),
            "mean_disagreement": float(result.disagreement.mean()),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_pcc_protein": checkpoint.get(
                "best_validation_pcc_protein"),
            "geometry_drift": checkpoint.get("geometry_drift"),
            "minutes": round((time.time() - started) / 60.0, 2)})
        save_json(fold_dir / "fold_record.json", summaries[-1])
        if not partial:
            save_json(args.output / "fold_summary.json", summaries)
        print(f"fold={fold_id} PCC-P={metrics['pcc_protein_mean']:.4f} "
              f"prior={prior_metrics['pcc_protein_mean']:.4f} "
              f"({summaries[-1]['minutes']} min)", flush=True)

    if args.tune:
        scores = [s["validation_pcc_protein"] for s in summaries
                  if s["validation_pcc_protein"] is not None]
        save_json(args.output / "tuning_summary.json", {
            "dataset": args.dataset_name, "variant": args.variant,
            "split_mode": args.split_mode, "folds": summaries,
            "fold_mean_validation_pcc_protein":
                float(np.mean(scores)) if scores else None,
            "config": config_to_dict(config)})
        print(f"\n[{args.dataset_name}/{args.split_mode}] TUNE fold-mean "
              f"validation PCC-P="
              f"{np.mean(scores) if scores else float('nan'):.4f}", flush=True)
        return

    if partial:
        print(f"\nfolds {requested} complete; run aggregate_folds.py.", flush=True)
        return

    pooled = evaluate(pooled_prediction, pooled_target)
    pooled_prior_metrics = evaluate(pooled_prior, pooled_target)
    np.savez_compressed(args.output / "pooled_out_of_fold.npz",
                        prediction=pooled_prediction, target=pooled_target,
                        prior=pooled_prior,
                        protein_names=np.asarray(dataset.protein_names, dtype=object))
    save_json(args.output / "pooled_metrics.json", pooled)
    save_json(args.output / "summary.json", {
        "dataset": args.dataset_name, "variant": args.variant,
        "split_mode": args.split_mode, "folds": summaries,
        "fold_mean_pcc_protein": float(np.mean([s["test_pcc_protein"] for s in summaries])),
        "fold_sd_pcc_protein": float(np.std([s["test_pcc_protein"] for s in summaries], ddof=1)),
        "pooled_metrics": pooled, "pooled_prior_metrics": pooled_prior_metrics})
    print(f"\n[{args.dataset_name}/{args.variant}/{args.split_mode}] "
          f"fold-mean PCC-P="
          f"{np.mean([s['test_pcc_protein'] for s in summaries]):.4f}", flush=True)


if __name__ == "__main__":
    main()
