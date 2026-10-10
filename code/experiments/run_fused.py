from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from epigrade.aligner import CrossModalAligner
from epigrade.config import load_config
from epigrade.data import load_h5ad_dataset, load_or_create_splits, prepare_fold_data
from epigrade.fusion import FusionSettings, Standardiser, fused_predictions, train_network
from epigrade.metrics import evaluate, protein_pearson
from epigrade.reliability import atlas_features
from epigrade.training import (
    build_memory, config_to_dict, encode_rna, retrieve_training_priors, save_json,
    set_seed, train_alignment,
)
from aggregate_folds import aggregate
from run_epigrade import apply_overrides

PLAIN_VARIANTS = ("full", "ensemble", "network", "network_fused", "average", "prior_only")
INPUT_VARIANTS = ("prior_input", "prior_input_crossfit")
RETRIEVAL_VARIANTS = ("with_consistency", "with_spread_weights", "with_consensus")


def load_fusion_settings(path: Path) -> FusionSettings:
    raw = yaml.safe_load(path.read_text()).get("fusion", {})
    if "quantiles" in raw:
        raw["quantiles"] = tuple(float(q) for q in raw["quantiles"])
    return FusionSettings(**raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--split-mode", default="donor", choices=("cell", "donor"))
    parser.add_argument("--group-key", default="donor")
    parser.add_argument("--lodo", action="store_true")
    parser.add_argument("--splits", type=Path)
    parser.add_argument("--folds", default="")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--suffix", default="",
                        help="Appended to every variant directory, e.g. _s100.")
    parser.add_argument("--input-ablation", action="store_true")
    parser.add_argument("--retrieval-variant", default="plain",
                        choices=("plain",) + RETRIEVAL_VARIANTS)
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fused.yaml")
    return parser.parse_args()


def new_aligner(rna_dim: int, protein_dim: int, config, device) -> CrossModalAligner:
    a = config.alignment
    return CrossModalAligner(rna_dim=rna_dim, protein_dim=protein_dim,
                             hidden_dim=a.hidden_dim, latent_dim=a.latent_dim,
                             projection_dim=a.projection_dim, depth=a.depth,
                             dropout=a.dropout, temperature_init=a.temperature_init).to(device)


def retrieve(aligner, rna, protein, bank, query, config, device):
    memory = build_memory(config)
    memory.fit(encode_rna(aligner, rna[bank], device), protein[bank])
    return memory.retrieve(encode_rna(aligner, rna[query], device),
                           config.retrieval.neighbors)


def prior_input_prediction(prepared, fit, validation, test, fit_prior, fit_disagreement,
                           val_result, test_result, settings, device, seed):
    rna, protein = prepared.rna_normalized, prepared.protein_normalized
    genes = Standardiser.fit(rna[fit], clip=settings.gene_clip)
    scale = Standardiser.fit(protein[fit])

    def features(cells, prior, disagreement):
        return np.concatenate([genes.transform(rna[cells]), scale.transform(prior),
                               np.log1p(disagreement / scale.scale)], axis=1).astype(np.float32)

    (prediction,), _ = train_network(
        features(fit, fit_prior, fit_disagreement), scale.transform(protein[fit]),
        features(validation, val_result.protein_prior, val_result.disagreement),
        scale.transform(protein[validation]),
        [features(test, test_result.protein_prior, test_result.disagreement)],
        settings, device, seed)
    return scale.inverse(prediction)


def crossfit_priors(prepared, fit, config, device, seed):
    rna, protein = prepared.rna_normalized, prepared.protein_normalized
    half = np.random.default_rng(seed).permutation(fit.size) % 2 == 0
    prior = np.empty((fit.size, protein.shape[1]), np.float32)
    disagreement = np.empty_like(prior)
    for side in (True, False):
        bank, query = fit[half == side], fit[half != side]
        aligner = new_aligner(rna.shape[1], protein.shape[1], config, device)
        train_alignment(aligner, rna[bank], prepared.rna_counts[bank], protein[bank],
                        prepared.protein_counts[bank], config, device, log=lambda s: None)
        result = retrieve(aligner, rna, protein, bank, query, config, device)
        prior[half != side], disagreement[half != side] = \
            result.protein_prior, result.disagreement
    return prior, disagreement


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    settings = load_fusion_settings(args.config)
    if args.retrieval_variant in ("with_consistency", "with_consensus"):
        config = replace(config, retrieval=replace(config.retrieval, use_consistency=True))
    if args.retrieval_variant in ("with_spread_weights", "with_consensus"):
        config = replace(config, retrieval=replace(config.retrieval, use_spread_weights=True))
    device = torch.device(args.device)

    dataset = load_h5ad_dataset(args.data, config.experiment.protein_obsm_key)
    n_cells, n_proteins = dataset.protein_counts.shape
    groups, folds = None, config.experiment.folds
    if args.split_mode == "donor":
        groups = dataset.obs[args.group_key].astype(str).to_numpy()
        n_groups = int(np.unique(groups).size)
        folds = n_groups if args.lodo else min(folds, n_groups)
    config = apply_overrides(config, args.set)
    splits = load_or_create_splits(args.splits, n_cells, folds, config.experiment.seed,
                                   config.experiment.validation_fraction_of_reference,
                                   groups=groups)

    if args.retrieval_variant != "plain":
        variants = (args.retrieval_variant,)
    else:
        variants = PLAIN_VARIANTS + (INPUT_VARIANTS if args.input_ablation else ())
    run_dirs = {}
    for variant in variants:
        run_dir = args.output_root / f"{args.dataset_name}_{args.split_mode}_{variant}{args.suffix}"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_json(run_dir / "splits.json", splits)
        save_json(run_dir / "config.json", {**config_to_dict(config),
                                            "fusion": settings.__dict__})
        save_json(run_dir / "run_settings.json", {
            "dataset": args.dataset_name, "variant": variant + args.suffix,
            "split_mode": args.split_mode, "group_key": args.group_key,
            "seed_offset": args.seed_offset, "retrieval_variant": args.retrieval_variant,
            "n_cells": int(n_cells), "n_proteins": int(n_proteins),
            "protein_names": list(dataset.protein_names)})
        run_dirs[variant] = run_dir

    requested = (sorted({int(v) for v in args.folds.split(",") if v.strip()})
                 if args.folds else list(range(len(splits))))
    for fold_id in requested:
        split = splits[fold_id]
        started = time.time()
        print(f"\n===== {args.dataset_name} [fused/{args.split_mode}] "
              f"fold {fold_id + 1}/{len(splits)} =====", flush=True)
        seed = config.experiment.seed + fold_id + args.seed_offset
        set_seed(seed)
        fit, validation, test = (np.asarray(split[k], dtype=np.int64)
                                 for k in ("fit", "validation", "test"))
        prepared = prepare_fold_data(
            dataset, fit, config.preprocessing.normalization_target_sum,
            config.preprocessing.genes_per_protein, config.preprocessing.gene_cap)
        rna, protein = prepared.rna_normalized, prepared.protein_normalized

        aligner = new_aligner(rna.shape[1], n_proteins, config, device)
        train_alignment(aligner, rna[fit], prepared.rna_counts[fit], protein[fit],
                        prepared.protein_counts[fit], config, device)

        reference = np.concatenate([fit, validation])
        val_result = retrieve(aligner, rna, protein, fit, validation, config, device)
        test_result = retrieve(aligner, rna, protein, reference, test, config, device)

        fusion = fused_predictions(rna[fit], protein[fit], rna[validation],
                                   protein[validation], rna[test],
                                   val_result.protein_prior, test_result.protein_prior,
                                   settings, device, seed)
        predictions = dict(fusion.predictions)
        predictions["prior_only"] = test_result.protein_prior
        if args.retrieval_variant != "plain":
            predictions = {args.retrieval_variant: predictions["full"]}
        if args.input_ablation and args.retrieval_variant == "plain":
            leaky = retrieve_training_priors(aligner, rna[fit], protein[fit], config, device)
            predictions["prior_input"] = prior_input_prediction(
                prepared, fit, validation, test, leaky.protein_prior, leaky.disagreement,
                val_result, test_result, settings, device, seed)
            cf_prior, cf_disagreement = crossfit_priors(prepared, fit, config, device, seed)
            predictions["prior_input_crossfit"] = prior_input_prediction(
                prepared, fit, validation, test, cf_prior, cf_disagreement,
                val_result, test_result, settings, device, seed)

        target = protein[test]
        prior_metrics = evaluate(test_result.protein_prior, target)
        reference_types = (dataset.obs["celltype_l1"].astype(str).to_numpy()[reference]
                           if dataset.obs is not None and "celltype_l1" in dataset.obs
                           else None)
        minutes = round((time.time() - started) / 60.0, 2)
        for variant, run_dir in run_dirs.items():
            prediction = predictions[variant]
            fold_dir = run_dir / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            metrics = evaluate(prediction, target)
            np.savez_compressed(
                fold_dir / "outputs.npz", prediction=prediction, target=target,
                retrieval_prior=test_result.protein_prior,
                disagreement=test_result.disagreement,
                consistency=test_result.consistency,
                retrieval_similarity=test_result.mean_similarity,
                retrieval_indices=test_result.indices, test_indices=test,
                protein_names=np.asarray(prepared.protein_names, dtype=object))
            save_json(fold_dir / "selected_genes.json", prepared.selected_genes)
            save_json(fold_dir / "test_metrics.json", metrics)
            save_json(fold_dir / "prior_metrics.json", prior_metrics)
            if variant in ("full",) + RETRIEVAL_VARIANTS:
                features = atlas_features(rna[reference], protein[reference],
                                          cell_types=reference_types,
                                          disagreement=test_result.disagreement.mean(axis=0))
                save_json(fold_dir / "reliability_features.json",
                          {"protein": list(prepared.protein_names),
                           **{k: v.tolist() for k, v in features.items()},
                           "observed_pcc": protein_pearson(prediction, target).tolist()})
                save_json(fold_dir / "fusion_weights.json", {
                    "protein": list(prepared.protein_names),
                    "network": fusion.fusion_weights[:, 0].tolist(),
                    "prior": fusion.fusion_weights[:, 1].tolist(),
                    "intercept": fusion.fusion_weights[:, 2].tolist()})
            save_json(fold_dir / "fold_record.json", {
                "fold": fold_id, "seed": seed, "n_fit": int(fit.size),
                "n_validation": int(validation.size), "n_test": int(test.size),
                "selected_gene_count": len(prepared.selected_genes),
                "test_pcc_protein": metrics["pcc_protein_mean"],
                "test_pcc_cell": metrics["pcc_cell_mean"],
                "test_cmd_protein": metrics["cmd_protein"], "test_mmd": metrics["mmd"],
                "test_nw1": metrics["nw1_mean"], "test_rmse": metrics["rmse"],
                "prior_pcc_protein": prior_metrics["pcc_protein_mean"],
                "mean_consistency": float(test_result.consistency.mean()),
                "mean_disagreement": float(test_result.disagreement.mean()),
                "best_epoch": fusion.best_epochs, "minutes": minutes})
        line = "  ".join(f"{v}={protein_pearson(predictions[v], target).mean():.4f}"
                         for v in run_dirs)
        print(f"fold={fold_id} {line} ({minutes} min)", flush=True)

    for run_dir in run_dirs.values():
        summary = aggregate(run_dir)
        if summary is not None:
            print(f"[{run_dir.name}] fold-mean PCC-P="
                  f"{summary['fold_mean_pcc_protein']:.4f}", flush=True)


if __name__ == "__main__":
    main()
