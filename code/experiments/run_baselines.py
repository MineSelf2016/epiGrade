from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.baselines import (
    expression_knn_transfer,
    global_mean_predictor,
    seurat_anchor_transfer,
)
from epigrade.config import load_config
from epigrade.external import ExternalMethodError, available, run_external
from epigrade.data import load_h5ad_dataset, prepare_fold_data
from epigrade.metrics import evaluate
from epigrade.training import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--reference-run", required=True, type=Path,
                        help="Completed run directory whose splits.json defines the folds.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--anchor-cap", type=int, default=40000)
    parser.add_argument("--external", nargs="*", default=[],
                        choices=["scipenn", "totalvi", "seurat_r"],
                        help="Published methods to run in their own environments.")
    parser.add_argument("--external-seed", type=int, default=0,
                        help="Seed handed to the external methods, for "
                             "replicates that an ensemble can average.")
    parser.add_argument("--external-cap", type=int, default=40000,
                        help="Reference cells given to an external method.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "epigrade.yaml")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Restrict the in-environment baselines to these names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    splits = json.loads((args.reference_run / "splits.json").read_text())
    settings = json.loads((args.reference_run / "run_settings.json").read_text())
    dataset = load_h5ad_dataset(args.data, config.experiment.protein_obsm_key)
    rng = np.random.default_rng(config.experiment.seed)

    methods = {
        "global_mean": lambda ref_rna, ref_prot, qry_rna: global_mean_predictor(
            ref_prot, qry_rna.shape[0]),
        "knn_expression": lambda ref_rna, ref_prot, qry_rna: expression_knn_transfer(
            ref_rna, ref_prot, qry_rna, neighbors=config.retrieval.neighbors,
            space="expression"),
        "knn_pca": lambda ref_rna, ref_prot, qry_rna: expression_knn_transfer(
            ref_rna, ref_prot, qry_rna, neighbors=config.retrieval.neighbors,
            space="pca"),
        "seurat_anchor": seurat_anchor_transfer,
    }
    if args.only is not None:
        methods = {k: v for k, v in methods.items() if k in args.only}
    requested = [name for name in args.external if available(name)]
    for name in args.external:
        if name not in requested:
            print(f"{name}: environment not built, skipping", flush=True)
    per_fold: dict[str, list[dict]] = {
        name: [] for name in list(methods) + requested}

    for fold_id, split in enumerate(splits):
        fit = np.asarray(split["fit"], dtype=np.int64)
        validation = np.asarray(split["validation"], dtype=np.int64)
        test = np.asarray(split["test"], dtype=np.int64)
        reference = np.concatenate([fit, validation])

        prepared = prepare_fold_data(
            dataset, fit, config.preprocessing.normalization_target_sum,
            config.preprocessing.genes_per_protein, config.preprocessing.gene_cap)
        reference_rna = prepared.rna_normalized[reference]
        reference_protein = prepared.protein_normalized[reference]
        query_rna = prepared.rna_normalized[test]
        target = prepared.protein_normalized[test]

        capped = np.arange(reference.size)
        if reference.size > args.anchor_cap:
            capped = np.sort(rng.choice(reference.size, args.anchor_cap, replace=False))
            print(f"  fold {fold_id}: anchor reference capped at {args.anchor_cap}")

        for name, function in methods.items():
            started = time.time()
            use_full = name in ("global_mean", "knn_expression", "knn_pca")
            ref_rna = reference_rna if use_full else reference_rna[capped]
            ref_prot = reference_protein if use_full else reference_protein[capped]
            try:
                prediction = function(ref_rna, ref_prot, query_rna)
            except Exception as error:
                print(f"  fold {fold_id} {name} FAILED: {error}", flush=True)
                continue
            metrics = evaluate(prediction, target)
            per_fold[name].append({
                "fold": fold_id,
                "test_pcc_protein": metrics["pcc_protein_mean"],
                "test_pcc_cell": metrics["pcc_cell_mean"],
                "test_cmd_protein": metrics["cmd_protein"],
                "test_mmd": metrics["mmd"],
                "test_nw1": metrics["nw1_mean"],
                "test_rmse": metrics["rmse"],
                "minutes": round((time.time() - started) / 60.0, 2),
            })
            print(f"  fold {fold_id} {name:<16} PCC-P={metrics['pcc_protein_mean']:.4f} "
                  f"({per_fold[name][-1]['minutes']} min)", flush=True)

        for name in requested:
            started = time.time()
            subset = capped if reference.size > args.external_cap else \
                np.arange(reference.size)
            if subset.size > args.external_cap:
                subset = np.sort(rng.choice(subset, args.external_cap,
                                            replace=False))
            diagnostics: dict = {}
            try:
                prediction = run_external(
                    name,
                    prepared.rna_counts[reference][subset],
                    prepared.protein_counts[reference][subset],
                    prepared.rna_counts[test], diagnostics=diagnostics,
                    extra={"seed": args.external_seed})
            except (ExternalMethodError, Exception) as error:
                print(f"  fold {fold_id} {name} FAILED: {error}", flush=True)
                continue
            metrics = evaluate(prediction, target)
            fold_dir = args.output / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(fold_dir / f"{name}.npz",
                                prediction=prediction.astype(np.float32),
                                target=target.astype(np.float32))
            per_fold[name].append({
                "fold": fold_id,
                "test_pcc_protein": metrics["pcc_protein_mean"],
                "test_pcc_cell": metrics["pcc_cell_mean"],
                "test_cmd_protein": metrics["cmd_protein"],
                "test_mmd": metrics["mmd"],
                "test_nw1": metrics["nw1_mean"],
                "test_rmse": metrics["rmse"],
                "own_units": True,
                **diagnostics,
                "n_reference": int(subset.size),
                "minutes": round((time.time() - started) / 60.0, 2),
            })
            print(f"  fold {fold_id} {name:<16} PCC-P={metrics['pcc_protein_mean']:.4f} "
                  f"({per_fold[name][-1]['minutes']} min)", flush=True)

    summary = {}
    for name, records in per_fold.items():
        if not records:
            continue
        values = [r["test_pcc_protein"] for r in records]
        summary[name] = {
            "folds": records,
            "fold_mean_pcc_protein": float(np.mean(values)),
            "fold_sd_pcc_protein": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "fold_mean_pcc_cell": float(np.mean([r["test_pcc_cell"] for r in records])),
            "fold_mean_cmd_protein": float(np.mean([r["test_cmd_protein"] for r in records])),
            "fold_mean_nw1": float(np.mean([r["test_nw1"] for r in records])),
            "fold_mean_rmse": float(np.mean([r["test_rmse"] for r in records])),
        }
    save_json(args.output / "baseline_summary.json", {
        "dataset": args.dataset_name,
        "split_mode": settings["split_mode"],
        "n_proteins": settings["n_proteins"],
        "methods": summary,
    })
    for name, block in summary.items():
        print(f"{name:<16} PCC-P={block['fold_mean_pcc_protein']:.4f} "
              f"+/- {block['fold_sd_pcc_protein']:.4f}")


if __name__ == "__main__":
    main()
