from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.baselines import (
    expression_knn_transfer,
    global_mean_predictor,
    seurat_anchor_transfer,
)
from epigrade.config import load_config
from epigrade.data import library_log_normalize, protein_targeted_gene_indices
from epigrade.metrics import evaluate, protein_pearson
from epigrade.aligner import CrossModalAligner
from epigrade.corrector import DisagreementGuidedCorrector
from epigrade.fusion import fused_predictions
from epigrade.panels import canonicalise_panel, collapse_duplicates
from epigrade.consensus import ReciprocalProteinMemory
from run_fused import load_fusion_settings
from epigrade.training import (
    build_memory,
    encode_rna,
    predict,
    retrieve_training_priors,
    save_json,
    set_seed,
    train_alignment,
    train_correction,
)


def harmonise(atlas_path: Path, query_path: Path, protein_key: str):
    atlas = ad.read_h5ad(atlas_path)
    query = ad.read_h5ad(query_path)

    def panel(adata):
        frame = adata.obsm[protein_key]
        names = list(frame.columns) if isinstance(frame, pd.DataFrame) else \
            [f"protein_{i}" for i in range(frame.shape[1])]
        values = frame.to_numpy(np.float32) if isinstance(frame, pd.DataFrame) \
            else np.asarray(frame, np.float32)
        _, groups = canonicalise_panel(names)
        return collapse_duplicates(values, groups)

    atlas_protein, atlas_symbols = panel(atlas)
    query_protein, query_symbols = panel(query)
    shared_proteins = sorted(set(atlas_symbols) & set(query_symbols))
    if len(shared_proteins) < 5:
        raise SystemExit(f"Only {len(shared_proteins)} shared proteins; transfer is not meaningful.")

    shared_genes = sorted(set(map(str, atlas.var_names)) & set(map(str, query.var_names)))
    if len(shared_genes) < 500:
        raise SystemExit(f"Only {len(shared_genes)} shared genes; transfer is not meaningful.")

    atlas_protein = atlas_protein[:, [atlas_symbols.index(p) for p in shared_proteins]]
    query_protein = query_protein[:, [query_symbols.index(p) for p in shared_proteins]]
    atlas = atlas[:, shared_genes].copy()
    query = query[:, shared_genes].copy()

    print(f"harmonised: {len(shared_genes)} genes, {len(shared_proteins)} proteins")
    print(f"  atlas {atlas.n_obs} cells   query {query.n_obs} cells")
    return atlas, query, atlas_protein, query_protein, shared_genes, shared_proteins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--atlas-name", required=True)
    parser.add_argument("--query-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--atlas-subsample", type=int, default=0,
                        help="Cap the atlas at n cells (0 = use all).")
    parser.add_argument("--validation-donors", type=int, default=2,
                        help="Atlas donors held out for checkpoint selection.")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fused.yaml")
    parser.add_argument("--consensus", action="store_true",
                        help="Consensus retrieval: consistency gating and "
                             "spread-rescaled weights (Section 3.3).")
    parser.add_argument("--model", default="fused", choices=("fused", "residual"),
                        help="'residual' reproduces the disagreement-guided corrector.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.consensus:
        from dataclasses import replace
        config = replace(config, retrieval=replace(
            config.retrieval, use_consistency=True, use_spread_weights=True))
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(config.experiment.seed)
    rng = np.random.default_rng(config.experiment.seed)

    atlas, query, atlas_protein, query_protein, genes, proteins = harmonise(
        args.atlas, args.query, config.experiment.protein_obsm_key
    )

    if args.atlas_subsample and atlas.n_obs > args.atlas_subsample:
        keep = rng.choice(atlas.n_obs, args.atlas_subsample, replace=False)
        keep.sort()
        atlas, atlas_protein = atlas[keep].copy(), atlas_protein[keep]
        print(f"  atlas subsampled to {atlas.n_obs} cells")

    target_sum = config.preprocessing.normalization_target_sum
    atlas_rna_norm = library_log_normalize(atlas.X, target_sum)
    query_rna_norm = library_log_normalize(query.X, target_sum)
    atlas_protein_norm = library_log_normalize(atlas_protein, target_sum)
    query_protein_norm = library_log_normalize(query_protein, target_sum)

    donors = atlas.obs["donor"].astype(str).to_numpy() if "donor" in atlas.obs \
        else np.zeros(atlas.n_obs, dtype=str)
    unique_donors = np.unique(donors)
    if unique_donors.size > args.validation_donors >= 1:
        held = set(rng.choice(unique_donors, args.validation_donors, replace=False))
        is_validation = np.isin(donors, list(held))
    else:
        is_validation = np.zeros(atlas.n_obs, dtype=bool)
        is_validation[rng.choice(atlas.n_obs, max(1, atlas.n_obs // 8), replace=False)] = True
    fit_indices = np.flatnonzero(~is_validation)
    validation_indices = np.flatnonzero(is_validation)
    print(f"  atlas fit={fit_indices.size} validation={validation_indices.size} "
          f"({'donor-held-out' if unique_donors.size > 1 else 'random'})")

    selected = protein_targeted_gene_indices(
        atlas_rna_norm[fit_indices], atlas_protein_norm[fit_indices],
        config.preprocessing.genes_per_protein, config.preprocessing.gene_cap,
    )
    to_dense = lambda m: np.asarray(m.todense() if hasattr(m, "todense") else m, np.float32)
    atlas_rna = to_dense(atlas_rna_norm[:, selected])
    query_rna = to_dense(query_rna_norm[:, selected])
    atlas_counts = to_dense(atlas.X[:, selected])
    save_json(args.output / "harmonisation.json", {
        "atlas": args.atlas_name, "query": args.query_name,
        "n_shared_genes": len(genes), "n_shared_proteins": len(proteins),
        "shared_proteins": proteins,
        "n_selected_genes": int(selected.size),
        "atlas_cells": int(atlas.n_obs), "query_cells": int(query.n_obs),
        "atlas_fit": int(fit_indices.size), "atlas_validation": int(validation_indices.size),
        "atlas_donors": sorted(map(str, unique_donors)),
    })

    results: dict[str, np.ndarray] = {}

    started = time.time()
    aligner = CrossModalAligner(
        rna_dim=atlas_rna.shape[1], protein_dim=atlas_protein_norm.shape[1],
        hidden_dim=config.alignment.hidden_dim, latent_dim=config.alignment.latent_dim,
        projection_dim=config.alignment.projection_dim, depth=config.alignment.depth,
        dropout=config.alignment.dropout, temperature_init=config.alignment.temperature_init,
    ).to(device)
    train_alignment(
        aligner, atlas_rna[fit_indices], atlas_counts[fit_indices],
        atlas_protein_norm[fit_indices], atlas_protein[fit_indices], config, device,
    )
    fit_protein = atlas_protein_norm[fit_indices]
    if args.model == "fused":
        memory = build_memory(config)
        memory.fit(encode_rna(aligner, atlas_rna[fit_indices], device), fit_protein)
        validation_prior = memory.retrieve(
            encode_rna(aligner, atlas_rna[validation_indices], device),
            config.retrieval.neighbors).protein_prior
        memory = build_memory(config)
        memory.fit(encode_rna(aligner, atlas_rna, device), atlas_protein_norm)
        retrieval = memory.retrieve(encode_rna(aligner, query_rna, device),
                                    config.retrieval.neighbors)
        fusion = fused_predictions(
            atlas_rna[fit_indices], fit_protein, atlas_rna[validation_indices],
            atlas_protein_norm[validation_indices], query_rna, validation_prior,
            retrieval.protein_prior, load_fusion_settings(args.config), device,
            config.experiment.seed)
        results["retrieval_prior"] = retrieval.protein_prior
        results["EpiGrade"] = fusion.predictions["full"]
        results["ensemble"] = fusion.predictions["ensemble"]
        save_json(args.output / "fusion_weights.json", {
            "protein": proteins, "network": fusion.fusion_weights[:, 0].tolist(),
            "prior": fusion.fusion_weights[:, 1].tolist(),
            "intercept": fusion.fusion_weights[:, 2].tolist()})
    else:
        fit_result = retrieve_training_priors(aligner, atlas_rna[fit_indices], fit_protein,
                                              config, device)
        corrector = DisagreementGuidedCorrector(
            protein_dim=atlas_protein_norm.shape[1], rna_latent_dim=config.alignment.latent_dim,
            hidden_dim=config.correction.hidden_dim, depth=config.correction.depth,
            dropout=config.correction.dropout,
        ).to(device)
        train_correction(
            aligner, corrector, atlas_rna[fit_indices], fit_protein,
            atlas_rna[validation_indices], atlas_protein_norm[validation_indices],
            fit_result.protein_prior, fit_result.disagreement, config, device,
        )
        memory = build_memory(config)
        memory.fit(encode_rna(aligner, atlas_rna, device), atlas_protein_norm)
        retrieval = memory.retrieve(encode_rna(aligner, query_rna, device),
                                    config.retrieval.neighbors)
        results["retrieval_prior"] = retrieval.protein_prior
        results["EpiGrade"] = predict(corrector, aligner, query_rna, retrieval.protein_prior,
                                    retrieval.disagreement, device,
                                    config.correction.batch_size)
    print(f"  EpiGrade done in {(time.time() - started) / 60:.1f} min")

    if not args.skip_baselines:
        cap = 40000
        reference_rows = np.arange(atlas_rna.shape[0]) if atlas_rna.shape[0] <= cap \
            else np.sort(rng.choice(atlas_rna.shape[0], cap, replace=False))
        for name, function in (
            ("global_mean", lambda: global_mean_predictor(atlas_protein_norm, query_rna.shape[0])),
            ("knn_expression", lambda: expression_knn_transfer(
                atlas_rna[reference_rows], atlas_protein_norm[reference_rows], query_rna,
                neighbors=config.retrieval.neighbors, space="expression")),
            ("knn_pca", lambda: expression_knn_transfer(
                atlas_rna[reference_rows], atlas_protein_norm[reference_rows], query_rna,
                neighbors=config.retrieval.neighbors, space="pca")),
            ("seurat_anchor", lambda: seurat_anchor_transfer(
                atlas_rna[reference_rows], atlas_protein_norm[reference_rows], query_rna)),
        ):
            started = time.time()
            try:
                results[name] = function()
                print(f"  {name} done in {(time.time() - started) / 60:.1f} min")
            except Exception as error:
                print(f"  {name} FAILED: {error}")

    summary: dict[str, object] = {}
    per_protein = {"protein": proteins}
    for name, prediction in results.items():
        metrics = evaluate(prediction, query_protein_norm)
        summary[name] = {k: v for k, v in metrics.items()
                         if not isinstance(v, np.ndarray)}
        per_protein[name] = protein_pearson(prediction, query_protein_norm)
        print(f"  {name:<16} PCC-P={metrics['pcc_protein_mean']:.4f} "
              f"PCC-C={metrics['pcc_cell_mean']:.4f} RMSE={metrics['rmse']:.4f}")

    pd.DataFrame(per_protein).to_csv(args.output / "per_protein_pcc.csv", index=False)

    from sklearn.cluster import KMeans

    n_clusters = min(12, max(2, query_protein_norm.shape[0] // 500))
    populations = KMeans(n_clusters=n_clusters, n_init=10,
                         random_state=config.experiment.seed).fit_predict(
                             query_protein_norm)
    confidence = retrieval.mean_similarity.reshape(-1)
    rows = []
    for level in range(n_clusters):
        mask = populations == level
        if mask.sum() < 20:
            continue
        row = {"population": int(level), "n_cells": int(mask.sum()),
               "mean_retrieval_confidence": float(confidence[mask].mean())}
        for name, prediction in results.items():
            row[name] = float(protein_pearson(prediction[mask],
                                              query_protein_norm[mask]).mean())
        rows.append(row)
    if rows:
        frame = pd.DataFrame(rows).sort_values("mean_retrieval_confidence")
        frame.to_csv(args.output / "per_population_pcc.csv", index=False)
        print("  per-population breakdown (sorted by atlas support):")
        print(frame.to_string(index=False))
    np.save(args.output / "populations.npy", populations)

    for key in ("celltype_l1", "celltype_l2", "donor", "condition"):
        if key not in query.obs:
            continue
        labels = query.obs[key].astype(str).to_numpy()
        rows = []
        for level in sorted(set(labels)):
            mask = labels == level
            if mask.sum() < 20:
                continue
            row = {"level": level, "n_cells": int(mask.sum())}
            for name, prediction in results.items():
                row[name] = float(protein_pearson(prediction[mask],
                                                  query_protein_norm[mask]).mean())
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(args.output / f"per_{key}_pcc.csv", index=False)

    np.save(args.output / "confidence.npy", retrieval.mean_similarity)
    np.save(args.output / "disagreement.npy", retrieval.disagreement)
    np.save(args.output / "consistency.npy", retrieval.consistency)
    np.savez_compressed(args.output / "predictions.npz",
                        target=query_protein_norm,
                        protein_names=np.asarray(proteins, dtype=object),
                        **results)
    save_json(args.output / "summary.json", {
        "atlas": args.atlas_name, "query": args.query_name,
        "n_shared_proteins": len(proteins), "n_shared_genes": len(genes),
        "methods": summary,
    })


if __name__ == "__main__":
    main()
