from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.data import library_log_normalize
from epigrade.gating import otsu_threshold, quadrant_fractions
from epigrade.metrics import protein_pearson


def _atlas_features(adata, protein, genes_subset: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    rna = library_log_normalize(adata.X, 10000.0)
    rna = np.asarray(rna.todense() if hasattr(rna, "todense") else rna, dtype=np.float32)
    variance = rna.var(axis=0)
    keep = np.argsort(variance)[::-1][:genes_subset]
    rna = rna[:, keep]

    rows = rng.choice(rna.shape[0], min(20000, rna.shape[0]), replace=False)
    rna, protein_sub = rna[rows], protein[rows]

    rna_c = rna - rna.mean(0)
    prot_c = protein_sub - protein_sub.mean(0)
    rna_n = np.maximum(np.sqrt((rna_c ** 2).sum(0)), 1e-8)
    prot_n = np.maximum(np.sqrt((prot_c ** 2).sum(0)), 1e-8)
    correlations = np.abs((rna_c.T @ prot_c) / (rna_n[:, None] * prot_n[None, :]))

    features = {
        "atlas_sd": protein.std(axis=0),
        "detection": (protein > 0).mean(axis=0),
        "max_gene_corr": correlations.max(axis=0),
        "top_gene_corr": np.sort(correlations, axis=0)[-10:].mean(axis=0),
    }

    if "celltype_l1" in adata.obs:
        labels = adata.obs["celltype_l1"].astype(str).to_numpy()
        overall = protein.var(axis=0) + 1e-8
        between = np.zeros(protein.shape[1])
        grand = protein.mean(axis=0)
        for level in np.unique(labels):
            mask = labels == level
            if mask.sum() < 20:
                continue
            between += mask.mean() * (protein[mask].mean(axis=0) - grand) ** 2
        features["celltype_ratio"] = between / overall
    return features


def build_table(results: Path, processed: Path, dataset: str, split: str,
                h5ad: str) -> pd.DataFrame | None:
    import anndata as ad

    run = results / f"{dataset}_{split}_full"
    if not (run / "pooled_out_of_fold.npz").exists():
        return None
    data = np.load(run / "pooled_out_of_fold.npz", allow_pickle=True)
    names = list(json.loads((run / "run_settings.json").read_text())["protein_names"])

    folds = sorted(p for p in run.glob("fold_*") if p.is_dir())
    blocks = [np.load(p / "outputs.npz", allow_pickle=True)["disagreement"]
              for p in folds if (p / "outputs.npz").exists()]
    disagreement = np.concatenate(blocks).mean(axis=0)

    adata = ad.read_h5ad(processed / h5ad)
    frame = adata.obsm["protein_expression"]
    protein_raw = frame.to_numpy(np.float32) if hasattr(frame, "to_numpy") else np.asarray(frame)
    protein = np.asarray(library_log_normalize(protein_raw, 10000.0), dtype=np.float32)

    features = _atlas_features(adata, protein)
    features["rel_disagreement"] = disagreement / np.maximum(features["atlas_sd"], 1e-8)

    table = pd.DataFrame({"protein": names, "dataset": dataset, **features})
    table["pcc_prior"] = protein_pearson(data["prior"], data["target"])
    table["pcc_full"] = protein_pearson(data["prediction"], data["target"])
    return table


def _gating_macros(results, big):
    run = results / "GSE3P_donor_full"
    path = run / "pooled_out_of_fold.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    names = json.loads((run / "run_settings.json").read_text())["protein_names"]
    target, prediction = data["target"], data["prediction"]
    measured_zero = (target <= 0).mean(0)
    imputed_zero = (prediction <= 1e-6).mean(0)
    bimodal = measured_zero > 0.05
    grade = big.set_index("protein").pcc_full
    ratio = prediction.std(0) / np.maximum(target.std(0), 1e-9)
    graded = np.array([grade.get(n, np.nan) for n in names])
    keep = ~np.isnan(graded)
    out = {
        "numZeroModeProteins": str(int(bimodal.sum())),
        "numZeroModeKept": str(int((imputed_zero[bimodal] > 0.05).sum())),
        "numZeroModeTotal": str(len(names)),
        "numSpreadLowGrade": f"{np.median(ratio[keep][graded[keep] < 0.3]):.2f}",
        "numSpreadHighGrade": f"{np.median(ratio[keep][graded[keep] > 0.8]):.2f}",
    }
    out["numGatingCells"] = f"{len(target):,}"
    for (mx, my), tag in ((("CD3", "CD19"), "Lineage"),
                          (("CD14", "CD16"), "Mono"),
                          (("CD152", "CD80"), "Activation")):
        if mx not in names or my not in names:
            continue
        ix, iy = names.index(mx), names.index(my)
        tx, ty = otsu_threshold(target[:, ix]), otsu_threshold(target[:, iy])
        for source, side in ((target, "Measured"), (prediction, "Imputed")):
            parts = quadrant_fractions(source[:, ix], source[:, iy], tx, ty)
            for key, corner in (("+-", "XPos"), ("-+", "YPos"), ("++", "Both")):
                out[f"numGate{tag}{side}{corner}"] = f"{100 * parts[key]:.1f}\\%"
    ratio_col = big.set_index("protein").celltype_ratio
    shared = [n for n in names if n in ratio_col.index]
    out["numLineageCorr"] = (
        f"{np.corrcoef(ratio_col[shared], grade[shared])[0, 1]:+.2f}")
    for marker, spelled in (("CD3", "CDThree"), ("CD19", "CDNineteen"),
                            ("CD14", "CDFourteen"), ("CD16", "CDSixteen"),
                            ("CD152", "CDOneFiveTwo"), ("CD80", "CDEighty")):
        if marker in grade.index:
            out[f"numGrade{spelled}"] = f"{grade[marker]:.2f}"
    return out


def write_macros(combined, path: Path, results: Path) -> None:
    import numpy as _np
    from scipy.stats import spearmanr as _spearman
    from epigrade.reliability import FEATURES as _FEATURES, ReliabilityEstimator as _Est

    def held_out(block, columns, folds=5, seed=0):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(block))
        out = np.zeros(len(block))
        for part in np.array_split(order, folds):
            train = np.setdiff1d(order, part)
            est = _Est().fit({c: block[c].values[train] for c in columns},
                             block.pcc_full.values[train])
            out[part] = est.predict({c: block[c].values[part] for c in columns})
        return out

    macros = {}
    big = combined[combined.dataset == "GSE3P"].reset_index(drop=True)
    small = combined[combined.dataset == "ECCITE"].reset_index(drop=True)
    cols_big = [c for c in _FEATURES if c in big and big[c].notna().all()]
    cols_small = [c for c in _FEATURES if c in small and small[c].notna().all()]

    macros["numBestProtein"] = f"{big.pcc_full.max():.2f}"
    macros["numWorstProtein"] = f"{big.pcc_full.min():.2f}"

    pred_big = held_out(big, cols_big)
    macros["numReliabilityRhoBig"] = f"$\\rho = {_spearman(pred_big, big.pcc_full).statistic:.2f}$"
    if len(small) > 20:
        pred_small = held_out(small, cols_small)
        macros["numReliabilityRhoSmall"] = \
            f"$\\rho = {_spearman(pred_small, small.pcc_full).statistic:.2f}$"
        shared = [c for c in cols_small if c in cols_big]
        est = _Est().fit({c: small[c].values for c in shared}, small.pcc_full.values)
        transferred = est.predict({c: big[c].values for c in shared})
        macros["numReliabilityTransfer"] = \
            f"$\\rho = {_spearman(transferred, big.pcc_full).statistic:.2f}$"

    order = np.argsort(pred_big)[::-1]
    observed = big.pcc_full.values[order]
    top = observed[: max(int(0.4 * len(observed)), 1)]
    macros["numReportAll"] = f"{observed.mean():.2f}"
    macros["numReportTop"] = f"{top.mean():.2f}"
    macros["numWorstTop"] = f"{top.min():.2f}"
    macros["numSelectionGain"] = f"{top.mean() - observed.mean():.2f} PCC-P"
    macros.update(_gating_macros(results, big))

    path.write_text(
        "% generated by protein_reliability.py -- do not edit\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
        + "\n", encoding="utf-8")
    print("macros written to", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--processed", type=Path, default=ROOT.parent / "data" / "processed")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "results_fused")
    args = parser.parse_args()

    tables = []
    for dataset, split, h5ad in (("GSE3P", "donor", "gse3p_3p.h5ad"),
                                 ("ECCITE", "donor", "eccite_5p.h5ad"),
                                 ("CBMC", "cell", "cbmc.h5ad")):
        table = build_table(args.results, args.processed, dataset, split, h5ad)
        if table is not None:
            tables.append(table)
            print(f"{dataset}: {len(table)} proteins, "
                  f"PCC-P range {table.pcc_full.min():.3f}-{table.pcc_full.max():.3f}")
    if not tables:
        print("no runs available")
        return
    combined = pd.concat(tables, ignore_index=True)
    combined.to_csv(args.out / "protein_reliability.csv", index=False)
    write_macros(combined, ROOT.parent / "paper" / "sections" / "numbers_reliability.tex",
                 args.results)

    predictors = [c for c in ("atlas_sd", "detection", "max_gene_corr",
                              "top_gene_corr", "celltype_ratio", "rel_disagreement")
                  if c in combined]
    print("\nSpearman correlation with per-protein accuracy (label-free features):")
    for dataset, block in combined.groupby("dataset", sort=False):
        line = f"  {dataset:<7}"
        for column in predictors:
            if block[column].notna().sum() > 3:
                rho = block[[column, "pcc_full"]].corr(method="spearman").iloc[0, 1]
                line += f"  {column}={rho:+.2f}"
        print(line)


if __name__ == "__main__":
    main()
