from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.data import library_log_normalize
from epigrade.training import save_json


def evaluate_features(features: np.ndarray, labels: np.ndarray, groups: np.ndarray | None,
                      folds: int = 5, seed: int = 42) -> dict[str, float]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels)
    if groups is not None and np.unique(groups).size >= folds:
        splitter = GroupKFold(n_splits=folds).split(features, labels, groups=groups)
        protocol = "donor-grouped"
    else:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True,
                                   random_state=seed).split(features, labels)
        protocol = "stratified"

    accuracies, macro_f1 = [], []
    for train_index, test_index in splitter:
        scaler = StandardScaler().fit(features[train_index])
        model = LogisticRegression(max_iter=2000, multi_class="multinomial",
                                  C=1.0, random_state=seed)
        model.fit(scaler.transform(features[train_index]), labels[train_index])
        predicted = model.predict(scaler.transform(features[test_index]))
        accuracies.append(accuracy_score(labels[test_index], predicted))
        macro_f1.append(f1_score(labels[test_index], predicted, average="macro"))
    return {"accuracy": float(np.mean(accuracies)), "accuracy_sd": float(np.std(accuracies, ddof=1)),
            "macro_f1": float(np.mean(macro_f1)), "macro_f1_sd": float(np.std(macro_f1, ddof=1)),
            "protocol": protocol, "n_features": int(features.shape[1])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path,
                        help="Completed run directory with pooled_out_of_fold.npz.")
    parser.add_argument("--baselines", type=Path,
                        help="Optional matching *_baselines directory.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--label-key", default="celltype_l1")
    parser.add_argument("--min-cells", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pooled = np.load(args.run / "pooled_out_of_fold.npz", allow_pickle=True)
    adata = ad.read_h5ad(args.data)
    if args.label_key not in adata.obs:
        raise SystemExit(f"{args.data} has no adata.obs['{args.label_key}']")
    labels = adata.obs[args.label_key].astype(str).to_numpy()
    groups = adata.obs["donor"].astype(str).to_numpy() if "donor" in adata.obs else None

    counts = pd.Series(labels).value_counts()
    keep_labels = set(counts[counts >= args.min_cells].index)
    mask = np.isin(labels, list(keep_labels))
    print(f"labels: {len(keep_labels)} populations with >= {args.min_cells} cells "
          f"({mask.sum()}/{mask.size} cells)")

    measured = np.asarray(pooled["target"], dtype=np.float32)
    feature_sets = {
        "measured_protein": measured,
        "EpiGrade": np.asarray(pooled["prediction"], dtype=np.float32),
        "retrieval_prior": np.asarray(pooled["prior"], dtype=np.float32),
    }

    from sklearn.decomposition import PCA
    rna = library_log_normalize(adata.X, 10000.0)
    rna = np.asarray(rna.todense() if hasattr(rna, "todense") else rna, dtype=np.float32)
    variance = rna.var(axis=0)
    top = np.argsort(variance)[::-1][:2000]
    components = min(measured.shape[1], 50, top.size)
    feature_sets["rna_only"] = PCA(n_components=components, random_state=42).fit_transform(rna[:, top])

    if args.baselines and (args.baselines / "baseline_summary.json").exists():
        summary = json.loads((args.baselines / "baseline_summary.json").read_text())
        best = max(summary["methods"].items(),
                   key=lambda kv: kv[1]["fold_mean_pcc_protein"])[0]
        print(f"strongest reference baseline: {best} "
              f"(predictions not stored per fold; reported by correlation only)")

    results = {}
    for name, features in feature_sets.items():
        results[name] = evaluate_features(features[mask], labels[mask],
                                          groups[mask] if groups is not None else None)
        row = results[name]
        print(f"  {name:<18} acc={row['accuracy']:.4f} macro-F1={row['macro_f1']:.4f} "
              f"({row['protocol']}, {row['n_features']} features)")

    save_json(args.output / "annotation_summary.json", {
        "data": str(args.data), "run": str(args.run),
        "label_key": args.label_key, "min_cells": args.min_cells,
        "n_populations": len(keep_labels), "n_cells": int(mask.sum()),
        "results": results,
    })


if __name__ == "__main__":
    main()
