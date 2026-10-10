from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.metrics import evaluate
from epigrade.training import save_json


def aggregate(run_dir: Path) -> dict | None:
    records, predictions, targets, priors, indices = [], [], [], [], []
    fold_dirs = [p for p in run_dir.glob("fold_*")
                 if p.is_dir() and p.name.split("_")[-1].isdigit()]
    for fold_dir in sorted(fold_dirs, key=lambda p: int(p.name.split("_")[-1])):
        record_path, output_path = fold_dir / "fold_record.json", fold_dir / "outputs.npz"
        if not (record_path.exists() and output_path.exists()):
            continue
        records.append(json.loads(record_path.read_text()))
        data = np.load(output_path, allow_pickle=True)
        predictions.append(data["prediction"])
        targets.append(data["target"])
        priors.append(data["retrieval_prior"])
        indices.append(data["test_indices"])
    if not records:
        return None

    settings = json.loads((run_dir / "run_settings.json").read_text())
    splits = json.loads((run_dir / "splits.json").read_text())
    if len(records) < len(splits):
        print(f"  {run_dir.name}: {len(records)}/{len(splits)} folds present, skipping")
        return None

    order = np.argsort(np.concatenate(indices))
    prediction = np.concatenate(predictions)[order]
    target = np.concatenate(targets)[order]
    prior = np.concatenate(priors)[order]

    pooled = evaluate(prediction, target)
    pooled_prior = evaluate(prior, target)
    np.savez_compressed(run_dir / "pooled_out_of_fold.npz",
                        prediction=prediction, target=target, prior=prior)
    summary = {
        "dataset": settings["dataset"], "variant": settings["variant"],
        "split_mode": settings["split_mode"], "nfe": settings.get("nfe", 1),
        "folds": records,
        "fold_mean_pcc_protein": float(np.mean([r["test_pcc_protein"] for r in records])),
        "fold_sd_pcc_protein": float(np.std([r["test_pcc_protein"] for r in records], ddof=1)),
        "pooled_metrics": pooled, "pooled_prior_metrics": pooled_prior,
    }
    save_json(run_dir / "summary.json", summary)
    print(f"  {run_dir.name}: {len(records)} folds, "
          f"PCC-P={summary['fold_mean_pcc_protein']:.4f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    args = parser.parse_args()
    for run_dir in sorted(args.results.glob("*/")):
        if run_dir.name.startswith("xc_") or (run_dir / "summary.json").exists():
            continue
        aggregate(run_dir)


if __name__ == "__main__":
    main()
