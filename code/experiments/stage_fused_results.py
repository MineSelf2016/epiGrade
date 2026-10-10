from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = ("CBMC_cell", "ECCITE_cell", "ECCITE_donor", "GSE3P_cell", "GSE3P_donor")


def link(source: Path, target: Path) -> None:
    if target.is_symlink() or target.exists():
        return
    target.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def stage_external(source: Path, target: Path, knn: Path) -> None:
    if not knn.exists():
        link(source, target)
        return
    if target.is_symlink():
        target.unlink()
    target.mkdir(exist_ok=True)
    block = json.loads((source / "baseline_summary.json").read_text())
    rerun = json.loads(knn.read_text())["methods"]
    for name in ("knn_expression", "knn_pca"):
        block["methods"][name] = rerun[name]
    block["knn_neighbors"] = 30
    (target / "baseline_summary.json").write_text(json.dumps(block, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=ROOT / "results_epigrade")
    parser.add_argument("--results", type=Path, default=ROOT / "results_fused")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    for setting in SETTINGS:
        external = args.legacy / f"{setting}_external"
        if external.exists():
            stage_external(external, args.results / f"{setting}_external",
                           args.results / "knn30" / setting / "baseline_summary.json")

        source = args.legacy / f"{setting}_full"
        if not (source / "summary.json").exists():
            continue
        target = args.results / f"{setting}_residual_corrector"
        target.mkdir(exist_ok=True)
        summary = json.loads((source / "summary.json").read_text())
        summary["variant"] = "residual_corrector"
        (target / "summary.json").write_text(json.dumps(summary, indent=2))
        for item in ("splits.json", "config.json", "pooled_out_of_fold.npz"):
            if (source / item).exists():
                link(source / item, target / item)
        print(f"staged {target.name}")

    for setting in ("ECCITE_donor", "GSE3P_donor"):
        external = args.legacy / "donor5" / f"{setting}_external"
        if external.exists():
            (args.results / "donor5").mkdir(exist_ok=True)
            link(external, args.results / "donor5" / f"{setting}_external")


if __name__ == "__main__":
    main()
