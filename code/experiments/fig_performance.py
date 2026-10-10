from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figure_style as fs
from epigrade.metrics import protein_pearson

METHOD_ROWS = [
    ("EpiGrade", "full", None),
    ("sciPENN", None, "scipenn"),
    ("Retrieved prior", "prior_only", None),
    ("Seurat 5.3.0", None, "seurat_r"),
    ("$k$NN (PCA)", None, "knn_pca"),
    ("$k$NN (expr.)", None, "knn_expression"),
]
COLUMNS = [("CBMC", "cell"), ("ECCITE", "cell"), ("ECCITE", "donor"),
           ("GSE3P", "cell"), ("GSE3P", "donor")]
COLUMN_LABEL = {("CBMC", "cell"): "CBMC\n13 proteins",
                ("ECCITE", "cell"): "PBMC 5'\ncell-level",
                ("ECCITE", "donor"): "PBMC 5'\ndonor-grouped",
                ("GSE3P", "cell"): "PBMC 3'\ncell-level",
                ("GSE3P", "donor"): "PBMC 3'\ndonor-grouped"}


def load_scores(results: Path, baselines: Path):
    model, base = {}, {}
    for path in results.glob("*/summary.json"):
        if path.parent.name.startswith("xc_"):
            continue
        d = json.loads(path.read_text())
        if "variant" in d:
            model[(d["dataset"], d["split_mode"], d["variant"])] = \
                [f["test_pcc_protein"] for f in d["folds"]]
    for path in list(baselines.glob("*_baselines/baseline_summary.json")) + \
            sorted(results.glob("*_external/baseline_summary.json")):
        d = json.loads(path.read_text())
        for name, block in d["methods"].items():
            base[(d["dataset"], d["split_mode"], name)] = \
                [f["test_pcc_protein"] for f in block["folds"]]
    return model, base


def panel_methods(ax, model, base):
    columns = [c for c in COLUMNS if (c[0], c[1], "full") in model]
    positions = np.arange(len(columns), dtype=float)
    width = 0.8 / len(METHOD_ROWS)
    for index, (label, variant, baseline) in enumerate(METHOD_ROWS):
        folds = [model.get((d, s, variant)) if variant else base.get((d, s, baseline))
                 for d, s in columns]
        means = [float(np.mean(v)) if v else np.nan for v in folds]
        offset = (index - (len(METHOD_ROWS) - 1) / 2) * width
        ax.bar(positions + offset, means, width * 0.86, label=label,
               color=fs.SERIES[index], alpha=0.62, zorder=3,
               hatch=fs.HATCHES[index], edgecolor="white", linewidth=0.45)
        for position, values in zip(positions + offset, folds):
            if not values:
                fs.skip(f"panel a: no folds for {label} in one setting")
                continue
            jitter = np.linspace(-width * 0.22, width * 0.22, len(values))
            ax.scatter(position + jitter, values, s=3.2, color=fs.INK,
                       linewidth=0, zorder=5)
    ax.set_xticks(positions)
    ax.set_xticklabels([COLUMN_LABEL[c] for c in columns])
    ax.set_ylabel("PCC-P")
    ax.set_ylim(0, 1.0)
    fs.grid(ax, axis="y")
    ax.legend(loc="lower center", ncol=5, fontsize=fs.small(), columnspacing=1.2,
              bbox_to_anchor=(0.5, -0.42))
    fs.panel_label(ax, "a", "Accuracy by method and protocol (bars: fold mean; "
                            "dots: individual folds)")


def panel_density(axes, results: Path, dataset: str, split: str, n_markers: int = 3):
    from matplotlib.colors import LinearSegmentedColormap

    run = results / f"{dataset}_{split}_full"
    data = np.load(run / "pooled_out_of_fold.npz", allow_pickle=True)
    prediction, target = data["prediction"], data["target"]
    names = list(json.loads((run / "run_settings.json").read_text())["protein_names"])
    ramp = LinearSegmentedColormap.from_list("density", fs.DENSITY)

    per_protein = protein_pearson(prediction, target)
    chosen = np.argsort(per_protein)[::-1][:n_markers]

    mappable = None
    for ax, column in zip(axes, chosen):
        x, y = target[:, column], prediction[:, column]
        high = float(max(np.quantile(x, 0.999), np.quantile(y, 0.999)))
        mappable = ax.hexbin(x, y, gridsize=34, cmap=ramp, bins="log",
                             linewidths=0, extent=(0, high, 0, high), mincnt=1,
                             zorder=3)
        ax.plot([0, high], [0, high], color=fs.INK_MUTED, linewidth=0.7,
                linestyle=(0, (4, 3)), zorder=4)
        ax.text(0.05, 0.95, f"{names[column]}\n$r={per_protein[column]:.2f}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=fs.small(),
                color=fs.INK_SECONDARY)
        ax.set_xlim(0, high); ax.set_ylim(0, high)
        ax.set_xlabel("Measured")
        fs.grid(ax)
    axes[0].set_ylabel("Predicted")
    return mappable


def panel_prior_vs_corrected(ax, results: Path, pairs):
    for index, (dataset, split, label) in enumerate(pairs):
        run = results / f"{dataset}_{split}_full"
        if not (run / "pooled_out_of_fold.npz").exists():
            fs.skip(f"panel c: {dataset}/{split} has no pooled output")
            continue
        data = np.load(run / "pooled_out_of_fold.npz", allow_pickle=True)
        prior = protein_pearson(data["prior"], data["target"])
        corrected = protein_pearson(data["prediction"], data["target"])
        ax.scatter(prior, corrected, s=14, facecolor=fs.SERIES[index],
                   marker=fs.MARKERS[index], edgecolor="white", linewidth=0.5,
                   zorder=3, label=label)
    low, high = 0.0, 1.0
    ax.plot([low, high], [low, high], color=fs.INK_MUTED, linewidth=0.7,
            linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel("PCC-P, retrieved prior")
    ax.set_ylabel("PCC-P, fused model")
    fs.grid(ax)
    ax.legend(loc="lower right", fontsize=fs.small())
    fs.panel_label(ax, "c", "Prior vs fused model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--baselines", type=Path, default=ROOT.parent / "results")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    plt = fs.apply()

    model, base = load_scores(args.results, args.baselines)
    figure = plt.figure(figsize=(7.2, 4.6))
    spec = figure.add_gridspec(2, 4, height_ratios=[1.0, 0.95],
                               hspace=0.75, wspace=0.44)

    panel_methods(figure.add_subplot(spec[0, :]), model, base)

    density_axes = [figure.add_subplot(spec[1, i]) for i in range(3)]
    mappable = panel_density(density_axes, args.results, "GSE3P", "donor")
    if mappable is not None:
        fs.colorbar(figure, mappable, density_axes[-1], "cells per bin")
    fs.panel_label(density_axes[0], "b", "Best-predicted markers, PBMC 3'")

    panel_prior_vs_corrected(
        figure.add_subplot(spec[1, 3]), args.results,
        [("GSE3P", "donor", "PBMC 3'"), ("ECCITE", "donor", "PBMC 5'"),
         ("CBMC", "cell", "CBMC")])

    fs.require_complete("fig_performance")
    fs.save(figure, args.out, "fig_performance")
    plt.close(figure)


if __name__ == "__main__":
    main()
