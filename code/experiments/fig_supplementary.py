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


def _run(results: Path, dataset: str, split: str, variant: str = "full"):
    run = results / f"{dataset}_{split}_{variant}"
    path = run / "pooled_out_of_fold.npz"
    if not path.exists():
        return None, None
    names = list(json.loads((run / "run_settings.json").read_text())["protein_names"])
    return np.load(path, allow_pickle=True), names


def figure_distributions(results: Path, out: Path, plt) -> None:
    data, names = _run(results, "GSE3P", "donor")
    if data is None:
        raise SystemExit("supp_fig_distributions: the GSE3P donor run is missing")
    per_protein = protein_pearson(data["prediction"], data["target"])
    order = np.argsort(per_protein)[::-1]
    chosen = [order[0], order[len(order) // 4], order[len(order) // 2],
              order[3 * len(order) // 4], order[-1]]

    figure, axes = plt.subplots(1, len(chosen), figsize=(7.2, 1.9), sharey=False)
    series = [("Measured", data["target"], fs.INK_SECONDARY),
              ("Retrieved prior", data["prior"], fs.SERIES[1]),
              ("Fused model", data["prediction"], fs.SERIES[0])]
    for ax, column in zip(axes, chosen):
        high = float(np.quantile(data["target"][:, column], 0.995))
        bins = np.linspace(0, max(high, 1e-3), 60)
        for order, (label, values, colour) in enumerate(series):
            ax.hist(values[:, column], bins=bins, density=True, histtype="step",
                    linewidth=1.4, color=colour, label=label, zorder=3,
                    linestyle=fs.LINESTYLES[order % len(fs.LINESTYLES)])
        ax.set_title(f"{names[column]}\n$r={per_protein[column]:.2f}$",
                     fontsize=6.2, color=fs.INK, loc="center")
        ax.set_yticks([])
        ax.set_xlabel("Abundance")
        fs.grid(ax, axis="x")
    axes[0].set_ylabel("Density")
    axes[-1].legend(loc="upper right", fontsize=fs.small())
    figure.subplots_adjust(wspace=0.28)
    fs.require_complete("supp_fig_distributions")
    fs.save(figure, out, "supp_fig_distributions")
    plt.close(figure)


def figure_splits(results: Path, out: Path, plt) -> None:
    pairs = [("ECCITE", "PBMC 5$'$"), ("GSE3P", "PBMC 3$'$")]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))

    for ax, (dataset, label) in zip(axes, pairs):
        rows = []
        for split, protocol in (("cell", "cell-level"), ("donor", "donor-grouped")):
            path = results / f"{dataset}_{split}_full" / "summary.json"
            if not path.exists():
                fs.skip(f"splits: {dataset}/{split} summary missing")
                continue
            d = json.loads(path.read_text())
            rows.append((protocol, [f["test_pcc_protein"] for f in d["folds"]]))
        if not rows:
            fs.skip(f"splits: no protocols found for {dataset}")
            ax.axis("off")
            continue
        everything = [v for _, values in rows for v in values]
        span = max(everything) - min(everything)
        ax.set_ylim(min(everything) - 0.10 * span, max(everything) + 0.22 * span)
        for index, (protocol, values) in enumerate(rows):
            x = np.full(len(values), index, dtype=float)
            x += np.linspace(-0.07, 0.07, len(values))
            ax.scatter(x, values, s=24, facecolor=fs.SERIES[index],
                       marker=fs.MARKERS[index], edgecolor="white",
                       linewidth=0.6, zorder=4)
            ax.plot([index - 0.18, index + 0.18], [np.mean(values)] * 2,
                    color=fs.INK, linewidth=1.4, zorder=5)
            ax.text(index, max(everything) + 0.14 * span,
                    f"SD {np.std(values, ddof=1):.4f}", ha="center", va="center",
                    fontsize=fs.small(), color=fs.INK_SECONDARY)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([p for p, _ in rows])
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_ylabel("PCC-P per fold")
        ax.set_xlabel("axis does not start at zero", fontsize=fs.small(),
                      color=fs.INK_MUTED, labelpad=1)
        ax.set_title(label, fontsize=7.0, color=fs.INK, pad=9)
        fs.grid(ax, axis="y")
    figure.subplots_adjust(wspace=0.3)
    fs.require_complete("supp_fig_splits")
    fs.save(figure, out, "supp_fig_splits")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    plt = fs.apply()
    figure_distributions(args.results, args.out, plt)
    figure_splits(args.results, args.out, plt)


if __name__ == "__main__":
    main()
