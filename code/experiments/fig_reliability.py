from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figure_style as fs
from epigrade.reliability import FEATURES, ReliabilityEstimator

PANELS = [("GSE3P", "PBMC 3$'$, 211 proteins"), ("ECCITE", "PBMC 5$'$, 48 proteins")]


def held_out_prediction(block: pd.DataFrame, columns, folds: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(block))
    predicted = np.zeros(len(block))
    for part in np.array_split(order, folds):
        train = np.setdiff1d(order, part)
        estimator = ReliabilityEstimator().fit(
            {c: block[c].values[train] for c in columns},
            block.pcc_full.values[train])
        predicted[part] = estimator.predict({c: block[c].values[part] for c in columns})
    return predicted


def panel_spread(ax, table):
    for index, (dataset, label) in enumerate(PANELS):
        block = table[table.dataset == dataset].sort_values("pcc_full", ascending=False)
        x = np.linspace(0, 1, len(block))
        ax.plot(x, block.pcc_full.values, linewidth=2.0, color=fs.SERIES[index],
                linestyle=fs.LINESTYLES[index % len(fs.LINESTYLES)],
                zorder=3, label=label)
    ax.axhline(0.0, color=fs.INK_MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=2)
    ax.set_xlabel("Protein, ranked within panel")
    ax.set_ylabel("PCC-P")
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["best", "median", "worst"])
    fs.grid(ax)
    ax.legend(loc="upper right", fontsize=fs.small())
    fs.panel_label(ax, "a", "Per-protein accuracy")


def panel_predictable(ax, table):
    for index, (dataset, label) in enumerate(PANELS):
        block = table[table.dataset == dataset].reset_index(drop=True)
        columns = [c for c in FEATURES if c in block and block[c].notna().all()]
        if len(block) < 20 or not columns:
            fs.skip(f"panel b: {dataset} has too few proteins or no usable features")
            continue
        predicted = held_out_prediction(block, columns)
        rho = spearmanr(predicted, block.pcc_full).statistic
        ax.scatter(predicted, block.pcc_full, s=12, facecolor=fs.SERIES[index],
                   marker=fs.MARKERS[index], edgecolor="white", linewidth=0.4,
                   zorder=3,
                   label=f"{label.split(',')[0]} ($\\rho={rho:.2f}$)")
    ax.plot([0, 1], [0, 1], color=fs.INK_MUTED, linewidth=0.7,
            linestyle=(0, (4, 3)), zorder=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted reliability (held-out protein)")
    ax.set_ylabel("Observed PCC-P")
    fs.grid(ax)
    ax.legend(loc="upper left", fontsize=fs.small())
    fs.panel_label(ax, "b", "Reliability estimation")


def panel_selection(ax, table):
    dataset = "GSE3P"
    block = table[table.dataset == dataset].reset_index(drop=True)
    columns = [c for c in FEATURES if c in block and block[c].notna().all()]
    predicted = held_out_prediction(block, columns)
    order = np.argsort(predicted)[::-1]
    observed = block.pcc_full.values[order]

    kept = np.arange(1, len(observed) + 1)
    mean_curve = np.cumsum(observed) / kept
    worst_curve = np.minimum.accumulate(observed)
    fraction = kept / len(observed)

    ax.plot(fraction, mean_curve, linewidth=2.0, color=fs.SERIES[0],
            linestyle=fs.LINESTYLES[0], zorder=4, label="mean of reported")
    ax.plot(fraction, worst_curve, linewidth=2.0, color=fs.SERIES[1],
            linestyle=fs.LINESTYLES[1], zorder=4, label="worst reported")
    ax.axhline(observed.mean(), color=fs.INK_MUTED, linewidth=0.8,
               linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.03, observed.mean() + 0.02, f"report all: {observed.mean():.2f}",
            fontsize=fs.small(), color=fs.INK_SECONDARY, ha="left", va="bottom")
    for mark in (0.4,):
        index = max(int(mark * len(observed)) - 1, 0)
        ax.scatter([fraction[index]], [mean_curve[index]], s=26, zorder=5,
                   marker="o", facecolor=fs.SERIES[0], edgecolor="white",
                   linewidth=0.8)
        ax.annotate(f"top {int(mark*100)}%: {mean_curve[index]:.2f}",
                    xy=(fraction[index], mean_curve[index]),
                    xytext=(fraction[index] + 0.08, mean_curve[index] + 0.10),
                    fontsize=fs.small(), color=fs.INK_SECONDARY,
                    arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                    color=fs.INK_MUTED))
    ax.set_xlabel("Fraction of the panel reported")
    ax.set_ylabel("PCC-P")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fs.grid(ax)
    ax.legend(loc="lower left", fontsize=fs.small())
    fs.panel_label(ax, "c", "Accuracy after selection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path,
                        default=ROOT.parent / "results_fused" / "protein_reliability.csv")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    if not args.table.exists():
        print("run protein_reliability.py first")
        return
    plt = fs.apply()
    table = pd.read_csv(args.table)

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    panel_spread(axes[0], table)
    panel_predictable(axes[1], table)
    panel_selection(axes[2], table)
    figure.subplots_adjust(wspace=0.42)
    fs.require_complete("fig_reliability")
    fs.save(figure, args.out, "fig_reliability")
    plt.close(figure)


if __name__ == "__main__":
    main()
