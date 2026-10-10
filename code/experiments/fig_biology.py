from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from epigrade.gating import otsu_threshold, quadrant_fractions

import figure_style as fs

GATES = [("CD3", "CD19", "T versus B"),
         ("CD14", "CD16", "monocyte subsets"),
         ("CD152", "CD80", "activation state")]

ANNOTATE = {"CD11B": (5, -3), "CD123": (6, -8), "CD177": (8, -3)}

QUADRANT_NAMES = {0: {"+-": "T", "-+": "B"}}


def load(results: Path, dataset: str, split: str):
    run = results / f"{dataset}_{split}_full"
    path = run / "pooled_out_of_fold.npz"
    if not path.exists():
        fs.skip(f"{dataset}/{split} has no pooled output")
        return None, None, None
    data = np.load(path, allow_pickle=True)
    names = list(json.loads((run / "run_settings.json").read_text())["protein_names"])
    return data["target"], data["prediction"], names


def biaxial(ax, x, y, ramp, limits=None):
    high_x = float(np.quantile(x, 0.999)) if limits is None else limits[0]
    high_y = float(np.quantile(y, 0.999)) if limits is None else limits[1]
    mappable = ax.hexbin(x, y, gridsize=40, cmap=ramp, bins="log", linewidths=0,
                         extent=(0, high_x, 0, high_y), mincnt=1, zorder=3)
    ax.set_xlim(0, high_x)
    ax.set_ylim(0, high_y)
    fs.grid(ax)
    return mappable, (high_x, high_y)


def draw_gate(ax, x, y, threshold_x, threshold_y, column, show_names):
    line = dict(color=fs.INK, linewidth=0.7, linestyle=(0, (3, 1.8)), zorder=5)
    ax.axvline(threshold_x, **line)
    ax.axhline(threshold_y, **line)
    parts = quadrant_fractions(x, y, threshold_x, threshold_y)
    corners = {"++": (0.97, 0.96, "right", "top"),
               "-+": (0.03, 0.96, "left", "top"),
               "+-": (0.97, 0.05, "right", "bottom"),
               "--": (0.03, 0.05, "left", "bottom")}
    for key, (fx, fy, ha, va) in corners.items():
        label = f"{100 * parts[key]:.1f}%"
        if show_names and key in QUADRANT_NAMES.get(column, {}):
            label = f"{QUADRANT_NAMES[column][key]}  {label}"
        ax.text(fx, fy, label, transform=ax.transAxes, ha=ha, va=va,
                fontsize=fs.MIN_FONT, color=fs.INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.72))
    return parts


def panel_gates(axes, target, prediction, names, grades, ramp):
    mappable = None
    for column, (marker_x, marker_y, caption) in enumerate(GATES):
        if marker_x not in names or marker_y not in names:
            fs.skip(f"gate {marker_x}/{marker_y}: marker absent from the panel")
            continue
        ix, iy = names.index(marker_x), names.index(marker_y)
        _, limits = biaxial(axes[0][column], target[:, ix], target[:, iy], ramp)
        mappable, _ = biaxial(axes[1][column], prediction[:, ix],
                              prediction[:, iy], ramp, limits)

        threshold_x = otsu_threshold(target[:, ix])
        threshold_y = otsu_threshold(target[:, iy])
        draw_gate(axes[0][column], target[:, ix], target[:, iy],
                  threshold_x, threshold_y, column, show_names=True)
        draw_gate(axes[1][column], prediction[:, ix], prediction[:, iy],
                  threshold_x, threshold_y, column, show_names=False)

        worst = min(grades.get(marker_x, np.nan), grades.get(marker_y, np.nan))
        axes[0][column].set_title(
            f"{'abc'[column]}  {caption}\ngrade {worst:.2f}",
            fontsize=fs.small() + 0.6, color=fs.INK, pad=3, loc="left")
        for row in (0, 1):
            axes[row][column].set_xlabel(marker_x)
        axes[0][column].set_ylabel(marker_y)
        axes[1][column].set_ylabel(marker_y)
    axes[0][0].text(-0.42, 0.5, "measured", transform=axes[0][0].transAxes,
                    rotation=90, ha="center", va="center", weight="normal",
                    fontsize=fs.small() + 1.0, color=fs.INK)
    axes[1][0].text(-0.42, 0.5, "imputed", transform=axes[1][0].transAxes,
                    rotation=90, ha="center", va="center", weight="normal",
                    fontsize=fs.small() + 1.0, color=fs.INK)
    return mappable


def panel_lineage(figure, ax, table):
    block = table[table.dataset == "GSE3P"]
    if block.empty or "celltype_ratio" not in block:
        fs.skip("lineage panel: no cell-type variance ratio available")
        return
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cool = LinearSegmentedColormap.from_list("cool", fs.COOL)
    dropout = np.clip(1.0 - block.detection.values, 1e-3, 1.0)
    points = ax.scatter(block.celltype_ratio, block.pcc_full, s=15, c=dropout,
                        cmap=cool, norm=LogNorm(vmin=1e-3, vmax=1.0),
                        marker=fs.MARKERS[0], edgecolor="white",
                        linewidth=0.4, zorder=3)
    cax = ax.inset_axes([0.34, 0.28, 0.19, 0.045])
    bar = figure.colorbar(points, cax=cax, orientation="horizontal")
    bar.ax.tick_params(labelsize=fs.MIN_FONT, width=0.4, length=1.8, pad=1.2)
    bar.set_label("fraction of cells with no detection", fontsize=fs.MIN_FONT,
                  color=fs.INK_SECONDARY, labelpad=2)
    bar.outline.set_linewidth(0.5)
    from matplotlib.patches import Ellipse
    import matplotlib.patheffects as path_effects
    halo = [path_effects.withStroke(linewidth=1.8, foreground="white")]
    rings = []
    for column, (marker_x, marker_y, _) in enumerate(GATES):
        pairs = block[block.protein.isin((marker_x, marker_y))]
        if len(pairs) != 2:
            continue
        cx, cy = pairs.celltype_ratio.mean(), pairs.pcc_full.mean()
        width = pairs.celltype_ratio.max() - pairs.celltype_ratio.min() + 0.075
        height = pairs.pcc_full.max() - pairs.pcc_full.min() + 0.105
        rings.append((cx, cy, width, height))
        ax.add_patch(Ellipse((cx, cy), width, height, fill=False,
                             edgecolor=fs.INK, linewidth=0.7,
                             linestyle=(0, (3, 1.8)), zorder=4))
        ax.annotate("abc"[column], xy=(cx + width / 2, cy),
                    xytext=(4, 0), textcoords="offset points",
                    ha="left", va="center", weight="bold",
                    fontsize=fs.small(), color=fs.INK, zorder=5,
                    path_effects=halo)

    for _, row in block[block.protein.isin(ANNOTATE)].iterrows():
        if any(abs(row.celltype_ratio - cx) <= w / 2
               and abs(row.pcc_full - cy) <= h / 2 for cx, cy, w, h in rings):
            fs.skip(f"lineage panel: {row.protein} is labelled inside a ring")
        ax.annotate(row.protein, xy=(row.celltype_ratio, row.pcc_full),
                    xytext=ANNOTATE[row.protein], textcoords="offset points",
                    fontsize=fs.small() - 0.5, color=fs.INK_SECONDARY,
                    zorder=5, path_effects=halo)

    r = np.corrcoef(block.celltype_ratio, block.pcc_full)[0, 1]
    ax.text(0.98, 0.04, f"$r = {r:+.2f}$ over {len(block)} proteins",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fs.small(), color=fs.INK_SECONDARY)
    ax.set_xlabel("Cell-type-associated variation\n"
                  "(between-cell-type variance / total variance)")
    ax.set_ylabel("Observed PCC-P")
    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0, 1.0)
    fs.grid(ax)
    fs.panel_label(ax, "d", "Accuracy and cell-type-associated variation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    plt = fs.apply()
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("density", fs.DENSITY)

    target, prediction, names = load(args.results, "GSE3P", "donor")
    if target is None:
        fs.require_complete("fig_biology")
        return
    table = pd.read_csv(args.results / "protein_reliability.csv")
    grades = dict(zip(table[table.dataset == "GSE3P"].protein,
                      table[table.dataset == "GSE3P"].pcc_full))

    figure = plt.figure(figsize=(7.2, 5.6))
    spec = figure.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.25],
                               hspace=0.55, wspace=0.46,
                               right=0.94)
    axes = [[figure.add_subplot(spec[row, col]) for col in range(3)]
            for row in (0, 1)]
    mappable = panel_gates(axes, target, prediction, names, grades, ramp)
    if mappable is not None:
        cax = axes[1][2].inset_axes([1.10, 0.02, 0.055, 0.96])
        bar = figure.colorbar(mappable, cax=cax)
        bar.ax.tick_params(labelsize=fs.MIN_FONT, width=0.4, length=1.8, pad=1.2)
        bar.set_label("cells per bin", fontsize=fs.MIN_FONT,
                      color=fs.INK_SECONDARY, labelpad=2)
        bar.outline.set_linewidth(0.5)
    panel_lineage(figure, figure.add_subplot(spec[2, :]), table)

    fs.require_complete("fig_biology")
    fs.save(figure, args.out, "fig_biology")
    plt.close(figure)


if __name__ == "__main__":
    main()
