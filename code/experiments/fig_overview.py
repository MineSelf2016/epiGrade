from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figure_style as fs

BOX = dict(boxstyle="round,pad=0.42", linewidth=0.8)


def box(ax, x, y, text, facecolor, edgecolor, width=None, fontsize=6.2, weight="normal"):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, weight=weight,
            color=fs.INK, zorder=4,
            bbox={**BOX, "facecolor": facecolor, "edgecolor": edgecolor})


def arrow(ax, x0, y0, x1, y1, style="-|>", colour=None, linestyle="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=2,
                arrowprops=dict(arrowstyle=style, linewidth=0.9, linestyle=linestyle,
                                color=colour or fs.INK_SECONDARY,
                                shrinkA=2, shrinkB=2))


def stage_header(ax, x, y, number, title):
    ax.text(x, y, f"Stage {number}", ha="center", va="center", fontsize=fs.small(),
            color=fs.INK_MUTED)
    ax.text(x, y - 0.055, title, ha="center", va="center", fontsize=6.2,
            color=fs.INK)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    plt = fs.apply()

    figure, ax = plt.subplots(figsize=(7.2, 2.6))
    figure.subplots_adjust(left=0.015, right=0.985, top=0.93, bottom=0.03)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02); ax.axis("off")

    tint = {"rna": "#eaf1fc", "protein": "#fdeee7", "signal": "#e6f6f0",
            "neutral": "#f4f4f2"}

    stage_header(ax, 0.165, 0.95, 1, "Cross-modal alignment")
    box(ax, 0.07, 0.70, "Reference\nRNA", tint["rna"], fs.SERIES[0])
    box(ax, 0.07, 0.34, "Reference\nprotein", tint["protein"], fs.SERIES[1])
    box(ax, 0.235, 0.70, "RNA\nencoder", tint["neutral"], fs.INK_MUTED)
    box(ax, 0.235, 0.34, "Protein\nencoder", tint["neutral"], fs.INK_MUTED)
    arrow(ax, 0.115, 0.70, 0.19, 0.70)
    arrow(ax, 0.115, 0.34, 0.19, 0.34)
    ax.annotate("", xy=(0.235, 0.44), xytext=(0.235, 0.60), zorder=2,
                arrowprops=dict(arrowstyle="<|-|>", linewidth=0.9, color=fs.SERIES[2]))
    ax.text(0.262, 0.52, "contrastive\nalignment", fontsize=fs.small(), ha="left",
            va="center", color=fs.SERIES[2])

    stage_header(ax, 0.525, 0.95, 2, "Retrieval and disagreement")
    box(ax, 0.415, 0.70, "Query\nRNA", tint["rna"], fs.SERIES[0])
    box(ax, 0.415, 0.34, "$K$ nearest\nreference cells", tint["neutral"], fs.INK_MUTED)
    arrow(ax, 0.415, 0.60, 0.415, 0.45)
    arrow(ax, 0.30, 0.52, 0.365, 0.40, style="-|>", linestyle=(0, (3, 2)))

    box(ax, 0.615, 0.53, "weighted mean\n$\\rightarrow$ prior $x_0$",
        tint["protein"], fs.SERIES[1])
    box(ax, 0.615, 0.17, "weighted spread\n$\\rightarrow$ disagreement $d$",
        tint["signal"], fs.SERIES[2], weight="bold")
    arrow(ax, 0.478, 0.37, 0.545, 0.50)
    arrow(ax, 0.478, 0.30, 0.545, 0.20, colour=fs.SERIES[2])
    ax.text(0.415, 0.21, "measured reference\nprotein profiles", fontsize=fs.small(),
            ha="center", va="center", color=fs.INK_MUTED, style="italic")

    stage_header(ax, 0.868, 0.95, 3, "Protein-specific fusion")
    box(ax, 0.845, 0.74, "network\nensemble $\\bar{f}$", tint["neutral"], fs.INK_MUTED,
        fontsize=6.6, weight="bold")
    arrow(ax, 0.415, 0.805, 0.79, 0.805, linestyle=(0, (3, 2)))
    ax.text(0.60, 0.835, "query transcriptome", fontsize=fs.small(),
            ha="center", va="bottom", color=fs.INK_MUTED)
    box(ax, 0.845, 0.48, "per-protein\nfusion", tint["neutral"], fs.INK_MUTED,
        fontsize=6.6, weight="bold")
    ax.annotate("", xy=(0.845, 0.555), xytext=(0.845, 0.665), zorder=2,
                arrowprops=dict(arrowstyle="-|>", linewidth=0.9, color=fs.INK_SECONDARY))
    arrow(ax, 0.685, 0.52, 0.795, 0.49)
    box(ax, 0.962, 0.48, "predicted\nprotein", tint["protein"], fs.SERIES[1])
    arrow(ax, 0.895, 0.48, 0.925, 0.48)
    ax.text(0.845, 0.335, "$\\hat{y}_p = a_p \\bar{f}_p + b_p x_{0,p} + c_p$",
            fontsize=6.4, ha="center", va="center", color=fs.INK)
    ax.text(0.845, 0.265, "weights fitted on held-out\nreference cells",
            fontsize=fs.small(), ha="center", va="center", color=fs.INK_MUTED,
            style="italic")
    box(ax, 0.962, 0.17, "reliability\ngrade", tint["signal"], fs.SERIES[2],
        weight="bold")
    arrow(ax, 0.685, 0.17, 0.918, 0.17, colour=fs.SERIES[2])

    for x in (0.345, 0.735):
        ax.plot([x, x], [0.04, 0.90], color=fs.GRID, linewidth=0.8, zorder=1)

    fs.save(figure, args.out, "fig_overview")
    plt.close(figure)


if __name__ == "__main__":
    main()
