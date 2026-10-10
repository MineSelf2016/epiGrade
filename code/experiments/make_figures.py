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

import figure_style as fs

SERIES, INK, INK_SECONDARY, INK_MUTED, GRID = (
    fs.SERIES, fs.INK, fs.INK_SECONDARY, fs.INK_MUTED, fs.GRID)

METHOD_ORDER = ["EpiGrade", "retrieval_prior", "seurat_anchor", "knn_pca",
                "knn_expression", "global_mean"]
METHOD_LABEL = {"EpiGrade": "EpiGrade", "retrieval_prior": "Prior",
                "seurat_anchor": "Anchor", "knn_pca": "kNN-PCA",
                "knn_expression": "kNN-expr", "global_mean": "Mean"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "font.family": "sans-serif", "font.size": 7,
        "axes.labelsize": 7.5, "axes.titlesize": 6.8, "axes.titleweight": "normal",
        "axes.labelcolor": INK, "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.fontsize": 6.5, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def _within_study(results: Path, dataset: str, split: str = "donor") -> float | None:
    path = results / f"{dataset}_{split}_full" / "summary.json"
    if not path.exists():
        path = results / f"{dataset}_cell_full" / "summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())["fold_mean_pcc_protein"]


def panel_a(ax, results: Path, blocks: list[dict]) -> None:
    within = _within_study(results, "GSE3P")
    present = [m for m in METHOD_ORDER
               if m != "global_mean" and any(m in b["methods"] for b in blocks)]
    positions = np.arange(len(present), dtype=float)
    width = 0.8 / max(len(blocks), 1)

    for index, block in enumerate(blocks):
        offset = (index - (len(blocks) - 1) / 2) * width
        values = [block["methods"][m]["pcc_protein_mean"] if m in block["methods"]
                  else np.nan for m in present]
        ax.bar(positions + offset, values, width * 0.86,
               label=f"{block['atlas']}$\\rightarrow${block['query']} "
                     f"({block['n_shared_proteins']}P)",
               color=SERIES[index], hatch=fs.HATCHES[index],
               edgecolor="white", linewidth=0.45, zorder=3)

    if within is not None:
        ax.axhline(within, color=INK_MUTED, linewidth=0.8,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.text(len(present) - 0.45, within + 0.015,
                f"within study {within:.2f}", ha="right", va="bottom",
                fontsize=fs.small(), color=INK_SECONDARY, zorder=4)

    ax.set_xticks(positions)
    ax.set_xticklabels([METHOD_LABEL[m] for m in present], rotation=38,
                       ha="right", rotation_mode="anchor")
    ax.set_ylabel("PCC-P")
    ax.set_title("a  Transfer across studies", loc="left")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", handlelength=1.0, borderpad=0.15,
              labelspacing=0.25, fontsize=fs.small())


def panel_b(ax, blocks: list[dict], results: Path) -> None:
    plotted, annotations = False, []
    for index, block in enumerate(blocks):
        path = results / f"xc_{block['atlas']}_to_{block['query']}" / "per_population_pcc.csv"
        if not path.exists():
            fs.skip(f"panel b: {block['query']} has no per-population table")
            continue
        frame = pd.read_csv(path)
        if "EpiGrade" not in frame or "mean_retrieval_confidence" not in frame:
            fs.skip(f"panel b: {block['query']} table lacks the EpiGrade column")
            continue
        ax.scatter(frame["mean_retrieval_confidence"], frame["EpiGrade"],
                   s=8 + 40 * frame["n_cells"] / frame["n_cells"].max(),
                   facecolor=SERIES[index], marker=fs.MARKERS[index],
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"{block['atlas']}$\\rightarrow${block['query']}")
        r = np.corrcoef(frame["mean_retrieval_confidence"], frame["EpiGrade"])[0, 1]
        annotations.append(f"{block['query']}: $r={r:+.2f}$")
        plotted = True

    if not plotted:
        fs.skip("panel b: no per-population breakdown could be drawn")
    else:
        ax.set_xlabel("Retrieval confidence $\\bar{s}_q$")
        ax.set_ylabel("PCC-P")
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.03, 0.97, "\n".join(annotations), transform=ax.transAxes,
                ha="left", va="top", fontsize=fs.small(), color=INK_SECONDARY,
                linespacing=1.35)
        ax.legend(loc="lower left", handlelength=1.0, borderpad=0.15,
                  labelspacing=0.25, fontsize=fs.small())
    ax.set_title("b  Confidence is not diagnostic", loc="left")


def panel_c(ax, blocks: list[dict], results: Path) -> None:
    plotted = False
    for index, block in enumerate(blocks):
        path = results / f"xc_{block['atlas']}_to_{block['query']}" / "per_protein_pcc.csv"
        if not path.exists():
            fs.skip(f"panel c: {block['query']} has no per-protein table")
            continue
        frame = pd.read_csv(path)
        if "EpiGrade" not in frame:
            fs.skip(f"panel c: {block['query']} table lacks the EpiGrade column")
            continue
        values = np.sort(frame["EpiGrade"].to_numpy())[::-1]
        ax.plot(np.arange(1, values.size + 1), values, linewidth=2.0,
                marker=fs.MARKERS[index], markersize=3.6,
                linestyle=fs.LINESTYLES[index % len(fs.LINESTYLES)],
                color=SERIES[index],
                markeredgecolor="white", markeredgewidth=0.5, zorder=3,
                label=f"{block['atlas']}$\\rightarrow${block['query']}")
        plotted = True

    if not plotted:
        fs.skip("panel c: no per-protein breakdown could be drawn")
    else:
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=2)
        ax.set_xlabel("Protein rank")
        ax.set_ylabel("PCC-P")
        ax.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", handlelength=1.0, borderpad=0.15,
                  labelspacing=0.25, fontsize=fs.small())
    ax.set_title("c  Per-protein spread", loc="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=ROOT.parent / "results_fused")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt = _style()

    blocks = [json.loads(p.read_text())
              for p in sorted(args.results.glob("xc_*/summary.json"))
              if not p.parent.name.startswith("xc_consensus_")]
    if not blocks:
        print("no cross-cohort results yet; figure not generated")
        return
    missing = [b["query"] for b in blocks if "EpiGrade" not in b["methods"]]
    if missing:
        raise SystemExit(
            "EpiGrade is absent from the cross-cohort results for "
            f"{', '.join(missing)}; the stored method label is probably stale.")

    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    panel_a(axes[0], args.results, blocks)
    panel_b(axes[1], blocks, args.results)
    panel_c(axes[2], blocks, args.results)
    figure.subplots_adjust(wspace=0.36, bottom=0.26)
    fs.require_complete("fig_crosscohort")
    for extension in ("pdf", "png"):
        figure.savefig(args.out / f"fig_crosscohort.{extension}")
    plt.close(figure)
    print("wrote", args.out / "fig_crosscohort.pdf")


if __name__ == "__main__":
    main()
