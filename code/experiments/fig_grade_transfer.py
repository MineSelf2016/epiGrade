from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import figure_style as fs
from grade_transfer import COVERAGE, HEADLINE_COVERAGE, TRANSFERS

CURVES = [("grade", "Frozen grade"),
          ("disagreement", "Target disagreement only"),
          ("reference", "Accuracy in the reference"),
          ("random", "Random order")]


def panel_scatter(ax, block: pd.DataFrame, summary: pd.DataFrame, label: str):
    row = summary[summary.criterion == "grade"].iloc[0]
    ax.scatter(block["grade"], block["target_pcc"], s=16,
               facecolor=fs.SERIES[0], marker=fs.MARKERS[0], edgecolor="white",
               linewidth=0.4, zorder=3)
    limits = (0.0, 1.0)
    ax.plot(limits, limits, linewidth=0.6, linestyle=(0, (3, 2)),
            color=fs.INK_MUTED, zorder=2)
    below = float((block["grade"] > block["target_pcc"]).mean())
    ax.text(0.96, 0.06, f"{100 * below:.0f}% below", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=fs.MIN_FONT,
            color=fs.INK_SECONDARY)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontsize=fs.small() + 0.4, color=fs.INK, pad=3)
    ax.text(0.04, 0.96,
            f"$\\rho = {row.rho:.2f}$\n[{row.ci_low:.2f}, {row.ci_high:.2f}]\n"
            f"$n = {int(row.n_proteins)}$",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=fs.MIN_FONT, color=fs.INK_SECONDARY)
    fs.grid(ax)


def panel_coverage(ax, summary: pd.DataFrame):
    fractions = np.array(COVERAGE)
    columns = [f"cover_{int(100 * f)}" for f in COVERAGE]
    oracle = summary[summary.criterion == "oracle"]
    if not oracle.empty:
        ax.fill_between(100 * fractions, oracle.iloc[0][columns].to_numpy(float),
                        summary[summary.criterion == "random"].iloc[0][columns]
                        .to_numpy(float),
                        color=fs.INK_MUTED, alpha=0.12, linewidth=0, zorder=1)
    for index, (criterion, _) in enumerate(CURVES):
        block = summary[summary.criterion == criterion]
        if block.empty:
            fs.skip(f"coverage curve: {criterion} missing")
            continue
        ax.plot(100 * fractions, block.iloc[0][columns].to_numpy(float),
                color=fs.SERIES[index], marker=fs.MARKERS[index],
                linestyle=fs.LINESTYLES[index % len(fs.LINESTYLES)],
                markersize=3.0, linewidth=1.1, zorder=3)
    ax.axvline(100 * HEADLINE_COVERAGE, color=fs.INK_MUTED, linewidth=0.6,
               linestyle=(0, (1.5, 1.5)), zorder=2)
    ax.set_xlabel("Selected proteins (%)")
    fs.grid(ax)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    plt = fs.apply()

    per_protein = args.results / "grade_transfer.csv"
    summary_path = args.results / "grade_transfer_summary.csv"
    if not (per_protein.exists() and summary_path.exists()):
        fs.skip("grade_transfer.py has not been run")
        fs.require_complete("fig_grade_transfer")
        return
    proteins = pd.read_csv(per_protein)
    summary = pd.read_csv(summary_path)

    figure = plt.figure(figsize=(7.2, 4.6))
    spec = figure.add_gridspec(2, 3, height_ratios=[1.0, 0.86],
                               hspace=0.52, wspace=0.34)

    for column, (_, _, label, stem) in enumerate(TRANSFERS):
        block = proteins[proteins.transfer == stem]
        rows = summary[summary.transfer == stem]
        if block.empty or rows.empty:
            fs.skip(f"transfer {stem} has no stored result")
            continue
        top = figure.add_subplot(spec[0, column])
        panel_scatter(top, block, rows, label)
        fs.panel_label(top, "abc"[column], "")
        if column == 0:
            top.set_ylabel("Observed accuracy\nin the target cohort")
        top.set_xlabel("Frozen grade")

        bottom = figure.add_subplot(spec[1, column])
        panel_coverage(bottom, rows)
        fs.panel_label(bottom, "def"[column], "")
        if column == 0:
            bottom.set_ylabel("Mean accuracy\nof selected proteins")

    handles = [plt.Line2D([], [], color=fs.SERIES[i], marker=fs.MARKERS[i],
                          linestyle=fs.LINESTYLES[i % len(fs.LINESTYLES)],
                          markersize=3.0, linewidth=1.1, label=name)
               for i, (_, name) in enumerate(CURVES)]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=fs.INK_MUTED,
                                 alpha=0.12, edgecolor="none",
                                 label="Random to best possible"))
    figure.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
                  bbox_to_anchor=(0.5, -0.025), fontsize=fs.small(),
                  handlelength=2.4, columnspacing=1.5)

    fs.require_complete("fig_grade_transfer")
    fs.save(figure, args.out, "fig_grade_transfer")
    plt.close(figure)


if __name__ == "__main__":
    main()
