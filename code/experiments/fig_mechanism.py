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

DATASETS = [("GSE3P", "donor", "PBMC 3$'$"), ("ECCITE", "donor", "PBMC 5$'$"),
            ("CBMC", "cell", "CBMC")]


def _pooled(results: Path, dataset: str, split: str, variant: str = "full"):
    run = results / f"{dataset}_{split}_{variant}"
    path = run / "pooled_out_of_fold.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=True)
    names = list(json.loads((run / "run_settings.json").read_text())["protein_names"])
    return data, names


def _fold_disagreement(results: Path, dataset: str, split: str):
    run = results / f"{dataset}_{split}_full"
    folds = sorted(p for p in run.glob("fold_*") if p.is_dir())
    blocks = [np.load(p / "outputs.npz", allow_pickle=True)["disagreement"]
              for p in folds if (p / "outputs.npz").exists()]
    return np.concatenate(blocks).mean(axis=0) if blocks else None


def panel_signal(ax, results: Path):
    for index, (dataset, split, label) in enumerate(DATASETS):
        data, _ = _pooled(results, dataset, split)
        disagreement = _fold_disagreement(results, dataset, split)
        if data is None or disagreement is None:
            fs.skip(f"panel a: {dataset}/{split} missing pooled output or disagreement")
            continue
        spread = np.maximum(data["target"].std(axis=0), 1e-8)
        relative = disagreement / spread
        shortfall = 1.0 - protein_pearson(data["prior"], data["target"])
        ax.scatter(relative, shortfall, s=13, facecolor=fs.SERIES[index],
                   marker=fs.MARKERS[index], edgecolor="white", linewidth=0.4,
                   zorder=3,
                   label=f"{label} ($r={np.corrcoef(relative, shortfall)[0,1]:.2f}$)")
    ax.set_xlabel("Relative disagreement $d_p / \\sigma_p$")
    ax.set_ylabel("Prior shortfall $1 - $ PCC-P")
    fs.grid(ax)
    ax.legend(loc="lower right", fontsize=fs.small())
    fs.panel_label(ax, "a", "Disagreement and prior error")


def panel_panel_size(ax, results: Path):
    from epigrade.metrics import protein_pearson

    order = [("CBMC", "cell"), ("ECCITE", "cell"), ("ECCITE", "donor"),
             ("GSE3P", "cell"), ("GSE3P", "donor")]
    label = {"cell": "cell", "donor": "donor"}
    groups = []
    for dataset, split in order:
        run = results / f"{dataset}_{split}_full"
        path = run / "pooled_out_of_fold.npz"
        if not path.exists():
            fs.skip(f"panel b: {dataset}/{split} has no pooled output")
            continue
        data = np.load(path, allow_pickle=True)
        gain = (protein_pearson(data["prediction"], data["target"])
                - protein_pearson(data["prior"], data["target"]))
        size = json.loads((run / "run_settings.json").read_text())["n_proteins"]
        groups.append((dataset, split, size, gain))
    if not groups:
        fs.skip("panel b: no dataset has a pooled output")
        return

    span = max(g.max() for _, _, _, g in groups)
    whiskers = []
    for _, _, _, g in groups:
        q1, q3 = np.percentile(g, [25, 75])
        whiskers.append(max(g.min(), q1 - 1.5 * (q3 - q1)))
    floor = np.floor((min(whiskers) - 0.01) * 100) / 100
    ax.set_ylim(floor - 0.012, span * 1.34)
    label_y = span * 1.11

    rng = np.random.default_rng(0)
    for index, (dataset, split, size, gain) in enumerate(groups):
        colour = fs.SERIES[0] if split == "cell" else fs.SERIES[1]
        box = ax.boxplot([gain], positions=[index], widths=0.52, showfliers=False,
                         patch_artist=True, zorder=3,
                         medianprops=dict(color=fs.INK, linewidth=1.3),
                         whiskerprops=dict(color=fs.INK_MUTED, linewidth=0.7),
                         capprops=dict(color=fs.INK_MUTED, linewidth=0.7))
        for patch in box["boxes"]:
            patch.set(facecolor=colour, alpha=0.30, edgecolor=colour, linewidth=0.8,
                      hatch=fs.HATCHES[0 if split == "cell" else 1])
        jitter = rng.uniform(-0.17, 0.17, size=gain.size)
        inside = gain >= floor
        ax.scatter(index + jitter[inside], gain[inside], s=3.0, color=colour,
                   alpha=0.55, linewidth=0, zorder=4)
        below = int((~inside).sum())
        if below:
            ax.scatter([index], [floor - 0.006], marker="v", s=14, color=colour,
                       linewidth=0, zorder=5, clip_on=False)
            ax.text(index - 0.12, floor - 0.006, f"{below} below", ha="right",
                    va="center", fontsize=fs.small() - 0.5, color=fs.INK_SECONDARY)
        ax.text(index, label_y, f"{100 * (gain > 0).mean():.0f}%",
                ha="center", va="bottom", fontsize=fs.small(),
                color=fs.INK_SECONDARY, clip_on=False)

    ax.axhline(0.0, color=fs.INK_MUTED, linewidth=0.6, linestyle=(0, (3, 3)), zorder=2)
    ax.set_xticks(range(len(groups)))
    short = {"CBMC": "CBMC", "ECCITE": "PBMC 5'", "GSE3P": "PBMC 3'"}
    ax.set_xticklabels([f"{short.get(d, d)}\n{n}P\n{s}" for d, s, n, _ in groups],
                       fontsize=fs.small())
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel("PCC-P gained over the prior")
    ax.text(-0.52, span * 1.24, "proportion of proteins with improved accuracy",
            ha="left", va="bottom", fontsize=fs.small() - 0.5,
            color=fs.INK_MUTED)
    fs.grid(ax, axis="y")
    fs.panel_label(ax, "b", "Improvement by panel")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "paper" / "figures")
    args = parser.parse_args()
    plt = fs.apply()

    figure = plt.figure(figsize=(7.2, 2.55))
    spec = figure.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.28)
    panel_signal(figure.add_subplot(spec[0, 0]), args.results)
    panel_panel_size(figure.add_subplot(spec[0, 1]), args.results)
    fs.require_complete("fig_mechanism")
    fs.save(figure, args.out, "fig_mechanism")
    plt.close(figure)


if __name__ == "__main__":
    main()
