from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epigrade.reliability import FEATURES, ReliabilityEstimator

TRANSFERS = [
    ("xc_PBMC3P_to_BM", "GSE3P", "PBMC 3$'$ $\\rightarrow$ bone marrow", "BM"),
    ("xc_PBMC3P_to_COVID", "GSE3P", "PBMC 3$'$ $\\rightarrow$ COVID-19", "CovidThree"),
    ("xc_PBMC5P_to_COVID", "ECCITE", "PBMC 5$'$ $\\rightarrow$ COVID-19", "CovidFive"),
]
CRITERIA = ("grade", "grade_target", "reference", "disagreement")
COVERAGE = (0.2, 0.4, 0.6, 0.8, 1.0)
HEADLINE_COVERAGE = 0.4


def load_transfer(results: Path, run: str, reference: pd.DataFrame):
    directory = results / run
    harmonisation = json.loads((directory / "harmonisation.json").read_text())
    shared = list(harmonisation["shared_proteins"])
    achieved = pd.read_csv(directory / "per_protein_pcc.csv").set_index("protein")
    disagreement = np.load(directory / "disagreement.npy").mean(axis=0)

    keep = [p for p in shared if p in reference.index and p in achieved.index]
    if not keep:
        raise ValueError(f"{run}: no shared protein is described by the reference")
    position = {p: i for i, p in enumerate(shared)}
    block = reference.loc[keep]
    return pd.DataFrame({
        "protein": keep,
        "target_pcc": achieved.loc[keep, "EpiGrade"].to_numpy(),
        "target_disagreement": disagreement[[position[p] for p in keep]],
        **{c: block[c].to_numpy() for c in block.columns},
    })


def build_criteria(table: pd.DataFrame, estimator: ReliabilityEstimator,
                   columns: list[str]) -> dict[str, np.ndarray]:
    reference_side = {c: table[c].to_numpy() for c in columns}
    scores = {"grade": estimator.predict(reference_side),
              "reference": table["pcc_full"].to_numpy()}

    if "rel_disagreement" in columns and "atlas_sd" in table:
        target_side = dict(reference_side)
        target_side["rel_disagreement"] = (
            table["target_disagreement"].to_numpy()
            / np.maximum(table["atlas_sd"].to_numpy(), 1e-8))
        scores["grade_target"] = estimator.predict(target_side)
        scores["disagreement"] = -target_side["rel_disagreement"]
    return scores


def bootstrap_rho(x: np.ndarray, y: np.ndarray, draws: int, seed: int):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        pick = rng.integers(0, len(x), len(x))
        if len(np.unique(y[pick])) < 3:
            continue
        rho = spearmanr(x[pick], y[pick]).statistic
        if np.isfinite(rho):
            values.append(rho)
    if not values:
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def coverage_curve(score: np.ndarray, achieved: np.ndarray) -> dict[float, float]:
    order = np.argsort(score)[::-1]
    return {f: float(achieved[order[: max(int(f * len(order)), 1)]].mean())
            for f in COVERAGE}


def random_curve(achieved: np.ndarray, draws: int, seed: int) -> dict[float, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for f in COVERAGE:
        k = max(int(f * len(achieved)), 1)
        out[f] = float(np.mean([achieved[rng.permutation(len(achieved))[:k]].mean()
                                for _ in range(draws)]))
    return out


def analyse(results: Path, draws: int, seed: int):
    reference_table = pd.read_csv(results / "protein_reliability.csv")
    per_protein, summary = [], []

    for run, panel, label, stem in TRANSFERS:
        if not (results / run).exists():
            raise SystemExit(f"{run} is missing; run the cross-cohort experiments first")
        reference = reference_table[reference_table.dataset == panel].set_index("protein")
        columns = [c for c in FEATURES if c in reference and reference[c].notna().all()]
        estimator = ReliabilityEstimator().fit(
            {c: reference[c].to_numpy() for c in columns},
            reference["pcc_full"].to_numpy())

        table = load_transfer(results, run, reference)
        scores = build_criteria(table, estimator, columns)
        achieved = table["target_pcc"].to_numpy()

        block = table[["protein", "target_pcc"]].copy()
        block.insert(0, "transfer", stem)
        for name, score in scores.items():
            block[name] = score
        per_protein.append(block)

        curves = {name: coverage_curve(score, achieved)
                  for name, score in scores.items()}
        curves["oracle"] = coverage_curve(achieved, achieved)
        curves["random"] = random_curve(achieved, draws // 2, seed)

        for name, score in scores.items():
            rho = spearmanr(score, achieved)
            low, high = bootstrap_rho(score, achieved, draws, seed)
            summary.append({
                "transfer": stem, "label": label, "criterion": name,
                "n_proteins": len(achieved), "rho": rho.statistic,
                "p_value": rho.pvalue, "ci_low": low, "ci_high": high,
                **{f"cover_{int(100 * f)}": curves[name][f] for f in COVERAGE},
            })
        for name in ("oracle", "random"):
            summary.append({
                "transfer": stem, "label": label, "criterion": name,
                "n_proteins": len(achieved), "rho": np.nan, "p_value": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                **{f"cover_{int(100 * f)}": curves[name][f] for f in COVERAGE},
            })

    return pd.concat(per_protein, ignore_index=True), pd.DataFrame(summary)


def _range(values, digits: int = 2) -> str:
    low, high = float(np.min(values)), float(np.max(values))
    if abs(low - high) < 10 ** (-digits):
        return f"${low:.{digits}f}$"
    return f"${low:.{digits}f}$ to ${high:.{digits}f}$"


def write_macros(summary: pd.DataFrame, per_protein: pd.DataFrame,
                 path: Path) -> None:
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    macros: dict[str, str] = {}

    def add(name: str, value: str) -> None:
        macros[name.translate(str.maketrans(digits))] = value

    head = f"cover_{int(100 * HEADLINE_COVERAGE)}"
    grade = summary[summary.criterion == "grade"]
    whole = summary[summary.criterion == "oracle"]["cover_100"]

    add("numTransferRhoRange", _range(grade.rho))
    spelled = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    add("numTransferCount", spelled.get(len(grade), str(len(grade))))
    add("numTransferSharedRange",
        f"{int(grade.n_proteins.min())} to {int(grade.n_proteins.max())}")
    add("numTransferCILow", f"${grade.ci_low.min():.2f}$")
    for _, row in grade.iterrows():
        add(f"numTransferRho{row.transfer}", f"${row.rho:.2f}$")
        add(f"numTransferCI{row.transfer}",
            f"$[{row.ci_low:.2f}, {row.ci_high:.2f}]$")

    gains = grade[head].to_numpy() - whole.to_numpy()
    add("numTransferGainRange", _range(gains))
    add("numTransferCoverage", f"{int(100 * HEADLINE_COVERAGE)}\\%")
    add("numTransferWholeRange", _range(whole))
    add("numTransferTopRange", _range(grade[head]))

    for name, stem in (("reference", "Reference"), ("disagreement", "Disagreement"),
                       ("grade_target", "TargetAware")):
        block = summary[summary.criterion == name]
        if block.empty:
            continue
        add(f"numTransfer{stem}RhoRange", _range(block.rho))
        add(f"numTransfer{stem}GapRange",
            _range(grade[head].to_numpy() - block[head].to_numpy(), 3))

    optimism, below = [], []
    for _, block in per_protein.groupby("transfer", sort=False):
        optimism.append(block["grade"].mean() - block["target_pcc"].mean())
        below.append((block["grade"] > block["target_pcc"]).mean())
    add("numTransferOptimismRange", _range(optimism))
    add("numTransferOptimismWorst", f"${max(optimism):.2f}$")
    add("numTransferBelowRange",
        f"{100 * min(below):.0f}\\% to {100 * max(below):.0f}\\%")

    random_block = summary[summary.criterion == "random"]
    add("numTransferRandomGainRange",
        _range(random_block[head].to_numpy() - whole.to_numpy(), 3))

    path.write_text(
        "% generated by grade_transfer.py -- do not edit\n"
        + "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items()))
        + "\n", encoding="utf-8")
    print("macros written to", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT.parent / "results_fused")
    parser.add_argument("--sections", type=Path,
                        default=ROOT.parent / "paper" / "sections")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    per_protein, summary = analyse(args.results, args.draws, args.seed)
    per_protein.to_csv(args.results / "grade_transfer.csv", index=False)
    summary.to_csv(args.results / "grade_transfer_summary.csv", index=False)
    write_macros(summary, per_protein, args.sections / "numbers_transfer.tex")

    head = f"cover_{int(100 * HEADLINE_COVERAGE)}"
    print("\nOrdering of target accuracy by each criterion (Spearman):")
    for stem in summary.transfer.unique():
        block = summary[summary.transfer == stem]
        print(f"  {stem} (n = {int(block.n_proteins.iloc[0])})")
        for _, row in block.iterrows():
            rho = "     --" if np.isnan(row.rho) else f"{row.rho:+.2f}"
            interval = ("" if np.isnan(row.ci_low)
                        else f" [{row.ci_low:+.2f}, {row.ci_high:+.2f}]")
            print(f"    {row.criterion:14s} rho {rho}{interval:>20s}"
                  f"   top {int(100 * HEADLINE_COVERAGE)}% {row[head]:.3f}"
                  f"   all {row.cover_100:.3f}")


if __name__ == "__main__":
    main()
