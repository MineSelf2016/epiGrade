from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT.parent
sys.path.insert(0, str(ROOT))

METHOD = "EpiGrade"
FIGURE_SCRIPTS = sorted((ROOT / "experiments").glob("fig_*.py")) + \
    [ROOT / "experiments" / "make_figures.py"]


def check_method_label(results: Path) -> list[str]:
    problems = []
    for path in sorted(results.glob("xc_*/summary.json")):
        if path.parent.name.startswith("xc_consensus_"):
            continue
        methods = json.loads(path.read_text()).get("methods", {})
        if METHOD not in methods:
            problems.append(f"{path.parent.name}: no '{METHOD}' key "
                            f"(found {sorted(methods)})")
    for path in sorted(results.glob("xc_*/per_protein_pcc.csv")):
        if path.parent.name.startswith("xc_consensus_"):
            continue
        header = path.read_text().split("\n", 1)[0]
        if METHOD not in header.split(","):
            problems.append(f"{path.parent.name}/{path.name}: no '{METHOD}' column")
    return problems


def check_redundant_encoding() -> list[str]:
    problems = []
    for script in FIGURE_SCRIPTS:
        lines = script.read_text().split("\n")
        for index, line in enumerate(lines):
            if not re.search(r"ax\.(scatter|plot|bar|barh)\(", line):
                continue
            block = "\n".join(lines[index:index + 5])
            if "SERIES[" in block and not any(
                    key in block for key in ("marker=", "linestyle=", "hatch=")):
                problems.append(f"{script.name}:{index + 1} distinguishes a series "
                                f"by colour alone")
    return problems


def check_font_floor() -> list[str]:
    from experiments import figure_style
    problems = []
    for script in FIGURE_SCRIPTS:
        for match in re.finditer(r"fontsize=([0-9.]+)", script.read_text()):
            if float(match.group(1)) < 6.0:
                problems.append(f"{script.name}: fontsize={match.group(1)} "
                                f"below the 6 pt print floor")
    return problems


def check_tex_escapes() -> list[str]:
    problems = []
    for script in FIGURE_SCRIPTS:
        for number, line in enumerate(script.read_text().splitlines(), 1):
            if re.search(r"(?<!\\)\\%", line) and not line.lstrip().startswith("#"):
                problems.append(f"{script.name}:{number}: escaped '%' in a label; "
                                "matplotlib draws the backslash")
    return problems


def check_no_placeholders() -> list[str]:
    return [f"{s.name}: still draws a 'pending' placeholder"
            for s in FIGURE_SCRIPTS if "pending" in s.read_text()]


def main() -> None:
    results = REVISION / "results_fused"
    problems = (check_method_label(results) + check_redundant_encoding()
                + check_font_floor() + check_no_placeholders()
                + check_tex_escapes())
    if problems:
        print("FIGURE CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print(f"FIGURE CHECK PASSED ({len(FIGURE_SCRIPTS)} scripts)")


if __name__ == "__main__":
    main()
