from __future__ import annotations

import re
from collections import Counter

import numpy as np

_CONTROL_PATTERNS = (
    re.compile(r"isotype", re.IGNORECASE),
    re.compile(r"^(rat|mouse|rt)[-_]?ig[gm]", re.IGNORECASE),
    re.compile(r"^ig[gm]\d?[a-z]?$", re.IGNORECASE),
    re.compile(r"^(mopc|rtk|mpc)[-_]?\d+", re.IGNORECASE),
    re.compile(r"unmapped|blank|empty|^anti-pe$", re.IGNORECASE),
    re.compile(r"^hashtag|^htо|^hto[-_]?\d*$", re.IGNORECASE),
)

_SUFFIX_PATTERNS = (
    re.compile(r"--.*$"),
    re.compile(r"[-_](TSA|BIO|PROT|ADT|AB)$", re.IGNORECASE),
    re.compile(r"[-_](control|ctrl)$", re.IGNORECASE),
)

_ALIASES = {
    "MHCII": "HLA-DR", "HLA-DR-DP-DQ": "HLA-DR", "HLADR": "HLA-DR",
    "IL7RALPHA": "CD127", "IL-7RA": "CD127", "IL7RA": "CD127",
    "PD1": "CD279", "PD-1": "CD279", "PDL1": "CD274", "PD-L1": "CD274",
    "B220": "CD45R", "OX40": "CD134", "C-KIT": "CD117", "CKIT": "CD117",
    "PECAM": "CD31", "EPCAM": "CD326", "LAMP1": "CD107A",
    "TIM3": "CD366", "TIM-3": "CD366", "CLEC9A": "CD370",
    "BDCA1": "CD1C", "BDCA2": "CD303", "BDCA3": "CD141",
    "FCER1A": "FCERIA", "SIGLEC8": "SIGLEC-8",
    "HLA-ABC": "HLA-ABC", "HLAABC": "HLA-ABC",
    "TCRGD": "TCR-GD", "TCRG/D": "TCR-GD", "TCRGAMMADELTA": "TCR-GD",
    "TCRAB": "TCR-AB", "TCRA/B": "TCR-AB",
    "CD26(ADENOSINE)": "CD26", "ADENOSINE": "CD26",
    "TCRA7-2": "TCR-VA7.2", "TCR-V-7-2": "TCR-VA7.2", "TCRVA7-2": "TCR-VA7.2",
    "TCR-V-2": "TCR-VD2", "TCRVD2": "TCR-VD2",
    "TCR-V-9": "TCR-VG9", "TCRVG9": "TCR-VG9",
    "CCR7": "CD197", "CCR5": "CD195", "CCR6": "CD196", "CXCR3": "CD183",
    "CXCR5": "CD185", "CD197-CCR7": "CD197",
    "CD8A": "CD8", "CD8B": "CD8B", "CD79B": "CD79B",
    "ICOS": "CD278", "IL7RA": "CD127",
}

_CD_IN_NAME = re.compile(
    r"(?:^|[^A-Za-z0-9])CD(\d+[A-Za-z]{0,3})(?![0-9])", re.IGNORECASE
)


def is_control(name: str) -> bool:
    return any(pattern.search(str(name)) for pattern in _CONTROL_PATTERNS)


def canonical_protein(name: str) -> str | None:
    raw = str(name).strip()
    if not raw or is_control(raw):
        return None

    work = raw
    for pattern in _SUFFIX_PATTERNS:
        work = pattern.sub("", work)
    work = work.strip().strip("-_ ")
    if not work or is_control(work):
        return None

    work = re.sub(r"[-_](bio|biotin)$", "", work, flags=re.IGNORECASE)
    if re.match(r"^CD\d", work, re.IGNORECASE):
        work = re.sub(r"[-_]\d+$", "", work)
        work = re.sub(r"\.\d+$", "", work)

    upper = work.upper().replace(" ", "").replace(".", "-")
    if upper in _ALIASES:
        return _ALIASES[upper]

    match = _CD_IN_NAME.search(work)
    if match:
        return f"CD{match.group(1).upper()}"

    upper = re.sub(r"\(.*?\)", "", upper).strip("-_ ")
    if upper in _ALIASES:
        return _ALIASES[upper]
    return upper or None


def canonicalise_panel(names) -> tuple[list[str | None], dict[str, list[int]]]:
    canonical: list[str | None] = []
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        symbol = canonical_protein(name)
        canonical.append(symbol)
        if symbol is not None:
            groups.setdefault(symbol, []).append(index)
    return canonical, groups


def collapse_duplicates(matrix: np.ndarray, groups: dict[str, list[int]]
                        ) -> tuple[np.ndarray, list[str]]:
    symbols = sorted(groups)
    collapsed = np.empty((matrix.shape[0], len(symbols)), dtype=np.float32)
    for position, symbol in enumerate(symbols):
        columns = groups[symbol]
        block = matrix[:, columns]
        collapsed[:, position] = block.mean(axis=1) if len(columns) > 1 else block[:, 0]
    return collapsed, symbols


def intersect_panels(*panels) -> list[str]:
    sets = []
    for panel in panels:
        _, groups = canonicalise_panel(panel)
        sets.append(set(groups))
    if not sets:
        return []
    shared = set.intersection(*sets)
    return sorted(shared)


def panel_report(named_panels: dict[str, list[str]]) -> str:
    canon = {name: canonicalise_panel(panel)[1] for name, panel in named_panels.items()}
    lines = ["panel                 raw  canonical  controls dropped"]
    for name, panel in named_panels.items():
        dropped = sum(1 for feature in panel if is_control(feature))
        lines.append(f"{name:<20} {len(panel):>4} {len(canon[name]):>10} {dropped:>17}")
    lines.append("")
    lines.append("pairwise canonical overlap")
    names = list(named_panels)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = set(canon[a]) & set(canon[b])
            lines.append(f"  {a:<18} x {b:<18} {len(shared):>4}")
    return "\n".join(lines)
