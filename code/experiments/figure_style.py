from __future__ import annotations

MIN_FONT = 6.0

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
LINESTYLES = ("-", (0, (4, 1.6)), (0, (1.4, 1.4)), (0, (5, 1.4, 1, 1.4)))
HATCHES = ("", "///", "...", "\\\\", "xxx", "---", "+++", "ooo")

SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
          "#008300", "#4a3aa7", "#e34948")
SEQUENTIAL = ("#d7e5f8", "#9cc2ef", "#5b9be2", "#2a78d6", "#18508f")
DENSITY = ("#ffffff", "#fdeacb", "#f9c88a", "#f09a52", "#d95f3c", "#a63658",
           "#601d4a")
COOL = ("#a8d8e4", "#5fb0cc", "#3a7ec0", "#2b4a94", "#1f2a5e")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#ffffff"


def apply(scale: float = 1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 7 * scale,
        "axes.labelsize": 7.5 * scale, "axes.titlesize": 6.8 * scale,
        "axes.titleweight": "normal", "axes.titlepad": 4.0,
        "axes.labelcolor": INK, "axes.edgecolor": INK_MUTED, "axes.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "xtick.labelsize": max(6.5 * scale, MIN_FONT),
        "ytick.labelsize": max(6.5 * scale, MIN_FONT),
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.fontsize": max(6.2 * scale, MIN_FONT), "legend.frameon": False,
        "legend.handlelength": 1.0, "legend.borderpad": 0.15,
        "legend.labelspacing": 0.25, "legend.handletextpad": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "lines.linewidth": 2.0, "lines.markersize": 4.0,
        "hatch.linewidth": 0.5,
    })
    return plt


_SKIPPED: list[str] = []


def skip(reason: str) -> None:
    _SKIPPED.append(reason)


def require_complete(name: str) -> None:
    if _SKIPPED:
        listing = "\n  ".join(_SKIPPED)
        _SKIPPED.clear()
        raise SystemExit(f"{name}: incomplete, refusing to write.\n  {listing}")


def small(size: float = MIN_FONT) -> float:
    return max(float(size), MIN_FONT)


def grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax, letter: str, text: str) -> None:
    ax.set_title(f"{letter}  {text}", loc="left")


def colorbar(figure, mappable, ax, label: str, inset: bool = True) -> None:
    if inset:
        cax = ax.inset_axes([0.62, 0.08, 0.05, 0.34])
        bar = figure.colorbar(mappable, cax=cax)
        bar.ax.tick_params(labelsize=MIN_FONT - 0.5, width=0.4, length=1.6, pad=1.2)
        bar.set_label(label, fontsize=MIN_FONT - 0.5, color=INK_SECONDARY, labelpad=2)
    else:
        bar = figure.colorbar(mappable, ax=ax, fraction=0.046, pad=0.03)
        bar.ax.tick_params(labelsize=MIN_FONT, width=0.5, length=2)
        bar.set_label(label, fontsize=MIN_FONT, color=INK_SECONDARY)
    bar.outline.set_linewidth(0.5)


def save(figure, out_dir, name: str) -> None:
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        figure.savefig(out_dir / f"{name}.{extension}")
    print("wrote", out_dir / f"{name}.pdf")
