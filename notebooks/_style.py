"""
Shared plot style and checks for the example notebooks.

The palette was validated for colour-vision deficiency on the paper surface:
the first three series colours are distinguishable in any combination; the
fourth (ochre) is for line charts only, next to indigo.
"""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

PAPER = "#F7F3EA"
INK = "#1C1B17"
MUTED = "#6B675C"
GRID = "#E4DDCC"

OXIDE = "#B4442B"
VERDIGRIS = "#00806A"
INDIGO = "#3F51B5"
OCHRE = "#A86A00"
SERIES = [OXIDE, VERDIGRIS, INDIGO, OCHRE]

# Sequential (magnitude): one hue, light to dark. Diverging (sign): oxide / neutral / verdigris.
SEQUENTIAL = LinearSegmentedColormap.from_list("ledger_seq", ["#ECEEF8", "#8C98D6", INDIGO, "#1B2563"])
DIVERGING = LinearSegmentedColormap.from_list("ledger_div", [OXIDE, "#E6E1D6", VERDIGRIS])

_FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "img"


def use() -> None:
    """Apply the notebook style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": PAPER,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.facecolor": PAPER,
        "savefig.bbox": "tight",
        "axes.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "text.color": INK,
        "font.size": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.titleweight": "semibold",
        "figure.constrained_layout.use": True,
        "image.cmap": "ledger_seq",
    })
    if "ledger_seq" not in mpl.colormaps:
        mpl.colormaps.register(SEQUENTIAL)
        mpl.colormaps.register(DIVERGING)


def label_line_end(ax, x, y, text: str, color: str, dx: float = 4) -> None:
    """Write a series label just right of the last point of a line."""
    ax.annotate(text, xy=(x[-1], y[-1]), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.15", fc=PAPER, ec=color, lw=1))


def save(fig, name: str) -> None:
    """Save a figure to docs/img when FINPRICING_SAVE_FIGS is set (used for the README)."""
    if os.environ.get("FINPRICING_SAVE_FIGS"):
        _FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(_FIG_DIR / f"{name}.png")


_results: list[tuple[bool, str]] = []


def check(condition: bool, message: str) -> None:
    """Record and print a model check; raises if it fails, so a broken model stops the notebook."""
    ok = bool(condition)
    _results.append((ok, message))
    print(f"{'PASS' if ok else 'FAIL'}  {message}")
    if not ok:
        raise AssertionError(message)


def summary() -> None:
    """Print all checks run in this notebook."""
    print(f"{sum(ok for ok, _ in _results)}/{len(_results)} checks passed")
    for ok, message in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {message}")


plt.style.use("default")
use()
