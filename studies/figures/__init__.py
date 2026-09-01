"""
Figures for the shoulder calibration studies, kept apart from the code that computes them.

The calibrations are slow (a fold is minutes, the whole leave-one-out sweep the better part of an
hour) and figures are the part you iterate on, so nothing here re-solves anything it can avoid:
every script reads the arrays cached under ``results/`` by
:mod:`studies.shoulder_calibration_loo` and :mod:`studies.figures.calibration_steps`, and each is
runnable on its own.

    python studies/figures/frame_selection.py     # is the calibration subsample diverse enough?
    python studies/figures/calibration_steps.py   # what each of the three steps did
    python studies/figures/leave_one_out.py       # does it generalise?
    python studies/figures/frame_selection_replay.py   # pyorerun, needs a graphical session

Pass ``--save`` to write PNGs into ``results/figures/`` instead of opening windows, and
``--refresh`` (where a script has its own cache) to recompute rather than reuse it.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

# one colour per trial family, used everywhere so the two protocols stay recognisable across figures
KIND_COLORS = {"ANALYTIC": "tab:red", "FUNCTIONAL": "tab:blue"}
STEP_COLORS = {
    "free": "tab:gray",
    "reference": "tab:blue",
    "step1": "tab:purple",
    "step2": "tab:green",
    "step3": "tab:red",
}


def parse_args(description: str, refresh: bool = False) -> argparse.Namespace:
    """Standard command line: ``--save`` to write PNGs, ``--refresh`` to bypass a script's cache."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save", action="store_true", help="write PNGs to results/figures instead of showing")
    if refresh:
        parser.add_argument("--refresh", action="store_true", help="recompute instead of reusing the cache")
    return parser.parse_args()


def finish(named_figures: dict, save: bool) -> None:
    """Either write every figure to ``results/figures/<name>.png`` or open them all."""
    if not save:
        plt.show()
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, figure in named_figures.items():
        path = FIGURE_DIR / f"{name}.png"
        figure.savefig(path, dpi=150)
        print(f"wrote {path}")
