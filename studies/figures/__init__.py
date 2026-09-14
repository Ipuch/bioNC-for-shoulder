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

Pass ``--save`` to write PNGs into ``<tree>/figures/`` instead of opening windows, ``--refresh``
(where a script has its own cache) to recompute rather than reuse it, and ``--gh`` to read and write
the tree of the other glenohumeral model (see :func:`studies.results_root`).
"""

import argparse

import matplotlib.pyplot as plt

from studies import GLENOHUMERAL, RESULTS_ROOTS, results_root

# one colour per trial family, used everywhere so the two protocols stay recognisable across figures
KIND_COLORS = {"ANALYTIC": "tab:red", "FUNCTIONAL": "tab:blue"}
STEP_COLORS = {
    "free": "tab:gray",
    "reference": "tab:blue",
    "step1": "tab:purple",
    "step2": "tab:green",
    "step3": "tab:red",
}


def figure_dir(glenohumeral: str = GLENOHUMERAL):
    """Where this run's PNGs go, one directory per glenohumeral model."""
    return results_root(glenohumeral) / "figures"


def parse_args(description: str, refresh: bool = False) -> argparse.Namespace:
    """
    Standard command line: ``--save`` to write PNGs, ``--gh`` to pick the results tree, and
    ``--refresh`` to bypass a script's cache.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save", action="store_true", help="write PNGs to <tree>/figures instead of showing")
    parser.add_argument(
        "--gh",
        default=GLENOHUMERAL,
        choices=tuple(RESULTS_ROOTS),
        help=f"which glenohumeral model's results to read and write (default: {GLENOHUMERAL})",
    )
    if refresh:
        parser.add_argument("--refresh", action="store_true", help="recompute instead of reusing the cache")
    return parser.parse_args()


def finish(named_figures: dict, save: bool, glenohumeral: str = GLENOHUMERAL) -> None:
    """Either write every figure to ``<tree>/figures/<name>.png`` or open them all."""
    if not save:
        plt.show()
        return

    directory = figure_dir(glenohumeral)
    directory.mkdir(parents=True, exist_ok=True)
    for name, figure in named_figures.items():
        path = directory / f"{name}.png"
        figure.savefig(path, dpi=150)
        print(f"wrote {path}")
