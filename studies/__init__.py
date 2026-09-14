"""Short shoulder investigations built on the model builders in ``examples/``.

See ``studies/README.md`` for what each script asks and answers. The all-frames calibration
engine (``kinematic_calibration``) and the three-step session calibration
(``shoulder_calibration``) are importable, so a study, a figure or a test can reuse them
without re-running anything.
"""

from pathlib import Path

# The glenohumeral constraint of the calibration's step 3, and what names the tree it writes into.
# Both the drivers (which write) and the figures (which read) need that name, so it is resolved here
# once -- this package's __init__ is the only module both sides import that pulls nothing heavy.
GLENOHUMERAL = "spherical"
RESULTS_ROOTS = {"spherical": "results", "constant_length": "results_gh_constant"}


def results_root(glenohumeral: str = GLENOHUMERAL) -> Path:
    """
    Where a run writes: ``results/`` for the spherical glenohumeral joint, ``results_gh_constant/``
    for the constant-length one.

    Two whole trees rather than a suffix per file: a run produces folds, a calibration cache and
    figures that only mean anything together, and keeping them apart is what lets the two joint
    models be compared without either overwriting the other.
    """
    if glenohumeral not in RESULTS_ROOTS:
        raise ValueError(f"glenohumeral must be one of {tuple(RESULTS_ROOTS)}, got {glenohumeral!r}")
    return Path(__file__).resolve().parents[1] / RESULTS_ROOTS[glenohumeral]
