"""
Clinical tutorial - step 2: build the upper-limb model WITH joint constraints.

Same segments as step 1, but a constant-length "clavicle" now keeps the RCAS (thorax) <->
RCAJ (scapula) distance fixed and the glenohumeral joint is spherical (the shared RGJC point
of scapula and humerus is made coincident). The scapulothoracic joint stays FREE.

We still reconstruct directly from the markers (``Q_from_markers``, no optimisation) to *see*
the constrained model on the data; enforcing the constraints is step 3.

Run (from the repo root, inside the ``bionc`` conda env):
    python examples/clinical/02_build_model_with_constraints.py
"""

from pathlib import Path


from examples._shared.ik import load_markers
from examples._shared.viz import animate_model
from examples.clinical.model import build_model_constrained

DATA = str(Path(__file__).resolve().parents[1] / "data" / "testFlorent_clinicalData.c3d")


def main():
    model = build_model_constrained(DATA, marker_set="both")

    markers = load_markers(model, DATA)
    Qxp = model.Q_from_markers(markers)

    animate_model(model, Qxp, DATA, show_natural_vectors=True)


if __name__ == "__main__":
    main()
