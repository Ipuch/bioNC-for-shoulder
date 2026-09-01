"""
Henninger tutorial - step 2: build the upper-limb model WITH joint constraints.

Same segments as step 1, but the joints now carry anatomy: a constant-length "clavicle"
keeps the IJ (thorax) <-> AC (scapula) distance fixed, and the glenohumeral joint is
spherical (ball-and-socket: GSC and GSChum coincide). The scapulothoracic joint stays FREE.

We still only reconstruct the model directly from the markers here (``Q_from_markers``, no
optimisation) to *see* the constrained model on the data. Enforcing the constraints is the
job of the inverse kinematics -- that is step 3.

Run (from the repo root, inside the ``bionc`` conda env):
    python examples/henninger/02_build_model_with_constraints.py
"""

from pathlib import Path


from examples._shared.ik import load_markers
from examples._shared.viz import animate_model
from examples.henninger.model import build_model_constrained

DATA = str(Path(__file__).resolve().parents[1] / "data" / "testFlorent_HenningerData.c3d")


def main():
    model = build_model_constrained(DATA)

    markers = load_markers(model, DATA)
    Qxp = model.Q_from_markers(markers)

    animate_model(model, Qxp, DATA, show_natural_vectors=True)


if __name__ == "__main__":
    main()
