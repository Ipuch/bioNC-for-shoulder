"""
Clinical tutorial - step 1: build the upper-limb model WITHOUT joint constraints.

Every joint is FREE, so the three segments are only tied together by the markers they share.
The clinical dataset carries both anatomical landmarks and skin-cluster markers; here we track
them all (``marker_set="both"``). We calibrate the model, reconstruct it directly from the
markers with ``Q_from_markers`` (no optimisation) and visualise it against the raw data.

Run (from the repo root, inside the ``bionc`` conda env):
    python examples/clinical/01_build_model_no_constraints.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on the path

from examples._shared.ik import load_markers
from examples._shared.viz import animate_model
from examples.clinical.model import build_model_free

DATA = str(Path(__file__).resolve().parents[1] / "data" / "testFlorent_clinicalData.c3d")


def main():
    model = build_model_free(DATA, marker_set="both")

    markers = load_markers(model, DATA)
    Qxp = model.Q_from_markers(markers)

    animate_model(model, Qxp, DATA, show_natural_vectors=True)


if __name__ == "__main__":
    main()
