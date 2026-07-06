"""
Clinical tutorial - step 3: run the inverse kinematics.

We take the constrained model of step 2 and solve the inverse kinematics: the optimiser now
*enforces* the joint constraints while staying close to the experimental markers. We report
the marker RMSE, animate the reconstruction, and plot the joint angles (each joint decomposed
in its own Euler sequence).

Run (from the repo root, inside the ``bionc`` conda env):
    python examples/clinical/03_inverse_kinematics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root on the path

import matplotlib.pyplot as plt

from examples._shared.ik import joint_angles_over_trial, marker_rmse_mm, plot_joint_angles, run_ik
from examples._shared.viz import animate_model
from examples.clinical.model import build_model_constrained

DATA = str(Path(__file__).resolve().parents[1] / "data" / "testFlorent_clinicalData.c3d")


def main():
    model = build_model_constrained(DATA, marker_set="both")

    ik, Qopt = run_ik(model, DATA, method="dik")
    global_rmse, _ = marker_rmse_mm(ik)
    print(f"Clinical IK - global marker RMSE = {global_rmse:.3f} mm")

    animate_model(model, Qopt, DATA)

    angles = joint_angles_over_trial(model, Qopt)
    plot_joint_angles(model, angles, title="Clinical - inverse kinematics joint angles")
    plt.show()


if __name__ == "__main__":
    main()
