"""
Study - IK marker-set comparison (clinical data).

The clinical dataset carries both anatomical bone landmarks and skin-cluster markers. This
study runs the same constrained inverse kinematics three times, changing only which markers
are *technical* (tracked): cluster-only, anatomical-only, or both. It overlays the resulting
joint angles so the sensitivity of the reconstruction to the marker set is visible.

All three runs share the same first-frame initial guess, computed from the anatomical
landmarks (they define the segment axes and are always reconstructible).

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/marker_set_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import matplotlib.pyplot as plt
import numpy as np

from bionc.bionc_numpy.enums import InitialGuessModeType

from examples._shared.ik import euler_axis_labels, joint_angles_over_trial, model_joint_sequences, run_ik
from examples.clinical.model import build_model_constrained, first_frame_guess

DATA = str(Path(__file__).resolve().parents[1] / "examples" / "data" / "testFlorent_clinicalData.c3d")

MARKER_SETS = ("cluster", "anatomical", "both")
COLORS = {"cluster": "tab:blue", "anatomical": "tab:orange", "both": "tab:green"}


def run_marker_set(marker_set: str, q_init_first_frame) -> tuple:
    """Solve the IK for one marker set. Returns (model, joint_angles (3 x nb_joints x nb_frames))."""
    model = build_model_constrained(DATA, marker_set=marker_set)
    ik, Qopt = run_ik(
        model,
        DATA,
        method="dik",
        Q_init=q_init_first_frame,
        initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY,
    )
    print(f"[{marker_set}] global marker residual = {np.max(ik.sol()['total_marker_residuals']):.4e}")
    return model, joint_angles_over_trial(model, Qopt)


def main():
    q_init = first_frame_guess(DATA, marker_set="anatomical")

    models = {}
    joint_angles = {}
    for marker_set in MARKER_SETS:
        models[marker_set], joint_angles[marker_set] = run_marker_set(marker_set, q_init)

    # joint names / Euler sequences are identical across marker sets -> take any model
    sequences = model_joint_sequences(models["both"])
    joint_names = list(sequences)
    axis_labels = {name: euler_axis_labels(seq) for name, seq in sequences.items()}
    nb_joints = len(joint_names)

    nb_frames = next(iter(joint_angles.values())).shape[-1]
    time = np.linspace(0, 1, nb_frames)

    fig, axes = plt.subplots(
        3, nb_joints, figsize=(4.5 * nb_joints, 9), sharex=True, sharey=True, constrained_layout=True, squeeze=False
    )
    fig.suptitle("Upper-limb joint angles depending on the IK marker set (clinical)", fontsize=16, fontweight="bold")

    for row_index in range(3):
        for col_index, joint_name in enumerate(joint_names):
            axis = axes[row_index, col_index]
            for marker_set in MARKER_SETS:
                unwrapped = np.unwrap(joint_angles[marker_set][row_index, col_index, :])
                axis.plot(time, np.degrees(unwrapped), color=COLORS[marker_set], linewidth=2, label=marker_set)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(f"{axis_labels[joint_name][row_index]} rotation (deg)")
            if row_index == 0:
                axis.set_title(f"{joint_name}\nEuler sequence {sequences[joint_name].upper()}", fontsize=11)
            if row_index == 2:
                axis.set_xlabel("Normalized time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, title="IK marker set")
    plt.show()


if __name__ == "__main__":
    main()
