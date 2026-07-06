"""
Study - glenohumeral constraint comparison (Henninger data).

Every joint of the chain is left FREE (the segments are only tied by their shared markers)
*except* the glenohumeral (GH) joint, whose constraint is the variable under study. Three GH
models are compared:

    * "free"           : GH is also FREE -> raw data, no constraint anywhere; best-achievable
                         marker fit (lower bound of the RMSE).
    * "spherical"      : GH is a SPHERICAL joint (GSC and GSChum made coincident, ball-and-socket).
    * "constant_length": GH is a CONSTANT_LENGTH joint (sphere-on-sphere: constant GSC<->GSChum).

We report the post-optimisation marker RMSE, overlay the joint angles of the three models, and
plot how much each constraint deflects the angles relative to the raw (free) reference.

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/gh_constraint_comparison.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import matplotlib.pyplot as plt
import numpy as np

from examples._shared.ik import (
    euler_axis_labels,
    joint_angles_over_trial,
    marker_rmse_mm,
    model_joint_sequences,
    run_ik,
)
from examples.henninger.model import build_model

DATA = str(Path(__file__).resolve().parents[1] / "examples" / "data" / "testFlorent_HenningerData.c3d")

GH_CONFIGS = ("free", "spherical", "constant_length")
GH_LABELS = {"free": "FREE (raw)", "spherical": "GH spherical", "constant_length": "GH constant length"}
COLORS = {"free": "tab:gray", "spherical": "tab:blue", "constant_length": "tab:red"}
JOINT_NAMES = ["Freeflyer", "Scapulothoracic", "Glenohumeral"]


def run_config(gh_config: str) -> dict:
    """Build the all-FREE-except-GH model, solve the IK and gather the metrics."""
    model = build_model(DATA, clavicle_constraint=False, glenohumeral=gh_config)
    ik, Qopt = run_ik(model, DATA, method="dik")
    global_rmse, per_frame_rmse = marker_rmse_mm(ik)
    return dict(
        model=model,
        Qopt=Qopt,
        global_rmse=global_rmse,
        per_frame_rmse=per_frame_rmse,
        joint_angles=joint_angles_over_trial(model, Qopt),
    )


def _axis_metadata(results: dict):
    sequences = model_joint_sequences(results["free"]["model"])
    sequences = {name: sequences[name] for name in JOINT_NAMES}
    labels = {name: euler_axis_labels(seq) for name, seq in sequences.items()}
    return sequences, labels


def plot_joint_angles(results: dict) -> None:
    """Overlay the joint angles (Euler rotations) of the three GH configurations."""
    nb_frames = results["free"]["Qopt"].shape[-1]
    time = np.linspace(0, 1, nb_frames)
    sequences, axis_labels = _axis_metadata(results)

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Upper-limb joint angles - GH constraint comparison", fontsize=16, fontweight="bold")

    for row_index in range(3):
        for col_index, joint_name in enumerate(JOINT_NAMES):
            axis = axes[row_index, col_index]
            for gh_config in GH_CONFIGS:
                unwrapped = np.unwrap(results[gh_config]["joint_angles"][row_index, col_index, :])
                axis.plot(time, np.degrees(unwrapped), color=COLORS[gh_config], linewidth=2, label=GH_LABELS[gh_config])
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(f"{axis_labels[joint_name][row_index]} rotation (deg)")
            if row_index == 0:
                axis.set_title(f"{joint_name}\nEuler sequence {sequences[joint_name].upper()}", fontsize=11)
            if row_index == 2:
                axis.set_xlabel("Normalized time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, title="GH constraint")


def plot_joint_angle_differences(results: dict) -> None:
    """Joint-angle difference of each constrained GH model w.r.t. the "free" (raw) reference."""
    nb_frames = results["free"]["Qopt"].shape[-1]
    time = np.linspace(0, 1, nb_frames)
    sequences, axis_labels = _axis_metadata(results)
    constrained_configs = [c for c in GH_CONFIGS if c != "free"]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle("Joint-angle difference vs. FREE (raw) reference", fontsize=16, fontweight="bold")

    reference = results["free"]["joint_angles"]
    for row_index in range(3):
        for col_index, joint_name in enumerate(JOINT_NAMES):
            axis = axes[row_index, col_index]
            reference_series = np.unwrap(reference[row_index, col_index, :])
            for gh_config in constrained_configs:
                unwrapped = np.unwrap(results[gh_config]["joint_angles"][row_index, col_index, :])
                axis.plot(
                    time,
                    np.degrees(unwrapped - reference_series),
                    color=COLORS[gh_config],
                    linewidth=2,
                    label=GH_LABELS[gh_config],
                )
            axis.axhline(0.0, color="tab:gray", linewidth=1, linestyle="--", alpha=0.7)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(f"{axis_labels[joint_name][row_index]} diff (deg)")
            if row_index == 0:
                axis.set_title(f"{joint_name}\nEuler sequence {sequences[joint_name].upper()}", fontsize=11)
            if row_index == 2:
                axis.set_xlabel("Normalized time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, title="GH constraint (vs. raw)")


def plot_rmse(results: dict) -> None:
    """Bar plot of the global marker RMSE and per-frame RMSE curves."""
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Post-optimisation marker RMSE", fontsize=14, fontweight="bold")

    labels = [GH_LABELS[c] for c in GH_CONFIGS]
    global_rmse = [results[c]["global_rmse"] for c in GH_CONFIGS]
    ax_bar.bar(labels, global_rmse, color=[COLORS[c] for c in GH_CONFIGS])
    ax_bar.set_ylabel("Global RMSE (mm)")
    ax_bar.set_title("Global marker RMSE")
    ax_bar.grid(True, axis="y", alpha=0.25)
    for i, value in enumerate(global_rmse):
        ax_bar.text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)

    time = np.linspace(0, 1, results["free"]["Qopt"].shape[-1])
    for gh_config in GH_CONFIGS:
        ax_curve.plot(
            time, results[gh_config]["per_frame_rmse"], color=COLORS[gh_config], linewidth=2, label=GH_LABELS[gh_config]
        )
    ax_curve.set_xlabel("Normalized time")
    ax_curve.set_ylabel("RMSE (mm)")
    ax_curve.set_title("Per-frame marker RMSE")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend()


def main():
    results = {}
    for gh_config in GH_CONFIGS:
        print(f"\n=== Solving IK with GH = {GH_LABELS[gh_config]} ===")
        results[gh_config] = run_config(gh_config)

    print("\n===== Global marker RMSE (mm) =====")
    for gh_config in GH_CONFIGS:
        print(f"  {GH_LABELS[gh_config]:<22s}: {results[gh_config]['global_rmse']:.3f} mm")

    plot_joint_angles(results)
    plot_joint_angle_differences(results)
    plot_rmse(results)
    plt.show()


if __name__ == "__main__":
    main()
