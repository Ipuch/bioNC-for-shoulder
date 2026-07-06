"""Inverse-kinematics and joint-angle plotting helpers shared by examples and studies.

Everything that used to be copy-pasted across the ``ik_*`` scripts lives here once:
loading the c3d markers, running the differential IK, turning the optimal natural
coordinates into joint (Euler) angles, the marker RMSE, and a generic joint-angle plot
that labels each axis from the joint's own Euler sequence.
"""

import matplotlib.pyplot as plt
import numpy as np
from pyomeca import Markers

from bionc import InverseKinematics, NaturalCoordinates


def load_markers(model, c3d_filename: str, stride: int = 1) -> np.ndarray:
    """
    Technical markers of ``model`` as a ``(3 x nb_technical_markers x nb_frames)`` array in
    metres. Suitable both as IK input and as ``model.Q_from_markers`` input.
    """
    return Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()[:3, :, ::stride] / 1000


def run_ik(model, c3d_filename: str, *, method: str = "dik", stride: int = 1, **solve_kwargs):
    """
    Solve the inverse kinematics of ``model`` on ``c3d_filename``.

    Returns ``(ik, Qopt)`` where ``ik`` is the :class:`InverseKinematics` instance (call
    ``ik.sol()`` for residuals) and ``Qopt`` the optimal natural coordinates.
    """
    markers = load_markers(model, c3d_filename, stride=stride)
    ik = InverseKinematics(model, markers)
    Qopt = ik.solve(method=method, **solve_kwargs)
    return ik, Qopt


def joint_angles_over_trial(model, Qopt: np.ndarray) -> np.ndarray:
    """Joint Euler angles for the whole trial, ``(3 x nb_joints x nb_frames)`` in radians."""
    Qopt = np.asarray(Qopt)
    nb_frames = Qopt.shape[-1]
    shape = model.natural_coordinates_to_joint_angles(NaturalCoordinates(Qopt[:, 0])).shape
    angles = np.zeros((shape[0], shape[1], nb_frames))
    for i in range(nb_frames):
        angles[:, :, i] = model.natural_coordinates_to_joint_angles(NaturalCoordinates(Qopt[:, i]))
    return angles


def marker_rmse_mm(ik: InverseKinematics) -> tuple[float, np.ndarray]:
    """
    Post-optimisation marker RMSE in millimetres.

    Returns ``(global_rmse, per_frame_rmse)`` where the global value is taken over every
    technical marker and every frame.
    """
    residuals_norm = ik.sol()["marker_residuals_norm"]  # (nb_markers x nb_frames), in metres
    global_rmse = float(np.sqrt(np.mean(residuals_norm**2)) * 1000)
    per_frame_rmse = np.sqrt(np.mean(residuals_norm**2, axis=0)) * 1000
    return global_rmse, per_frame_rmse


def euler_axis_labels(sequence: str) -> list[str]:
    """
    Turn an Euler sequence string (e.g. "xyz", "yxy") into the ordered rotation-axis
    labels of the decomposition, disambiguating repeated axes (e.g. "yxy" -> Y1, X, Y2).
    """
    sequence = sequence.upper()
    totals = {axis: sequence.count(axis) for axis in set(sequence)}
    seen: dict[str, int] = {}
    labels = []
    for axis in sequence:
        seen[axis] = seen.get(axis, 0) + 1
        labels.append(f"{axis}{seen[axis]}" if totals[axis] > 1 else axis)
    return labels


def model_joint_sequences(model) -> dict[str, str]:
    """Map each joint name to its projection-basis Euler sequence string (e.g. "xyz")."""
    return {name: joint.projection_basis.value for name, joint in model.joints.items()}


def plot_joint_angles(model, angles: np.ndarray, *, title: str = "Joint angles", color: str = "tab:blue"):
    """
    Plot one model's joint angles on a ``3 x nb_joints`` grid.

    Each column is a joint, each row one axis of that joint's own Euler decomposition (so
    the y-labels read e.g. ``Y1 / X / Y2`` for a YXY glenohumeral joint). Angles are
    unwrapped and shown in degrees, with a single shared y-scale.
    """
    sequences = model_joint_sequences(model)
    joint_names = list(sequences)
    axis_labels = {name: euler_axis_labels(seq) for name, seq in sequences.items()}
    nb_joints = len(joint_names)
    time = np.linspace(0, 1, angles.shape[-1])

    fig, axes = plt.subplots(
        3, nb_joints, figsize=(4.5 * nb_joints, 9), sharex=True, sharey=True, constrained_layout=True, squeeze=False
    )
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for row_index in range(3):
        for col_index, joint_name in enumerate(joint_names):
            axis = axes[row_index, col_index]
            axis.plot(time, np.degrees(np.unwrap(angles[row_index, col_index, :])), color=color, linewidth=2)
            axis.grid(True, alpha=0.25)
            axis.set_ylabel(f"{axis_labels[joint_name][row_index]} rotation (deg)")
            if row_index == 0:
                axis.set_title(f"{joint_name}\nEuler sequence {sequences[joint_name].upper()}", fontsize=11)
            if row_index == 2:
                axis.set_xlabel("Normalized time")
    return fig
