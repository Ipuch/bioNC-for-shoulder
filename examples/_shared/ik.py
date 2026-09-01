"""Inverse-kinematics and joint-angle plotting helpers shared by examples and studies.

Everything that used to be copy-pasted across the ``ik_*`` scripts lives here once:
loading the c3d markers, running the differential IK, turning the optimal natural
coordinates into joint (Euler) angles, the marker RMSE, and a generic joint-angle plot
that labels each axis from the joint's own Euler sequence.
"""

from functools import lru_cache

import ezc3d
import matplotlib.pyplot as plt
import numpy as np
from pyomeca import Markers

from bionc import InverseKinematics, NaturalCoordinates


@lru_cache(maxsize=None)
def c3d_length_factor(c3d_filename: str) -> float:
    """
    Divisor turning this c3d's POINT units into metres: 1000 for millimetres, 1 for metres.

    Mirrors ``bionc.model_creation.c3d_data.C3dData._to_meter``, which is what builds the model
    geometry. Both must agree: some files of the same study are stored in mm (``testFlorent_*.c3d``)
    and others in m (the ``99007140-*`` dataset), and assuming mm on a metre file makes the tracked
    markers 1000x too small -- a ~125 mm marker RMSE instead of ~3 mm.

    Cached: reading one parameter costs a full ``ezc3d`` parse of the file, and ``load_markers``
    asks on every call -- a few hundred times over a leave-one-out sweep.
    """
    units = ezc3d.c3d(c3d_filename)["parameters"]["POINT"]["UNITS"]["value"]
    return 1000.0 if len(units) > 0 and units[0] in ("mm", "millimeter") else 1.0


def load_markers(model, c3d_filename: str, stride: int = 1) -> np.ndarray:
    """
    Technical markers of ``model`` as a ``(3 x nb_technical_markers x nb_frames)`` array in
    metres. Suitable both as IK input and as ``model.Q_from_markers`` input.
    """
    markers = Markers.from_c3d(c3d_filename, usecols=model.marker_names_technical).to_numpy()
    return markers[:3, :, ::stride] / c3d_length_factor(c3d_filename)


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


def rmse_mm(residual_norms) -> float:
    """RMS of per-marker residual norms [m] over everything given, in millimetres."""
    return float(np.sqrt(np.mean(np.asarray(residual_norms) ** 2)) * 1000)


def per_frame_rmse_mm(residual_norms) -> np.ndarray:
    """RMS of per-marker residual norms [m] over the markers of each frame, in millimetres.

    ``residual_norms`` is ``(nb_markers, nb_frames)``; the result is one value per frame.
    """
    return np.sqrt(np.mean(np.asarray(residual_norms) ** 2, axis=0)) * 1000


def marker_rmse_mm(ik: InverseKinematics) -> tuple[float, np.ndarray]:
    """
    Post-optimisation marker RMSE in millimetres.

    Returns ``(global_rmse, per_frame_rmse)`` where the global value is taken over every
    technical marker and every frame.
    """
    residual_norms = ik.sol()["marker_residuals_norm"]  # (nb_markers x nb_frames), in metres
    return rmse_mm(residual_norms), per_frame_rmse_mm(residual_norms)


def marker_groups(model) -> dict[str, np.ndarray]:
    """Indices into ``model.marker_names_technical`` grouped by the segment that carries them."""
    groups, offset = {}, 0
    for name in model.segments.keys():
        count = model.segments[name].nb_markers_technical
        groups[name] = np.arange(offset, offset + count)
        offset += count
    return groups


def solve_trial(model, c3d_filename: str, *, stride: int = 1, method: str = "dik") -> dict:
    """
    Differential IK of ``model`` over one whole trial, with the marker RMSE split by segment.

    Returns ``{markers, Qopt, rmse_mm, per_frame_rmse_mm, rmse_by_group_mm}`` -- lengths in
    millimetres. Used by the leave-one-out sweep to score a fold and by the replay script to
    reconstruct a trial, which is why it lives here rather than in either of them.
    """
    markers = load_markers(model, c3d_filename, stride=stride)
    ik = InverseKinematics(model, markers)
    Qopt = np.asarray(ik.solve(method=method))
    residual_norms = ik.sol()["marker_residuals_norm"]  # (nb_markers, nb_frames), in metres

    return dict(
        markers=markers,
        Qopt=Qopt,
        rmse_mm=rmse_mm(residual_norms),
        per_frame_rmse_mm=per_frame_rmse_mm(residual_norms),
        rmse_by_group_mm={
            name: rmse_mm(residual_norms[index]) for name, index in marker_groups(model).items() if len(index)
        },
    )


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
