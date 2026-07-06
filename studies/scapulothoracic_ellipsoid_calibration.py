"""
Study - scapulothoracic tangent-ellipsoid calibration (clinical data).

Following Naaim (2016/2017), the scapula is assumed to glide on an ellipsoid carried by the
thorax, its plane staying tangent to that ellipsoid (``JointType.ELLIPSOID_ON_PLANE``, one
holonomic constraint, no penetration by definition). We *calibrate* that ellipsoid from the
subject's motion with a nested (bilevel) scheme:

  * inner level : differential inverse kinematics (``method="dik"``) reconstructs the whole
                  trial with the ellipsoid constraint active;
  * outer level : a local least-squares optimizer (``scipy.optimize.least_squares``) varies the
                  six ellipsoid parameters ``theta = (a, b, c, cx, cy, cz)`` -- the three
                  semi-axis lengths [m] and the ellipsoid centre in the thorax frame -- to
                  minimize the scapula marker reconstruction error.

The outer loop is warm-started from a data-driven ellipsoid (centre at the thorax origin,
isotropic radius = mean thorax-origin-to-scapula distance) and runs on a subsampled set of
frames for speed; the final reconstruction and the reported angles use every frame.

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/scapulothoracic_ellipsoid_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from bionc import NaturalCoordinates, SegmentNaturalCoordinates, TransformationMatrixType
from bionc.bionc_numpy.enums import InitialGuessModeType

from examples._shared.ik import load_markers, run_ik
from examples.clinical.model import build_ellipsoid_model, build_model_constrained, first_frame_guess

C3D_FILENAME = str(Path(__file__).resolve().parents[1] / "examples" / "data" / "testFlorent_clinicalData.c3d")
_FIRST_FRAME = InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY
SCAPULA_MARKERS = ("RSAA", "RSIA", "RSRS", "RCAJ")  # calibration signal (scapula anatomical bone landmarks)
OPTIM_STRIDE = 12  # frame subsampling during the outer optimization (speed)


def warm_start() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Data-driven initial ellipsoid and parameter bounds.

    The centre is placed at the thorax origin (``location = 0`` in the thorax frame) and the
    three semi-axes are initialised to the mean thorax-origin-to-scapula distance, so the scapula
    sits on the ellipsoid surface at the start. ``theta = (a, b, c, cx, cy, cz)``.
    """
    model = build_model_constrained(C3D_FILENAME)
    markers = load_markers(model, C3D_FILENAME)
    Q = np.asarray(model.Q_from_markers(markers))
    idx_t = model.segments["THORAX"].index
    idx_s = model.segments["RSCAPULA"].index
    B = model.segments["THORAX"].compute_transformation_matrix(TransformationMatrixType.Buv)

    locs = np.zeros((3, Q.shape[1]))
    for k in range(Q.shape[1]):
        Qk = NaturalCoordinates(Q[:, k])
        Qt = SegmentNaturalCoordinates(Qk.vector(idx_t))
        Qs = SegmentNaturalCoordinates(Qk.vector(idx_s))
        uvw = np.column_stack([Qt.u, Qt.v, Qt.w])
        # scapula proximal point expressed in the thorax 'location' frame (exact inverse of the
        # convention used by add_natural_marker_from_segment_coordinates)
        locs[:, k] = B @ np.linalg.inv(uvw) @ (Qs.rp - Qt.rp)

    radius = float(np.linalg.norm(locs.mean(axis=1)))
    theta0 = np.array([radius, radius, radius, 0.0, 0.0, 0.0])
    lower = np.array([0.05, 0.05, 0.05, -0.15, -0.15, -0.15])
    upper = np.array([0.40, 0.40, 0.40, 0.15, 0.15, 0.15])
    return theta0, (lower, upper)


def scapula_marker_indices(model) -> list[int]:
    names = model.marker_names_technical
    return [i for i, name in enumerate(names) if name in SCAPULA_MARKERS]


def make_residual(q_init, stride: int):
    """Return f(theta) -> flat scapula marker residuals [m], for least_squares."""

    def residual(theta):
        model = build_ellipsoid_model(C3D_FILENAME, theta)
        ik, _ = run_ik(model, C3D_FILENAME, method="dik", stride=stride, Q_init=q_init, initial_guess_mode=_FIRST_FRAME)
        rn = np.asarray(ik.sol()["marker_residuals_norm"])  # (n_markers x n_frames), per-marker distance
        return rn[scapula_marker_indices(model), :].ravel()

    return residual


def scapulothoracic_angles(model, Qopt) -> np.ndarray:
    """Euler angles [3 x n_frames] of the scapulothoracic joint, degrees."""
    joint_order = list(model.joints.joint_names)
    col = joint_order.index("Scapulothoracic")
    angles = np.zeros((3, Qopt.shape[-1]))
    for k in range(Qopt.shape[-1]):
        angles[:, k] = model.natural_coordinates_to_joint_angles(NaturalCoordinates(Qopt[:, k]))[:, col]
    return np.degrees(angles)


def main():
    q_init = first_frame_guess(C3D_FILENAME)

    # --- baseline (FREE scapulothoracic joint), full resolution, for comparison ---
    baseline_model = build_model_constrained(C3D_FILENAME)
    baseline_ik, baseline_Q = run_ik(
        baseline_model, C3D_FILENAME, method="dik", Q_init=q_init, initial_guess_mode=_FIRST_FRAME
    )
    baseline_angles = scapulothoracic_angles(baseline_model, baseline_Q)

    # --- outer optimization of the ellipsoid parameters (subsampled frames) ---
    theta0, (lower, upper) = warm_start()
    print(f"warm start theta0 = {np.array2string(theta0, precision=4)}")
    residual = make_residual(q_init, stride=OPTIM_STRIDE)
    r0 = residual(theta0)
    print(f"initial scapula RMSE = {1e3 * np.sqrt(np.mean(r0 ** 2)):.2f} mm ({r0.size} residuals)")

    result = least_squares(
        residual, theta0, bounds=(lower, upper), diff_step=1e-3, x_scale="jac", verbose=2, max_nfev=200
    )
    theta_star = result.x
    print(f"\ncalibrated theta* = {np.array2string(theta_star, precision=4)}")
    print(f"outer status={result.status}, cost={result.cost:.6e}, nfev={result.nfev}")
    print(f"final (subsampled) scapula RMSE = {1e3 * np.sqrt(np.mean(result.fun ** 2)):.2f} mm")

    # --- final full-resolution reconstruction with the calibrated ellipsoid ---
    model = build_ellipsoid_model(C3D_FILENAME, theta_star)
    ik, Qopt = run_ik(model, C3D_FILENAME, method="dik", Q_init=q_init, initial_guess_mode=_FIRST_FRAME)
    sol = ik.sol()
    scap = scapula_marker_indices(model)
    scap_rmse_mm = 1e3 * np.sqrt(np.mean(np.asarray(sol["marker_residuals_norm"])[scap, :] ** 2))
    base_scap = scapula_marker_indices(baseline_model)
    base_rmse_mm = 1e3 * np.sqrt(np.mean(np.asarray(baseline_ik.sol()["marker_residuals_norm"])[base_scap, :] ** 2))
    print(f"\nscapula marker RMSE [mm]: FREE baseline = {base_rmse_mm:.2f}, calibrated ellipsoid = {scap_rmse_mm:.2f}")
    print(f"max tangency (joint) residual = {np.max(sol['total_joint_residuals']):.3e}")

    calibrated_angles = scapulothoracic_angles(model, Qopt)

    # --- plot scapulothoracic angles: calibrated vs FREE baseline ---
    labels = ["First (X)", "Second (Y)", "last (Z)"]
    time = np.linspace(0, 1, calibrated_angles.shape[-1])
    time_base = np.linspace(0, 1, baseline_angles.shape[-1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Scapulothoracic angles: FREE baseline vs calibrated tangent ellipsoid", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes):
        ax.plot(time_base, baseline_angles[i], color="tab:gray", lw=2, label="FREE baseline")
        ax.plot(time, calibrated_angles[i], color="tab:red", lw=2, label="ellipsoid (calibrated)")
        ax.set_title(labels[i])
        ax.set_xlabel("Normalized time")
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.set_ylabel("angle (deg)")
            ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
