"""
Study - scapulothoracic ellipsoid calibration by all-frames kinematic calibration (clinical data).

Following Naaim (2016/2017), the scapula glides on an ellipsoid carried by the thorax. Here the
ellipsoid is *calibrated* jointly with the motion in a single optimisation:

    minimise   sum_over_frames  1/2 ||model_markers(Q_f) - xp_markers_f||^2
    over        Q_f (natural coordinates of every frame)  and  the ellipsoid parameters p
    subject to  rigid-body + joint (ellipsoid) constraints on every frame,
                (optionally) direct-frame constraints (positive [u, v, w] determinant),

where the shared parameters ``p = (a, b, c, cx, cy, cz)`` are the ellipsoid semi-axes and the
ellipsoid centre location in the thorax. This is an inverse kinematics over ALL frames at once
(one CasADi NLP, IPOPT with exact Hessian), not a frame-by-frame solve wrapped in an outer loop.

The implementation is distilled from ``bionc``'s ``InverseKinematics`` (CasADi backend), keeping
only what is relevant: the per-frame symbolic Q, the marker objective, and the rigid-body / joint
/ direct-frame constraints. The ellipsoid parameters are made symbolic *after* ``model.to_mx()``
by swapping the joint's ``semi_axis_lengths`` and ``ellipsoid_center`` for MX symbols, exactly as:

    model_mx = model.to_mx()
    model_mx.joints["Scapulothoracic"].semi_axis_lengths = (p[0], p[1], p[2])
    model_mx.joints["Scapulothoracic"].ellipsoid_center = NaturalMarker(position=NaturalVector(p[3:6]))

Two ellipsoid joints are supported (heatmaps are irrelevant here and dropped):
  * "tangent" : the scapula plane stays tangent to the ellipsoid (ELLIPSOID_ON_PLANE);
  * "point"   : the centroid of the scapula landmarks (RSAA/RSIA/RSRS, i.e. AA/AI/TS) lies on
                the ellipsoid (POINT_ON_ELLIPSOID).

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/scapulothoracic_ellipsoid_calibration.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import matplotlib.pyplot as plt
import numpy as np

from bionc import NaturalCoordinates as NaturalCoordinatesNumpy
from bionc import SegmentNaturalCoordinates as SegmentNaturalCoordinatesNumpy
from bionc import TransformationMatrixType

from examples._shared.ik import load_markers, marker_rmse_mm, run_ik
from examples.clinical.model import (
    build_ellipsoid_model,
    build_model_free,
    build_model_constrained,
    build_point_on_ellipsoid_model,
    first_frame_guess,
)
from kinematic_calibration import KinematicCalibration, scapulothoracic_angles

DATA = str(Path(__file__).resolve().parents[1] / "examples" / "data" / "testFlorent_clinicalData.c3d")
STRIDE = 5 # frame subsampling: the whole (subsampled) trial is solved together in one NLP
ELLIPSOID_MODELS = ("tangent","point")  # extend to ("tangent", "point") to also calibrate the one-point joint
ELLIPSOID_BUILDERS = {"tangent": build_ellipsoid_model, "point": build_point_on_ellipsoid_model}
ELLIPSOID_LABELS = {"tangent": "tangent ellipsoid", "point": "one-point ellipsoid"}
MODEL_COLORS = {"tangent": "tab:red", "point": "tab:purple"}
ELLIPSOID_RGBA = {"tangent": (220, 70, 70, 90), "point": (150, 70, 200, 90)}  # rerun overlay colors
MARKER_SET = "anatomical"  # one marker set for the whole study (baseline + calibration must match)

def warm_start_theta(model, markers: np.ndarray) -> tuple:
    """
    Data-driven initial ellipsoid ``(a, b, c, cx, cy, cz)`` in thorax segment coordinates.

    Semi-axes ``(a, b, c)``: the mean thorax-origin-to-scapula-origin distance (so the scapula
    starts near the surface).

    Centre, in the thorax segment frame (X = antero-posterior, Y = vertical, Z = medio-lateral):
        * cx = 0                on the mid-sagittal plane (antero-posterior);
        * cy = vertical midpoint of C7 (``CV7``) and the bottom thoracic vertebra (``TV8``);
        * cz = medio-lateral midpoint of the acromioclavicular joint (``RCAJ``, "AC") and the thorax
               origin -> half-way out toward the shoulder; using the AC marker makes the sign follow
               the recorded side (right vs left) automatically.

    Landmark positions are expressed in the thorax frame with the ``Buv`` transform ``B`` via
    ``L = B @ inv([u, v, w]) @ (P - rp_thorax)`` -- the exact inverse of the segment-coordinate
    convention used by ``add_natural_marker_from_segment_coordinates``.
    """
    Q = np.asarray(model.Q_from_markers(markers))
    thorax = model.segments["THORAX"]
    idx_thorax = thorax.index
    idx_scapula = model.segments["RSCAPULA"].index
    B = np.asarray(thorax.compute_transformation_matrix(TransformationMatrixType.Buv), dtype=float)

    names = list(model.marker_names_technical)
    i_ac, i_c7, i_t8 = names.index("RCAJ"), names.index("CV7"), names.index("TV8")

    def to_thorax_frame(point_global, rp_thorax, uvw):
        return B @ np.linalg.inv(uvw) @ (point_global - rp_thorax)

    distances, ac_z, c7_y, t8_y = [], [], [], []
    for k in range(Q.shape[1]):
        Q_k = NaturalCoordinatesNumpy(Q[:, k])
        Qt = SegmentNaturalCoordinatesNumpy(Q_k.vector(idx_thorax))
        rp_thorax = np.asarray(Qt.rp, dtype=float).reshape(3)
        uvw = np.column_stack(
            [np.asarray(Qt.u).reshape(3), np.asarray(Qt.v).reshape(3), np.asarray(Qt.w).reshape(3)]
        )
        rp_scapula = np.asarray(SegmentNaturalCoordinatesNumpy(Q_k.vector(idx_scapula)).rp, dtype=float).reshape(3)

        distances.append(float(np.linalg.norm(rp_scapula - rp_thorax)))
        ac_z.append(to_thorax_frame(markers[:3, i_ac, k], rp_thorax, uvw)[2])  # medio-lateral
        c7_y.append(to_thorax_frame(markers[:3, i_c7, k], rp_thorax, uvw)[1])  # vertical
        t8_y.append(to_thorax_frame(markers[:3, i_t8, k], rp_thorax, uvw)[1])  # vertical

    radius = float(np.mean(distances))
    cx = 0.0
    cy = float((np.mean(c7_y) + np.mean(t8_y)) / 2)  # vertical midpoint C7 / T8
    cz = float(np.mean(ac_z) / 2)  # medio-lateral midpoint AC / thorax origin (origin_z = 0)
    # return (radius, radius, radius, cx, cy, cz)
    return (0.082997500000000002, 0.199991, 0.083001000000000005, cx, cy, cz)

def run_free_baseline():
    """FREE-scapulothoracic baseline IK (frame per frame). Returns ``(model, Qopt)``."""
    from bionc.bionc_numpy.enums import InitialGuessModeType

    # baseline = build_model_constrained(DATA, marker_set=MARKER_SET)
    baseline = build_model_free(DATA, marker_set=MARKER_SET)
    q_init = first_frame_guess(DATA, marker_set=MARKER_SET)
    ik, Qopt = run_ik(
        baseline,
        DATA,
        method="dik",
        stride=STRIDE,
        Q_init=q_init,
        initial_guess_mode=InitialGuessModeType.USER_PROVIDED_FIRST_FRAME_ONLY,
    )
    return baseline, Qopt, ik


def calibrate(ellipsoid_model: str, base_model, markers: np.ndarray, marker_set: str) -> dict:
    """Warm-start, build the requested ellipsoid model, and run the all-frames calibration."""
    theta0 = warm_start_theta(base_model, markers)
    print(f"[{ellipsoid_model}] warm start theta0 = {np.array2string(np.array(theta0), precision=4)}")

    model = ELLIPSOID_BUILDERS[ellipsoid_model](DATA, theta0, marker_set=marker_set)
    calibration = KinematicCalibration(model, markers, active_direct_frame_constraints=True)
    Qopt = calibration.solve()
    out = calibration.sol()

    print(f"[{ellipsoid_model}] success={out['success']}")
    print(f"[{ellipsoid_model}] semi-axes [m]      = {np.array2string(out['semi_axes'], precision=4)}")
    print(f"[{ellipsoid_model}] centre (natural)   = {np.array2string(out['ellipsoid_center_natural'], precision=4)}")
    print(f"[{ellipsoid_model}] marker RMSE        = {out['marker_rmse_mm']:.2f} mm")
    print(f"[{ellipsoid_model}] max joint residual = {np.max(out['max_joint_residual_per_frame']):.3e}")
    return dict(model=model, Qopt=Qopt, out=out)


def _rotation_to_xyzw(rotation: np.ndarray) -> list:
    """3x3 rotation matrix -> quaternion [x, y, z, w] (Shepperd's method)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        x, y, z, w = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        x, y, z, w = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        x, y, z, w = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        x, y, z, w = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def _axes_to_xyzw(axes: np.ndarray) -> list:
    """Nearest rotation (SVD) to the (possibly slightly non-orthogonal) axis columns, as xyzw."""
    u, _, vt = np.linalg.svd(axes)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rotation = u @ vt
    return _rotation_to_xyzw(rotation)


def _plane_triangle(center: np.ndarray, normal: np.ndarray, size: float = 0.12) -> list:
    """Three vertices of an equilateral triangle centred on ``center``, lying in the plane ``normal``."""
    n = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return [
        (center + size * e1).tolist(),
        (center + size * (-0.5 * e1 + 0.8660254 * e2)).tolist(),
        (center + size * (-0.5 * e1 - 0.8660254 * e2)).tolist(),
    ]


def _overlay_ellipsoids(ellipsoids, t_span: np.ndarray) -> None:
    """
    Log each calibrated ellipsoid, its scapula contact point, and (for the tangent joint) the
    scapula plane, moving with the segments on the pyorerun ``stable_time`` timeline. The ellipsoid
    rides on the thorax; the contact point and plane ride on the scapula.
    """
    import rerun as rr

    from bionc.bionc_numpy.natural_vector import NaturalVector

    for entity, model, Qopt, theta, color in ellipsoids:
        solid_color = (color[0], color[1], color[2], 255)  # opaque marker vs translucent surfaces

        radii = np.asarray(theta[:3], dtype=float)
        center_interp = np.asarray(NaturalVector(np.asarray(theta[3:6], dtype=float)).interpolate(), dtype=float)
        joint = model.joints["Scapulothoracic"]
        axes_interp = [np.asarray(axis.interpolation_matrix, dtype=float) for axis in joint.ellipsoid_axes]
        thorax_slice = slice(12 * model.segments["THORAX"].index, 12 * model.segments["THORAX"].index + 12)
        scapula_slice = slice(12 * model.segments["RSCAPULA"].index, 12 * model.segments["RSCAPULA"].index + 12)

        # the tangent joint carries a plane (point + normal); the one-point joint only a contact point
        plane_point = getattr(joint, "plane_point", None)
        if plane_point is not None:
            contact_interp = np.asarray(plane_point.interpolation_matrix, dtype=float)
            normal_interp = np.asarray(joint.plane_normal.interpolation_matrix, dtype=float)
        else:
            contact_interp = np.asarray(joint.contact_point.interpolation_matrix, dtype=float)
            normal_interp = None

        for k, t in enumerate(t_span):
            q_thorax = np.asarray(Qopt[thorax_slice, k], dtype=float).reshape(-1)
            q_scapula = np.asarray(Qopt[scapula_slice, k], dtype=float).reshape(-1)
            rr.set_time("stable_time", duration=float(t))

            center = center_interp @ q_thorax
            axes = np.column_stack([interp @ q_thorax for interp in axes_interp])
            axes /= np.linalg.norm(axes, axis=0, keepdims=True)
            rr.log(
                entity,
                rr.Ellipsoids3D(
                    half_sizes=[radii.tolist()],
                    centers=[center.tolist()],
                    quaternions=[rr.Quaternion(xyzw=_axes_to_xyzw(axes))],
                    colors=[color],
                    fill_mode="MajorWireframe",
                ),
            )

            contact = contact_interp @ q_scapula
            rr.log(f"{entity}/contact_point", rr.Points3D([contact.tolist()], colors=[solid_color], radii=[0.01]))

            if normal_interp is not None:
                rr.log(
                    f"{entity}/plane",
                    rr.Mesh3D(
                        vertex_positions=_plane_triangle(contact, normal_interp @ q_scapula),
                        triangle_indices=[[0, 1, 2]],
                        vertex_colors=[color, color, color],
                    ),
                )


def visualize(named_models: dict, markers: np.ndarray, marker_names, ellipsoids=()) -> None:
    """
    pyorerun window overlaying every post-calibration reconstruction (and the FREE baseline)
    on the same experimental markers, plus the calibrated ellipsoid trajectories. Each model gets
    a distinct name so rerun shows them as separate, individually toggleable entities.
    """
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun, PyoMarkers

    class _NamedBioncModel(BioncModelNoMesh):
        """BioncModelNoMesh with a per-instance name (the base class hard-codes a single name)."""

        def __init__(self, model, display_name, options=None):
            super().__init__(model, options)
            self._display_name = display_name

        @property
        def name(self):
            return self._display_name

    t_span = np.linspace(0, 1, markers.shape[2])
    prr = PhaseRerun(t_span=t_span)
    for display_name, (model, Q) in named_models.items():
        prr.add_animated_model(_NamedBioncModel(model, display_name), np.asarray(Q))

    pyomarkers = PyoMarkers(data=markers, marker_names=list(marker_names))
    pyomarkers.show_labels = False
    prr.add_xp_markers("experimental_markers", pyomarkers)
    prr.rerun()

    # rerun recording is now live -> overlay the calibrated ellipsoids on the same timeline
    _overlay_ellipsoids(ellipsoids, t_span)


def plot_residuals(results: dict, baseline_marker_rmse: np.ndarray, time: np.ndarray) -> None:
    """
    Residuals over the whole trial: per-frame marker reconstruction RMSE (left) and per-frame
    max joint-constraint residual (right, i.e. how well the ellipsoid constraint is satisfied).
    """
    fig, (ax_marker, ax_joint) = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    fig.suptitle("Residuals over the trial", fontsize=14, fontweight="bold")

    ax_marker.plot(time, baseline_marker_rmse, color="tab:gray", lw=2, label="FREE baseline")
    for model in ELLIPSOID_MODELS:
        ax_marker.plot(
            time,
            results[model]["out"]["per_frame_marker_rmse_mm"],
            color=MODEL_COLORS[model],
            lw=2,
            label=ELLIPSOID_LABELS[model],
        )
    ax_marker.set_title("Marker reconstruction RMSE")
    ax_marker.set_xlabel("Normalized time")
    ax_marker.set_ylabel("marker RMSE (mm)")
    ax_marker.grid(True, alpha=0.25)
    ax_marker.legend()

    for model in ELLIPSOID_MODELS:
        ax_joint.plot(
            time,
            results[model]["out"]["max_joint_residual_per_frame"],
            color=MODEL_COLORS[model],
            lw=2,
            label=ELLIPSOID_LABELS[model],
        )
    ax_joint.set_title("Max joint-constraint residual")
    ax_joint.set_xlabel("Normalized time")
    ax_joint.set_ylabel("residual")
    ax_joint.set_yscale("log")
    ax_joint.grid(True, which="both", alpha=0.25)
    ax_joint.legend()


def main():
    marker_set = MARKER_SET
    base_model = build_model_constrained(DATA, marker_set=marker_set)
    markers = load_markers(base_model, DATA, stride=STRIDE)

    # before optimisation: raw marker reconstruction (Q_from_markers), no constraint enforced
    before_angles = scapulothoracic_angles(base_model, np.asarray(base_model.Q_from_markers(markers)))

    baseline_model, baseline_Q, baseline_ik = run_free_baseline()
    baseline_angles = scapulothoracic_angles(baseline_model, baseline_Q)
    _, baseline_marker_rmse = marker_rmse_mm(baseline_ik)  # per-frame RMSE [mm] over the trial

    results = {model: calibrate(model, base_model, markers, marker_set=marker_set) for model in ELLIPSOID_MODELS}

    time = np.linspace(0, 1, markers.shape[2])

    # --- scapulothoracic angle curves (all in the YXZ basis) ---
    labels = ["First (Y)", "Second (X)", "last (Z)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle("Scapulothoracic angles: before optim vs FREE baseline vs calibrated ellipsoid(s)", fontsize=14, fontweight="bold")
    for i, axis in enumerate(axes):
        axis.plot(time, before_angles[i], color="black", lw=1.5, ls="--", label="before optim (raw markers)")
        axis.plot(time, baseline_angles[i], color="tab:gray", lw=2, label="FREE baseline")
        for model in ELLIPSOID_MODELS:
            axis.plot(time, results[model]["out"]["scapulothoracic_angles_deg"][i], color=MODEL_COLORS[model], lw=2, label=ELLIPSOID_LABELS[model])
        axis.set_title(labels[i])
        axis.set_xlabel("Normalized time")
        axis.grid(True, alpha=0.25)
        if i == 0:
            axis.set_ylabel("angle (deg)")
            axis.legend()

    # --- residuals over the whole trial ---
    plot_residuals(results, baseline_marker_rmse, time)

    # --- pyorerun comparison of every post-calibration reconstruction (+ calibrated ellipsoids) ---
    named_models = {"FREE baseline": (baseline_model, baseline_Q)}
    ellipsoids = []
    for model in ELLIPSOID_MODELS:
        named_models[ELLIPSOID_LABELS[model]] = (results[model]["model"], results[model]["Qopt"])
        ellipsoids.append(
            (
                f"scapulothoracic_ellipsoid/{model}",
                results[model]["model"],
                results[model]["Qopt"],
                results[model]["out"]["theta"],
                ELLIPSOID_RGBA[model],
            )
        )
    visualize(named_models, markers, base_model.marker_names_technical, ellipsoids=ellipsoids)

    plt.show()


if __name__ == "__main__":
    main()
