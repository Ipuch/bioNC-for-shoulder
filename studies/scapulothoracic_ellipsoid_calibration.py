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

from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np

from examples._shared.ik import load_markers, marker_rmse_mm, run_ik
from examples._shared.viz import named_bionc_model, overlay_ellipsoids
from examples.clinical.model import (
    build_ellipsoid_model,
    build_model_free,
    build_model_constrained,
    build_point_on_ellipsoid_model,
    first_frame_guess,
)
from studies.kinematic_calibration import EllipsoidSemiAxes, KinematicCalibration, MarkerPosition, scapulothoracic_angles
from studies.shoulder_calibration import (
    PARAMETER_PRIOR,
    contact_point_cloud,
    ellipsoid_bounds,
    fit_ellipsoid,
    thorax_reference,
)

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

    Express the scapula contact point in the thorax frame over the whole trial, then fit an
    ellipsoid to that cloud with :func:`~studies.shoulder_calibration.fit_ellipsoid`, sized against
    the subject's own thorax. See that module for why the fit needs a prior at all: the scapula
    sweeps too small a patch for its curvature to pin down a radius.
    """
    cloud = contact_point_cloud(model, markers)
    reference = thorax_reference(model, markers)
    fit = fit_ellipsoid(cloud, reference, bounds=ellipsoid_bounds(reference))
    print(f"warm start surface RMS = {fit['residual_mm']:.2f} mm, at bounds: {fit['at_bounds'] or 'none'}")
    return (*fit["semi_axes"], *fit["center"])


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

    reference = thorax_reference(base_model, markers)
    bounds = ellipsoid_bounds(reference)
    model = ELLIPSOID_BUILDERS[ellipsoid_model](DATA, theta0, marker_set=marker_set)
    calibration = KinematicCalibration(
        model,
        markers,
        parameters=[
            EllipsoidSemiAxes("Scapulothoracic", bounds=bounds["semi_axes"], prior=1.0),
            MarkerPosition(
                "THORAX",
                "ELLIPSOID_CENTER",
                targets=(("Scapulothoracic", "ellipsoid_center"),),
                half_range=bounds["center_half_range"],
                box_center=reference["center"],
                prior=1.0,
            ),
        ],
        active_direct_frame_constraints=True,
        # without it the semi-axes ride their bounds: a contact patch does not determine a radius
        regularization=PARAMETER_PRIOR * markers.shape[2],
    )
    Qopt = calibration.solve()
    out = calibration.sol()

    print(f"[{ellipsoid_model}] success={out['success']}")
    print(f"[{ellipsoid_model}] semi-axes [mm]     = {np.array2string(out['semi_axes'] * 1000, precision=1)}")
    print(f"[{ellipsoid_model}] centre (segment mm)= {np.array2string(out['ellipsoid_center_scs'] * 1000, precision=1)}")
    print(f"[{ellipsoid_model}] marker RMSE        = {out['marker_rmse_mm']:.2f} mm")
    print(f"[{ellipsoid_model}] max joint residual = {np.max(out['max_joint_residual_per_frame']):.3e}")
    print(f"[{ellipsoid_model}] parameters at bounds = {out['parameters_at_bounds'] or 'none'}")
    return dict(model=model, Qopt=Qopt, out=out)


def visualize(named_models: dict, markers: np.ndarray, marker_names, ellipsoids=()) -> None:
    """
    pyorerun window overlaying every post-calibration reconstruction (and the FREE baseline)
    on the same experimental markers, plus the calibrated ellipsoid trajectories. Each model gets
    a distinct name so rerun shows them as separate, individually toggleable entities.
    """
    from pyorerun import PhaseRerun, PyoMarkers

    t_span = np.linspace(0, 1, markers.shape[2])
    prr = PhaseRerun(t_span=t_span)
    for display_name, (model, Q) in named_models.items():
        prr.add_animated_model(named_bionc_model(model, display_name), np.asarray(Q))

    pyomarkers = PyoMarkers(data=markers, marker_names=list(marker_names))
    pyomarkers.show_labels = False
    prr.add_xp_markers("experimental_markers", pyomarkers)
    prr.rerun()

    # rerun recording is now live -> overlay the calibrated ellipsoids on the same timeline
    overlay_ellipsoids(ellipsoids, t_span)


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
                results[model]["out"]["semi_axes"],
                results[model]["out"]["ellipsoid_center_natural"],
                ELLIPSOID_RGBA[model],
            )
        )
    visualize(named_models, markers, base_model.marker_names_technical, ellipsoids=ellipsoids)

    plt.show()


if __name__ == "__main__":
    main()
