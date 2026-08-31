"""
What each of the three calibration steps actually did.

One figure per step, plus a summary of what closing the loop cost:

* **step 1 — the ellipsoid.** The contact-point cloud in the thorax frame with the calibrated
  ellipsoid drawn through it, in three orthogonal sections, and the distribution of signed
  point-to-surface distances before (warm start) and after. The cross-sections are the honest way to
  look at this fit: the ellipsoid is huge and the data covers a small patch of it.
* **step 2 — the joint centres.** The glenohumeral centres are calibrated so that one point of the
  scapula and one point of the humerus coincide. The figure shows, on an *unconstrained*
  reconstruction, how far apart those two points actually land — for the lab's ``RGJC`` pair and for
  the calibrated pair — plus the clavicle: the measured RCAS--RCAJ distance frame by frame against
  the single constant length the joint imposes.
* **step 3 — the closed loop.** Per-frame marker RMSE of each step against the all-FREE floor, the
  same split by marker group, and how far step 3 had to move what steps 1 and 2 had decided.

The calibration takes about a minute, so its results are cached in
``results/calibration/steps.npz``. Pass ``--refresh`` to recompute.

Run:
    python studies/figures/calibration_steps.py [--save] [--refresh]
"""

import numpy as np
from matplotlib import pyplot as plt

from bionc import NaturalCoordinates

from examples._shared.c3d_data import MultiC3dData, load_markers_multi, load_named_markers
from examples._shared.frames import scs_to_natural
from examples._shared.ik import rmse_mm
from examples.clinical.model import GH_GLENOID, GH_HEAD, build_model_free
from studies.shoulder_calibration import (
    MARKER_SET,
    calibrate,
    ellipsoid_surface_distance_mm,
    to_segment_frame,
    trials,
)
from studies.shoulder_calibration_loo import point_in_global
from studies.figures import RESULTS_DIR, STEP_COLORS, finish, parse_args

CACHE = RESULTS_DIR / "calibration" / "steps.npz"


def compute(paths) -> dict:
    """Run the three steps once and reduce them to the arrays the figures need."""
    result = calibrate(paths, marker_set=MARKER_SET, verbose=False)
    step1, step2, step3 = result.step1, result.step2, result.step3

    # an unconstrained reconstruction of the same frames: the yardstick the calibrated geometry is
    # measured against, since it is the one reconstruction that imposes none of it
    data = MultiC3dData(paths)
    free_model = build_model_free(data, marker_set=MARKER_SET)
    free_markers = load_markers_multi(free_model, paths, result.frames)
    free_Q = np.asarray(free_model.Q_from_markers(free_markers))

    def gap_mm(scapula_point, humerus_point):
        first = point_in_global(free_model, "RSCAPULA", scapula_point, free_Q)
        second = point_in_global(free_model, "RHUMERUS", humerus_point, free_Q)
        return np.linalg.norm(first - second, axis=0) * 1000

    scapula = result.model.segments["RSCAPULA"]
    humerus = result.model.segments["RHUMERUS"]
    contact_global = point_in_global(
        free_model, "RSCAPULA", scapula.marker_from_name("SCAP_CENTROID").position, free_Q
    )

    # the clavicle as the data sees it: RCAS is a thorax marker the model does not track
    clavicle_markers = load_named_markers(paths, ("RCAS", "RCAJ"), result.frames)
    clavicle_measured_mm = np.linalg.norm(clavicle_markers[:, 0] - clavicle_markers[:, 1], axis=0) * 1000

    warm = step1["warm_start"]
    arrays = dict(
        cloud=step1["cloud"],
        cloud_free=to_segment_frame(free_model, free_Q, contact_global, segment="THORAX"),
        thorax_center=step1["reference"]["center"],
        thorax_scale=step1["reference"]["scale"],
        warm_semi_axes=warm["semi_axes"],
        warm_center=warm["center"],
        step1_semi_axes=step1["sol"]["semi_axes"],
        step1_center=step1["sol"]["ellipsoid_center_scs"],
        step3_semi_axes=step3["sol"]["semi_axes"],
        step3_center=step3["sol"]["ellipsoid_center_scs"],
        step1_rmse=step1["sol"]["per_frame_marker_rmse_mm"],
        step2_rmse=step2["sol"]["per_frame_marker_rmse_mm"],
        step3_rmse=step3["sol"]["per_frame_marker_rmse_mm"],
        gap_uncalibrated_mm=gap_mm(scapula.marker_from_name("RGJC").position, humerus.marker_from_name("RGJC").position),
        gap_calibrated_mm=gap_mm(
            scs_to_natural(result.model, "RSCAPULA", step3["centres"]["glenoid"]),
            scs_to_natural(result.model, "RHUMERUS", step3["centres"]["head"]),
        ),
        clavicle_measured_mm=clavicle_measured_mm,
        clavicle_calibrated_mm=np.array([step3["sol"]["parameters"]["Clavicle.length"] * 1000]),
        clavicle_initial_mm=np.array([step2["sol"]["theta0"][-1] * 1000]),
        drift_semi_axes=step3["sol"]["semi_axes"] - step1["sol"]["semi_axes"],
        drift_center=step3["sol"]["ellipsoid_center_scs"] - step1["sol"]["ellipsoid_center_scs"],
        drift_glenoid=step3["centres"]["glenoid"] - step2["centres"]["glenoid"],
        drift_head=step3["centres"]["head"] - step2["centres"]["head"],
        drift_clavicle=np.array(
            [step3["sol"]["parameters"]["Clavicle.length"] - step2["sol"]["parameters"]["Clavicle.length"]]
        ),
    )
    arrays["free_rmse"] = _free_rmse(free_model, free_markers, free_Q)
    return arrays


def _free_rmse(free_model, markers, Q) -> np.ndarray:
    """Per-frame marker RMSE [mm] of the unconstrained reconstruction on the calibration frames."""
    per_frame = np.zeros(Q.shape[1])
    for frame in range(Q.shape[1]):
        residual = np.asarray(
            free_model.markers_constraints(markers[:3, :, frame], NaturalCoordinates(Q[:, frame]), only_technical=True)
        ).reshape(3, -1, order="F")
        per_frame[frame] = rmse_mm(np.linalg.norm(residual, axis=0))
    return per_frame


def load(paths, refresh: bool = False) -> dict:
    """Cached :func:`compute`."""
    if CACHE.exists() and not refresh:
        print(f"reusing {CACHE} (pass --refresh to recompute)")
        return dict(np.load(CACHE))
    print("running the three calibration steps ...")
    arrays = compute(paths)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **arrays)
    print(f"cached to {CACHE}")
    return arrays


# ------------------------------------------------------------------------------------ step 1
def _ellipse_section(semi_axes, center, first: int, second: int, third: int, offset: float):
    """Outline of the ellipsoid cut by the plane ``axis[third] = center[third] + offset``."""
    shrink = 1 - (offset / semi_axes[third]) ** 2
    if shrink <= 0:
        return None
    angle = np.linspace(0, 2 * np.pi, 200)
    return (
        center[first] + semi_axes[first] * np.sqrt(shrink) * np.cos(angle),
        center[second] + semi_axes[second] * np.sqrt(shrink) * np.sin(angle),
    )


def plot_step1(data: dict):
    """The contact cloud with the calibrated ellipsoid through it, and the residual distribution."""
    cloud = data["cloud"] * 1000
    semi_axes, center = data["step3_semi_axes"] * 1000, data["step3_center"] * 1000
    planes = [(0, 1, 2, "X antero-posterior", "Y"), (0, 2, 1, "X", "Z medio-lateral"), (2, 1, 0, "Z", "Y")]

    figure, axes = plt.subplots(1, 4, figsize=(17, 4.6), constrained_layout=True)
    figure.suptitle(
        "Step 1 — the scapulothoracic ellipsoid, cut through the middle of the contact patch",
        fontsize=14,
        fontweight="bold",
    )

    for axis, (first, second, third, xlabel, ylabel) in zip(axes, planes):
        offset = float(np.mean(cloud[third]) - center[third])
        section = _ellipse_section(semi_axes, center, first, second, third, offset)
        if section is not None:
            axis.plot(*section, color=STEP_COLORS["step3"], lw=2, label="calibrated ellipsoid")
        axis.scatter(cloud[first], cloud[second], s=14, color="tab:gray", alpha=0.7, label="contact point")
        axis.scatter([center[first]], [center[second]], marker="+", s=140, color="black", label="centre")
        axis.set_xlabel(f"{xlabel} (mm)")
        axis.set_ylabel(f"{ylabel} (mm)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    # Three ellipsoids against the same cloud. Step 3 is allowed to be slightly worse here than
    # step 1: step 1 answers only to the contact point, step 3 has the clavicle and the glenohumeral
    # joint pulling on it too, so the gap between them is the price of closing the loop.
    distances = {
        "warm start": (ellipsoid_surface_distance_mm(data["cloud"], data["warm_semi_axes"], data["warm_center"]), "tab:gray"),
        "step 1": (ellipsoid_surface_distance_mm(data["cloud"], data["step1_semi_axes"], data["step1_center"]), STEP_COLORS["step1"]),
        "step 3": (ellipsoid_surface_distance_mm(data["cloud"], data["step3_semi_axes"], data["step3_center"]), STEP_COLORS["step3"]),
    }
    everything = np.concatenate([values for values, _ in distances.values()])
    bins = np.linspace(everything.min(), everything.max(), 40)
    for label, (values, color) in distances.items():
        axes[3].hist(values, bins=bins, histtype="step", lw=1.8, color=color,
                     label=f"{label} — RMS {np.sqrt(np.mean(values**2)):.1f} mm")
    axes[3].axvline(0, color="black", lw=1)
    axes[3].set_xlabel("signed distance to the surface (mm)")
    axes[3].set_ylabel("frames")
    axes[3].set_title("how well the contact point sits on the surface")
    axes[3].grid(True, axis="y", alpha=0.25)
    axes[3].legend(fontsize=8)
    return figure


# ------------------------------------------------------------------------------------ step 2
def plot_step2(data: dict):
    """The glenohumeral centres and the clavicle length, as the unconstrained data sees them."""
    figure, (axis_gh, axis_hist, axis_clav) = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    figure.suptitle(
        "Step 2 — functional joint centres and clavicle length, measured on a FREE reconstruction",
        fontsize=14,
        fontweight="bold",
    )

    uncalibrated, calibrated = data["gap_uncalibrated_mm"], data["gap_calibrated_mm"]
    frames = np.arange(len(uncalibrated))
    axis_gh.plot(frames, uncalibrated, lw=1, color="tab:blue",
                 label=f"lab RGJC pair (mean {uncalibrated.mean():.1f} mm)")
    axis_gh.plot(frames, calibrated, lw=1, color=STEP_COLORS["step2"],
                 label=f"calibrated centres (mean {calibrated.mean():.1f} mm)")
    axis_gh.set_xlabel("calibration frame")
    axis_gh.set_ylabel("scapula point to humerus point (mm)")
    axis_gh.set_title("the two points a spherical GH must fuse")
    axis_gh.grid(True, alpha=0.25)
    axis_gh.legend(fontsize=8)

    bins = np.linspace(0, max(uncalibrated.max(), calibrated.max()), 40)
    axis_hist.hist(uncalibrated, bins=bins, alpha=0.55, color="tab:blue", label="lab RGJC pair")
    axis_hist.hist(calibrated, bins=bins, alpha=0.75, color=STEP_COLORS["step2"], label="calibrated")
    axis_hist.set_xlabel("separation (mm)")
    axis_hist.set_ylabel("frames")
    axis_hist.set_title("calibration pulls the two centres together")
    axis_hist.grid(True, axis="y", alpha=0.25)
    axis_hist.legend(fontsize=8)

    measured = data["clavicle_measured_mm"]
    warm_start, calibrated_length = float(data["clavicle_initial_mm"][0]), float(data["clavicle_calibrated_mm"][0])
    axis_clav.hist(measured, bins=35, color="tab:gray", alpha=0.75,
                   label=f"measured RCAS-RCAJ ({measured.min():.0f}-{measured.max():.0f} mm)")
    axis_clav.axvline(calibrated_length, color=STEP_COLORS["step3"], lw=2,
                      label=f"calibrated {calibrated_length:.2f} mm\n(warm start {warm_start:.2f} mm — they agree,\nso the length needs no prior)")
    axis_clav.set_xlabel("clavicle length (mm)")
    axis_clav.set_ylabel("frames")
    axis_clav.set_title("the scatter a constant length has to absorb")
    axis_clav.grid(True, axis="y", alpha=0.25)
    axis_clav.legend(fontsize=8)
    return figure


# ------------------------------------------------------------------------------------ step 3
def plot_step3(data: dict):
    """What the closed loop costs in marker fit, and how far it moved steps 1 and 2."""
    figure, (axis_rmse, axis_box, axis_drift) = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    figure.suptitle("Step 3 — closing the loop: what it costs and what it moved", fontsize=14, fontweight="bold")

    series = {
        "FREE (floor)": (data["free_rmse"], STEP_COLORS["free"]),
        "step 1 (thorax+scapula)": (data["step1_rmse"], STEP_COLORS["step1"]),
        "step 2 (ST free)": (data["step2_rmse"], STEP_COLORS["step2"]),
        "step 3 (+ ellipsoid)": (data["step3_rmse"], STEP_COLORS["step3"]),
    }
    for label, (values, color) in series.items():
        axis_rmse.plot(values, lw=1, color=color, alpha=0.85, label=f"{label} — {np.sqrt(np.mean(values**2)):.2f} mm")
    axis_rmse.set_xlabel("calibration frame (pooled over the 8 trials)")
    axis_rmse.set_ylabel("marker RMSE (mm)")
    axis_rmse.set_title("per-frame fit")
    axis_rmse.grid(True, alpha=0.25)
    axis_rmse.legend(fontsize=8)

    axis_box.boxplot([values for values, _ in series.values()], tick_labels=[label.split(" (")[0] for label in series],
                     showfliers=False)
    axis_box.set_ylabel("marker RMSE (mm)")
    axis_box.set_title("distribution over frames")
    axis_box.grid(True, axis="y", alpha=0.25)
    axis_box.tick_params(axis="x", rotation=20)

    drift = {
        "ellipsoid\nsemi-axes": np.linalg.norm(data["drift_semi_axes"]),
        "ellipsoid\ncentre": np.linalg.norm(data["drift_center"]),
        "glenoid\ncentre": np.linalg.norm(data["drift_glenoid"]),
        "humeral head\ncentre": np.linalg.norm(data["drift_head"]),
        "clavicle\nlength": abs(float(data["drift_clavicle"][0])),
    }
    axis_drift.bar(list(drift), [value * 1000 for value in drift.values()], color="tab:purple", alpha=0.8)
    axis_drift.set_ylabel("displacement (mm)")
    axis_drift.set_title("how far step 3 moved steps 1 and 2")
    axis_drift.grid(True, axis="y", alpha=0.25)
    axis_drift.tick_params(axis="x", labelsize=8)
    return figure


def main():
    arguments = parse_args(__doc__, refresh=True)
    data = load(trials(), refresh=arguments.refresh)
    figures = {
        "calibration_step1_ellipsoid": plot_step1(data),
        "calibration_step2_joint_centres": plot_step2(data),
        "calibration_step3_closed_loop": plot_step3(data),
    }
    finish(figures, arguments.save)


if __name__ == "__main__":
    main()
