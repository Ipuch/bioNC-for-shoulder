"""
Three-step calibration of the shoulder kinematic chain, pooling every trial of a session.

The chain carries parameters no single trial identifies well -- the thoracic ellipsoid the scapula
glides on, the clavicle length, the glenohumeral centre in each of the two bones -- so they are
estimated once, from all the trials at once, by all-frames kinematic calibration
(:class:`~studies.kinematic_calibration.KinematicCalibration`).

    step 1  THORAX + RSCAPULA only, joined *only* by a one-point scapulothoracic ellipsoid joint.
            Nothing else can absorb the residual, so the ellipsoid is what the data has to explain.
            -> semi-axes (a, b, c) and centre; optionally the orientation.

    step 2  Full chain, scapulothoracic left FREE, clavicle CONSTANT_LENGTH, glenohumeral SPHERICAL
            on two dedicated centres. -> clavicle length, glenoid centre, humeral-head centre.
            This is a SCoRE-style functional joint-centre estimation written as a constrained IK.

    step 3  Everything at once, closing the loop: clavicle + ellipsoid + spherical GH, warm-started
            from steps 1 and 2. -> the final model, and how far step 3 had to move steps 1 and 2.

Why the glenohumeral joint has no calibrated *length*: with the two centres free as well, the
constraint is rank deficient -- if ``(c_glenoid, c_head, L)`` fits, so does ``(c_glenoid, c_head + d,
||d||)`` for any ``d``, because ``||P_s - P_h|| = ||R_h d||`` is then constant. The centres are the
identifiable half, so they are what is calibrated. Add ``JointLength("Glenohumeral")`` to the step-3
parameter list once the centres are pinned by other data.

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/shoulder_calibration.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import numpy as np
from scipy.optimize import least_squares

from bionc import NaturalCoordinates

from examples._shared.c3d_data import MultiC3dData, load_markers_multi, select_calibration_frames
from examples._shared.frames import segment_transformation_matrix
from examples.clinical.model import (
    GH_GLENOID,
    GH_HEAD,
    add_glenohumeral_centres,
    build_model_constrained,
    build_scapulothoracic_ellipsoid_model,
    glenohumeral_centres,
    set_glenohumeral_centres,
)
from kinematic_calibration import (
    EllipsoidOrientation,
    EllipsoidSemiAxes,
    JointLength,
    KinematicCalibration,
    MarkerPosition,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data" / "99007140-40.19107308-20260825"
SCAPULA_LANDMARKS = ("RSAA", "RSIA", "RSRS")
THORAX_LANDMARKS = ("SJN", "SXS", "CV7", "TV8")
MARKER_SET = "anatomical"
FRAMES_PER_TRIAL = 35

# Weight (per calibration frame) of the ridges that hold the weakly identified parameters near
# anatomy. The scapula sweeps a patch roughly 55 x 52 x 67 mm on this subject; over that patch the
# best-fit ellipsoid surface residual moves by 0.05 mm as the radius goes from 80 mm to 300 mm,
# against a 5.1 mm noise floor. The curvature signal is two orders of magnitude below the noise, so
# the radius is simply *not* identifiable from the contact point -- see
# studies/figures/frame_selection.py, which draws the patch, and the leave-one-out study. Each
# parameter block scales this weight by its own ``prior``; the ridge is what keeps the solution
# interior to its bounds instead of riding them, and the strength is reported, not hidden.
PARAMETER_PRIOR = 0.02


def trials(data_dir: Path = DATA_DIR) -> list[str]:
    """Every c3d of the session, sorted; ``ANALYTIC*`` first, then ``FUNCTIONAL*``."""
    return sorted(str(path) for path in Path(data_dir).glob("*.c3d"))


def trial_label(path) -> str:
    """``...-PROTOCOL01-FUNCTIONAL2-01.c3d`` -> ``FUNCTIONAL2``."""
    return Path(path).stem.split("-")[-2]


def trial_kind(path) -> str:
    """``ANALYTIC`` or ``FUNCTIONAL``."""
    return "".join(character for character in trial_label(path) if not character.isdigit())


# ----------------------------------------------------------------- ellipsoid warm start
def contact_point_cloud(model, markers: np.ndarray, Q: np.ndarray = None) -> np.ndarray:
    """
    The scapula contact point over every frame, expressed in the THORAX segment coordinate system.

    The contact point is the centroid of the three scapula landmarks (RSAA/RSIA/RSRS, the ISB
    AA/AI/TS), which is what the one-point ellipsoid joint constrains. Expressing it in the thorax
    frame turns the trial into a point cloud on the surface to be fitted -- the ellipsoid warm start
    is then an ordinary surface fit rather than a guess.

    ``Q`` defaults to ``model.Q_from_markers(markers)``; pass an inverse-kinematics solution to place
    the cloud with a reconstruction that enforced the model's constraints.

    Returns ``(3, nb_frames)`` in metres.
    """
    Q = np.asarray(model.Q_from_markers(markers) if Q is None else Q)
    thorax = model.segments["THORAX"]
    M = segment_transformation_matrix(model, "THORAX")
    names = list(model.marker_names_technical)
    landmarks = [names.index(name) for name in SCAPULA_LANDMARKS]

    cloud = np.zeros((3, Q.shape[1]))
    for frame in range(Q.shape[1]):
        Q_thorax = NaturalCoordinates(Q[:, frame]).vector(thorax.index)
        rp = np.asarray(Q_thorax.rp, dtype=float).reshape(3)
        uvw = np.column_stack(
            [np.asarray(Q_thorax.u).reshape(3), np.asarray(Q_thorax.v).reshape(3), np.asarray(Q_thorax.w).reshape(3)]
        )
        contact = markers[:3, landmarks, frame].mean(axis=1)
        cloud[:, frame] = M @ np.linalg.inv(uvw) @ (contact - rp)
    return cloud


def to_segment_frame(model, Q: np.ndarray, points_global: np.ndarray, segment: str = "THORAX") -> np.ndarray:
    """
    Express a global trajectory ``(3, nb_frames)`` in a segment's own coordinate system, per frame.

    The inverse of the convention ``add_natural_marker_from_segment_coordinates`` uses:
    ``L = M @ inv([u, v, w]) @ (P - rp)`` with ``M`` from
    :func:`~examples._shared.frames.segment_transformation_matrix`.
    """
    Q = np.asarray(Q)
    segment_object = model.segments[segment]
    M = segment_transformation_matrix(model, segment)

    local = np.zeros_like(np.asarray(points_global, dtype=float))
    for frame in range(local.shape[1]):
        Q_segment = NaturalCoordinates(Q[:, frame]).vector(segment_object.index)
        rp = np.asarray(Q_segment.rp, dtype=float).reshape(3)
        uvw = np.column_stack(
            [
                np.asarray(Q_segment.u).reshape(3),
                np.asarray(Q_segment.v).reshape(3),
                np.asarray(Q_segment.w).reshape(3),
            ]
        )
        local[:, frame] = M @ np.linalg.inv(uvw) @ (points_global[:, frame] - rp)
    return local


def _rodrigues(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def thorax_reference(model, markers: np.ndarray) -> dict:
    """
    A subject-sized box for the thoracic ellipsoid, from the four thorax landmarks themselves.

    The scapula only ever sweeps a *patch* of the thorax, and a patch does not determine an
    ellipsoid: on this dataset the surface fit sits at ~5.1 mm whether the semi-axes are 80 mm or
    300 mm. Something has to set the scale, and the least arbitrary choice is the subject's own
    thorax: ``scale`` is the largest half-extent of SJN/SXS/CV7/TV8 in the thorax segment frame and
    ``center`` the centre of their bounding box. All the ellipsoid bounds are then multiples of the
    subject's anatomy rather than magic numbers -- and, crucially, the calibration reports which
    parameters end up riding those bounds.

    Returns ``{"center": (3,), "scale": float}`` in metres.
    """
    Q = np.asarray(model.Q_from_markers(markers))
    thorax = model.segments["THORAX"]
    M = segment_transformation_matrix(model, "THORAX")
    names = list(model.marker_names_technical)
    landmarks = [names.index(name) for name in THORAX_LANDMARKS]

    points = []
    for frame in range(0, Q.shape[1], max(1, Q.shape[1] // 40)):
        Q_thorax = NaturalCoordinates(Q[:, frame]).vector(thorax.index)
        rp = np.asarray(Q_thorax.rp, dtype=float).reshape(3)
        uvw = np.column_stack(
            [np.asarray(Q_thorax.u).reshape(3), np.asarray(Q_thorax.v).reshape(3), np.asarray(Q_thorax.w).reshape(3)]
        )
        for landmark in landmarks:
            points.append(M @ np.linalg.inv(uvw) @ (markers[:3, landmark, frame] - rp))

    points = np.array(points).T
    return dict(
        center=(points.max(axis=1) + points.min(axis=1)) / 2,
        scale=float(((points.max(axis=1) - points.min(axis=1)) / 2).max()),
    )


def ellipsoid_bounds(reference: dict, semi_axis_scale=(0.6, 1.8), center_half_range: float = 0.15) -> dict:
    """Boxes for the ellipsoid unknowns, as multiples of the subject's thorax (see :func:`thorax_reference`)."""
    return dict(
        semi_axes=(semi_axis_scale[0] * reference["scale"], semi_axis_scale[1] * reference["scale"]),
        center_half_range=center_half_range,
    )


def fit_ellipsoid(
    cloud: np.ndarray,
    reference: dict,
    with_orientation: bool = False,
    bounds: dict = None,
    prior: float = PARAMETER_PRIOR,
) -> dict:
    """
    Least-squares ellipsoid through a point cloud, as the warm start of the real calibration.

    A general algebraic quadric fit is *not* used on purpose: an unconstrained quadric fitted to a
    patch happily comes back as a hyperboloid, and the algebraic residual ``sum_i (a_i.(P-C))^2/s_i
    - 1`` is not scale invariant -- it rewards inflating the ellipsoid until it looks like a plane.
    So the ellipsoid form is imposed from the start, the residual is the *radial surface distance*
    (metres, comparable across sizes), and the unknowns are boxed by :func:`ellipsoid_bounds`.

    Returns ``{"semi_axes", "center", "rotation", "residual_mm", "at_bounds"}``: lengths in metres,
    the centre in thorax segment coordinates, ``rotation`` a 3x3 whose columns are the principal
    axes, and the names of the unknowns that ended on a bound.
    """
    bounds = bounds or ellipsoid_bounds(reference)
    semi_low, semi_high = bounds["semi_axes"]
    center0, half_range = reference["center"], bounds["center_half_range"]

    start = np.concatenate([np.full(3, reference["scale"]), center0, np.zeros(3)])
    lower = np.concatenate([np.full(3, semi_low), center0 - half_range, np.full(3, -0.6)])
    upper = np.concatenate([np.full(3, semi_high), center0 + half_range, np.full(3, 0.6)])
    if not with_orientation:
        lower[6:], upper[6:], start[6:] = -1e-9, 1e-9, 0.0
    start = np.clip(start, lower, upper)

    # The ridge is not cosmetic: over a ~4x change in radius the cloud residual moves 0.05 mm against
    # a 5.1 mm noise floor (see the module docstring), so without a prior the fit simply runs to
    # whichever bound is furthest. The prior says "the thoracic ellipsoid is about thorax-sized".
    prior_scale = np.sqrt(prior * cloud.shape[1])

    def residual(parameters):
        semi_axes, center, rotation_vector = parameters[:3], parameters[3:6], parameters[6:]
        surface = ellipsoid_surface_distance_mm(cloud, semi_axes, center, _rodrigues(rotation_vector)) / 1000
        return np.concatenate([surface, prior_scale * (semi_axes - reference["scale"])])

    solution = least_squares(residual, start, bounds=(lower, upper))
    semi_axes, center, rotation_vector = solution.x[:3], solution.x[3:6], solution.x[6:]

    width = np.maximum(upper - lower, 1e-12)
    margin = np.minimum(solution.x - lower, upper - solution.x) / width
    labels = ["a", "b", "c", "cx", "cy", "cz", "rx", "ry", "rz"]
    at_bounds = [label for label, close, free in zip(labels, margin <= 0.01, width > 1e-6) if close and free]

    return dict(
        semi_axes=semi_axes,
        center=center,
        rotation=_rodrigues(rotation_vector),
        residual_mm=float(np.sqrt(np.mean(residual(solution.x) ** 2)) * 1000),
        at_bounds=at_bounds,
    )


def ellipsoid_surface_distance_mm(cloud: np.ndarray, semi_axes, center, rotation=None) -> np.ndarray:
    """
    Approximate signed distance [mm] of each cloud point to the ellipsoid surface, along the radius.

    Positive is outside. This is the parameter-free way to compare two calibrated ellipsoids over
    the patch the scapula actually visits -- very different ``(a, b, c, centre)`` triplets can
    describe nearly the same surface locally, so comparing the surfaces beats comparing the numbers.
    """
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    offset = np.asarray(cloud, dtype=float) - np.asarray(center, dtype=float).reshape(3, 1)
    projections = rotation.T @ offset
    normalized = np.sqrt(np.sum((projections / np.asarray(semi_axes, dtype=float).reshape(3, 1)) ** 2, axis=0))
    return np.linalg.norm(offset, axis=0) * (1 - 1 / np.maximum(normalized, 1e-9)) * 1000


# ----------------------------------------------------------------------------- the pipeline
@dataclass
class CalibrationResult:
    """Everything the three steps produced, plus the final calibrated model."""

    train_paths: list[str]
    frames: dict[str, np.ndarray]
    step1: dict = field(default_factory=dict)
    step2: dict = field(default_factory=dict)
    step3: dict = field(default_factory=dict)
    model: object = None

    @property
    def parameters(self) -> dict[str, float]:
        """The final (step-3) calibrated parameters, ``{label: value}``."""
        return self.step3["sol"]["parameters"]

    def summary(self) -> str:
        lines = [f"calibrated on {len(self.train_paths)} trials, {sum(len(f) for f in self.frames.values())} frames"]
        for step in ("step1", "step2", "step3"):
            sol = getattr(self, step)["sol"]
            at_bounds = sol["parameters_at_bounds"]
            lines.append(
                f"  {step}: success={sol['success']}  RMSE={sol['marker_rmse_mm']:.2f} mm"
                f"  max joint residual={np.max(sol['max_joint_residual_per_frame']):.2e}"
                + (f"  AT BOUNDS: {', '.join(at_bounds)}" if at_bounds else "")
            )
        for label, value in self.parameters.items():
            if "rotation" in label:
                lines.append(f"    {label:<38s} {np.degrees(value):9.2f} deg")
            else:
                lines.append(f"    {label:<38s} {value * 1000:9.2f} mm")
        return "\n".join(lines)


def rebuild_calibrated_model(
    train_paths,
    *,
    semi_axes,
    ellipsoid_center_scs,
    glenoid_scs,
    head_scs,
    clavicle_length,
    rotation=None,
    marker_set: str = MARKER_SET,
    ellipsoid_joint: str = "point",
):
    """
    Rebuild a step-3 model from calibrated values alone, without re-running the optimisation.

    A bionc model cannot be pickled into a results file, so the leave-one-out folds cache numbers
    and this puts a model back together from them. The segment geometry has to come from the same
    ``train_paths`` the fold used, or the natural coordinates the parameters were expressed in no
    longer mean the same thing.
    """
    model = build_scapulothoracic_ellipsoid_model(
        MultiC3dData([str(path) for path in train_paths]),
        (*np.asarray(semi_axes, dtype=float), *np.asarray(ellipsoid_center_scs, dtype=float)),
        joint=ellipsoid_joint,
        rotation=rotation,
        marker_set=marker_set,
        calibratable_gh_centres=True,
    )
    set_glenohumeral_centres(model, glenoid_scs=glenoid_scs, head_scs=head_scs)
    model.joints["Clavicle"].length = float(clavicle_length)
    return model


def _ellipsoid_parameters(calibrate_orientation: bool, bounds: dict, reference: dict):
    parameters = [
        EllipsoidSemiAxes("Scapulothoracic", bounds=bounds["semi_axes"], prior=1.0),
        MarkerPosition(
            "THORAX",
            "ELLIPSOID_CENTER",
            targets=(("Scapulothoracic", "ellipsoid_center"),),
            half_range=bounds["center_half_range"],
            box_center=reference["center"],  # "inside the thorax", not "near the previous answer"
            prior=1.0,
        ),
    ]
    if calibrate_orientation:
        parameters.append(EllipsoidOrientation("Scapulothoracic", "THORAX", prior=1.0))
    return parameters


def _joint_parameters():
    """
    The glenoid centre, the humeral-head centre and the clavicle length.

    The two centres get a light ridge toward their warm start (the lab's own ``RGJC`` estimate).
    The spherical constraint identifies them only through the *variation* of the relative
    glenohumeral rotation, and on this dataset one direction stays nearly flat: letting the glenoid
    travel 63 mm buys 0.01 mm of marker RMSE. Without a prior the optimiser slides freely along
    that direction; with it, the well-determined directions still move and the flat one stays near
    anatomy. The clavicle length needs none -- it lands on 151.3 mm whatever the starting point.
    """
    return [
        MarkerPosition(
            "RSCAPULA", GH_GLENOID, targets=(("Glenohumeral", "parent_point"),), half_range=0.08, prior=0.25
        ),
        MarkerPosition("RHUMERUS", GH_HEAD, targets=(("Glenohumeral", "child_point"),), half_range=0.08, prior=0.25),
        JointLength("Clavicle"),
    ]


def _run(model, markers, parameters, Q_init=None, verbose: bool = True, **kwargs) -> tuple:
    options = None if verbose else {**KinematicCalibration._default_options(), "ipopt.print_level": 0, "print_time": False}
    calibration = KinematicCalibration(model, markers, parameters=parameters, **kwargs)
    Qopt = calibration.solve(Q_init=Q_init, options=options)
    return calibration, Qopt, calibration.sol()


def calibrate(
    train_paths,
    *,
    frames_per_trial: int = FRAMES_PER_TRIAL,
    marker_set: str = MARKER_SET,
    calibrate_orientation: bool = False,
    ellipsoid_joint: str = "point",
    parameter_prior: float = PARAMETER_PRIOR,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Run the three calibration steps on ``train_paths`` and return the calibrated model + diagnostics.

    Every model is built from a :class:`~examples._shared.c3d_data.MultiC3dData` over *these trials
    only*, so the segment geometry and the data-driven joint lengths are trained on the same set as
    the calibrated parameters -- what a held-out evaluation needs to stay honest.
    """
    train_paths = [str(path) for path in train_paths]
    data = MultiC3dData(train_paths)

    base = build_model_constrained(data, marker_set=marker_set)
    frames = select_calibration_frames(base, train_paths, per_trial=frames_per_trial)
    result = CalibrationResult(train_paths=train_paths, frames=frames)

    # the marker objective grows with the frame count, so the ridge has to as well
    regularization = parameter_prior * sum(len(index) for index in frames.values())

    # --- step 1: the ellipsoid, on a thorax + scapula chain that has nothing else to hide behind
    base_markers = load_markers_multi(base, train_paths, frames)
    cloud = contact_point_cloud(base, base_markers)
    reference = thorax_reference(base, base_markers)
    bounds = ellipsoid_bounds(reference)
    warm_start = fit_ellipsoid(cloud, reference, with_orientation=calibrate_orientation, bounds=bounds)
    theta0 = (*warm_start["semi_axes"], *warm_start["center"])
    model1 = build_scapulothoracic_ellipsoid_model(
        data,
        theta0,
        joint=ellipsoid_joint,
        rotation=warm_start["rotation"],
        marker_set=marker_set,
        clavicle_constraint=False,
        include_humerus=False,
    )
    markers1 = load_markers_multi(model1, train_paths, frames)
    calibration1, Qopt1, sol1 = _run(
        model1,
        markers1,
        _ellipsoid_parameters(calibrate_orientation, bounds, reference),
        regularization=regularization,
        verbose=verbose,
    )
    calibration1.apply_to(model1)
    result.step1 = dict(
        model=model1,
        Qopt=Qopt1,
        sol=sol1,
        warm_start=warm_start,
        reference=reference,
        bounds=bounds,
        cloud=cloud,
    )

    # --- step 2: clavicle length and the two glenohumeral centres, scapulothoracic left free
    model2 = add_glenohumeral_centres(build_model_constrained(data, marker_set=marker_set))
    markers2 = load_markers_multi(model2, train_paths, frames)
    calibration2, Qopt2, sol2 = _run(model2, markers2, _joint_parameters(), regularization=regularization, verbose=verbose)
    calibration2.apply_to(model2)
    result.step2 = dict(model=model2, Qopt=Qopt2, sol=sol2, centres=glenohumeral_centres(model2))

    # --- step 3: the closed loop, warm-started from both halves
    model3 = build_scapulothoracic_ellipsoid_model(
        data,
        (*sol1["semi_axes"], *sol1["ellipsoid_center_scs"]),
        joint=ellipsoid_joint,
        rotation=sol1.get("ellipsoid_axes_scs", warm_start["rotation"] if calibrate_orientation else None),
        marker_set=marker_set,
        calibratable_gh_centres=True,
    )
    centres = result.step2["centres"]
    set_glenohumeral_centres(model3, glenoid_scs=centres["glenoid"], head_scs=centres["head"])
    model3.joints["Clavicle"].length = float(sol2["parameters"]["Clavicle.length"])

    markers3 = load_markers_multi(model3, train_paths, frames)
    calibration3, Qopt3, sol3 = _run(
        model3,
        markers3,
        _ellipsoid_parameters(calibrate_orientation, bounds, reference) + _joint_parameters(),
        Q_init=Qopt2,
        regularization=regularization,
        verbose=verbose,
    )
    calibration3.apply_to(model3)
    result.step3 = dict(model=model3, Qopt=Qopt3, sol=sol3, centres=glenohumeral_centres(model3))
    result.model = model3
    return result


def main():
    paths = trials()
    print(f"calibrating on all {len(paths)} trials: {', '.join(trial_label(path) for path in paths)}")
    result = calibrate(paths)
    print(result.summary())


if __name__ == "__main__":
    main()
