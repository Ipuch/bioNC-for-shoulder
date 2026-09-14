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

    step 3  Everything at once, closing the loop: clavicle + ellipsoid + glenohumeral, warm-started
            from steps 1 and 2. -> the final model, and how far step 3 had to move steps 1 and 2.

Step 3's glenohumeral joint is the one thing this study offers in two versions, selected by
``glenohumeral=``:

    "spherical"        the two centres are made coincident (3 constraints) and step 3 re-solves them
                       together with the ellipsoid and the clavicle. The default, and the model the
                       ``results/`` tree describes.
    "constant_length"  the two centres are held a calibrated distance ``L`` apart (1 constraint), so
                       the humeral head sits anywhere on a sphere of radius ``L`` about the glenoid
                       rather than being nailed to a point. Written into ``results_gh_constant/``.

Why the *length* is calibrated only in that second version, and only with the centres frozen: with
the two centres free as well, the constraint is rank deficient -- if ``(c_glenoid, c_head, L)`` fits,
so does ``(c_glenoid, c_head + d, ||d||)`` for any ``d``, because ``||P_s - P_h|| = ||R_h d||`` is
then constant. Pinning the centres at what step 2 found is exactly the condition that makes ``L``
identifiable, so the constant-length step 3 calibrates the ellipsoid, the clavicle and ``L``, and
leaves the two centres alone.

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/shoulder_calibration.py [--gh {spherical,constant_length}]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


import numpy as np
from scipy.optimize import least_squares

from bionc import NaturalCoordinates

from examples._shared.c3d_data import MultiC3dData, load_markers_multi, select_calibration_frames
from examples._shared.frames import point_in_global, rodrigues_matrix, scs_to_natural, segment_transformation_matrix
from examples.clinical.model import (
    GH_GLENOID,
    GH_HEAD,
    add_glenohumeral_centres,
    build_model_constrained,
    build_model_free,
    build_scapulothoracic_ellipsoid_model,
    glenohumeral_centres,
    set_glenohumeral_centres,
)
from studies import GLENOHUMERAL, RESULTS_ROOTS
from studies.kinematic_calibration import (
    EllipsoidOrientation,
    EllipsoidSemiAxes,
    JointLength,
    KinematicCalibration,
    MarkerPosition,
    labels_at_bounds,
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

# Box for the constant-length glenohumeral radius [m]. Absolute rather than a multiple of its warm
# start, which is itself a small measured gap: the range that means something is anatomical. The
# floor is far enough from zero to keep the constraint Jacobian well conditioned, and a length that
# ends up riding it is the calibration saying the data wants a spherical joint after all -- which
# ``parameters_at_bounds`` reports rather than hides.
GH_LENGTH_BOUNDS = (0.001, 0.050)


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
def to_segment_frame(model, Q: np.ndarray, points_global: np.ndarray, segment: str = "THORAX") -> np.ndarray:
    """
    Express a global trajectory ``(3, nb_frames)`` in a segment's own coordinate system, per frame.

    The inverse of the convention ``add_natural_marker_from_segment_coordinates`` uses:
    ``L = M @ inv([u, v, w]) @ (P - rp)`` with ``M`` from
    :func:`~examples._shared.frames.segment_transformation_matrix`.

    This is the one place that conversion is written; everything that needs a global point in a
    segment frame goes through here.
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
        local[:, frame] = M @ np.linalg.solve(uvw, points_global[:, frame] - rp)
    return local


def _marker_indices(model, names) -> list[int]:
    """Positions of ``names`` within ``model.marker_names_technical``."""
    technical = list(model.marker_names_technical)
    return [technical.index(name) for name in names]


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
    contacts = markers[:3, _marker_indices(model, SCAPULA_LANDMARKS), :].mean(axis=1)
    return to_segment_frame(model, Q, contacts, segment="THORAX")


def thorax_reference(model, markers: np.ndarray, nb_frames_sampled: int = 40) -> dict:
    """
    A subject-sized box for the thoracic ellipsoid, from the four thorax landmarks themselves.

    The scapula only ever sweeps a *patch* of the thorax, and a patch does not determine an
    ellipsoid: on this dataset the surface fit sits at ~5.1 mm whether the semi-axes are 80 mm or
    300 mm. Something has to set the scale, and the least arbitrary choice is the subject's own
    thorax: ``scale`` is the largest half-extent of SJN/SXS/CV7/TV8 in the thorax segment frame and
    ``center`` the centre of their bounding box. All the ellipsoid bounds are then multiples of the
    subject's anatomy rather than magic numbers -- and, crucially, the calibration reports which
    parameters end up riding those bounds.

    The landmarks barely move in the thorax frame, so ``nb_frames_sampled`` frames spread over the
    trial are enough to size the box; the rest would only repeat them.

    Returns ``{"center": (3,), "scale": float}`` in metres.
    """
    Q = np.asarray(model.Q_from_markers(markers))
    stride = max(1, Q.shape[1] // nb_frames_sampled)
    Q_sampled = Q[:, ::stride]

    points = np.hstack(
        [
            to_segment_frame(model, Q_sampled, markers[:3, landmark, ::stride], segment="THORAX")
            for landmark in _marker_indices(model, THORAX_LANDMARKS)
        ]
    )
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

    Returns ``{"semi_axes", "center", "rotation", "residual_mm", "prior_pull_mm", "at_bounds"}``:
    lengths in metres, the centre in thorax segment coordinates, ``rotation`` a 3x3 whose columns
    are the principal axes, and the names of the unknowns that ended on a bound.

    ``residual_mm`` is the RMS *surface* distance and nothing else. The ridge terms are part of the
    minimised vector but not of that number: they are not distances, and with ``prior_scale`` of
    order ``sqrt(prior * nb_points)`` they can outweigh the surface block entirely, which would
    quietly inflate the one figure this study asks the reader to compare against a 5.1 mm noise
    floor. How hard the ridge is pulling is reported separately as ``prior_pull_mm``, the RMS
    deviation of the semi-axes from the subject's thorax scale.
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
        surface = ellipsoid_surface_distance_mm(cloud, semi_axes, center, rodrigues_matrix(rotation_vector)) / 1000
        return np.concatenate([surface, prior_scale * (semi_axes - reference["scale"])])

    solution = least_squares(residual, start, bounds=(lower, upper))
    semi_axes, center, rotation_vector = solution.x[:3], solution.x[3:6], solution.x[6:]

    labels = ["a", "b", "c", "cx", "cy", "cz", "rx", "ry", "rz"]
    at_bounds = labels_at_bounds(solution.x, lower, upper, labels)

    surface_mm = solution.fun[: cloud.shape[1]] * 1000  # the ridge block is deliberately excluded
    return dict(
        semi_axes=semi_axes,
        center=center,
        rotation=rodrigues_matrix(rotation_vector),
        residual_mm=float(np.sqrt(np.mean(surface_mm**2))),
        prior_pull_mm=float(np.sqrt(np.mean((semi_axes - reference["scale"]) ** 2)) * 1000),
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
class CalibrationStep:
    """
    One solved calibration step: the model it produced, its reconstruction, its diagnostics.

    ``sol`` is :meth:`~studies.kinematic_calibration.KinematicCalibration.sol`'s dict; it stays a
    dict because it is also what the leave-one-out folds serialise into ``.npz``. The remaining
    fields are step-specific and stay ``None`` where they do not apply: step 1 carries the ellipsoid
    warm start it was seeded from and the cloud it was fitted to, steps 2 and 3 the glenohumeral
    centres they solved for.
    """

    model: object
    Qopt: np.ndarray
    sol: dict
    centres: dict = None
    warm_start: dict = None
    reference: dict = None
    bounds: dict = None
    cloud: np.ndarray = None


@dataclass
class CalibrationResult:
    """Everything the three steps produced, plus the final calibrated model."""

    train_paths: list[str]
    frames: dict[str, np.ndarray]
    step1: CalibrationStep = None
    step2: CalibrationStep = None
    step3: CalibrationStep = None
    model: object = None

    @property
    def steps(self) -> dict[str, CalibrationStep]:
        return {"step1": self.step1, "step2": self.step2, "step3": self.step3}

    @property
    def parameters(self) -> dict[str, float]:
        """The final (step-3) calibrated parameters, ``{label: value}``."""
        return self.step3.sol["parameters"]

    def summary(self) -> str:
        lines = [f"calibrated on {len(self.train_paths)} trials, {sum(len(f) for f in self.frames.values())} frames"]
        for name, step in self.steps.items():
            at_bounds = step.sol["parameters_at_bounds"]
            lines.append(
                f"  {name}: success={step.sol['success']}  RMSE={step.sol['marker_rmse_mm']:.2f} mm"
                f"  max joint residual={np.max(step.sol['max_joint_residual_per_frame']):.2e}"
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
    glenohumeral: str = GLENOHUMERAL,
    gh_length: float = None,
):
    """
    Rebuild a step-3 model from calibrated values alone, without re-running the optimisation.

    A bionc model cannot be pickled into a results file, so the leave-one-out folds cache numbers
    and this puts a model back together from them. The segment geometry has to come from the same
    ``train_paths`` the fold used, or the natural coordinates the parameters were expressed in no
    longer mean the same thing.

    ``glenohumeral`` and ``gh_length`` have to match the run that produced the numbers -- a fold
    caches both, so replaying one never has to guess which joint it was calibrated with.
    """
    model = build_scapulothoracic_ellipsoid_model(
        MultiC3dData([str(path) for path in train_paths]),
        (*np.asarray(semi_axes, dtype=float), *np.asarray(ellipsoid_center_scs, dtype=float)),
        joint=ellipsoid_joint,
        rotation=rotation,
        marker_set=marker_set,
        calibratable_gh_centres=True,
        gh_constraint=glenohumeral,
        gh_length=gh_length,
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
            report_as="ellipsoid_center",
        ),
    ]
    if calibrate_orientation:
        parameters.append(EllipsoidOrientation("Scapulothoracic", "THORAX", prior=1.0))
    return parameters


def _joint_parameters(glenohumeral: str = GLENOHUMERAL):
    """
    What the chain's joints contribute as unknowns, which depends on the glenohumeral model.

    ``"spherical"`` -- the glenoid centre, the humeral-head centre and the clavicle length. The two
    centres get a light ridge toward their warm start (the lab's own ``RGJC`` estimate). The
    spherical constraint identifies them only through the *variation* of the relative glenohumeral
    rotation, and on this dataset one direction stays nearly flat: letting the glenoid travel 63 mm
    buys 0.01 mm of marker RMSE. Without a prior the optimiser slides freely along that direction;
    with it, the well-determined directions still move and the flat one stays near anatomy. The
    clavicle length needs none -- it lands on 151.3 mm whatever the starting point.

    ``"constant_length"`` -- the clavicle length and the glenohumeral radius, and *not* the two
    centres. Freeing all three at once is the rank-deficient case the module docstring describes;
    the centres are therefore held at what step 2 decided, which leaves the radius as the one
    unknown the constraint actually determines.
    """
    if glenohumeral == "constant_length":
        return [JointLength("Clavicle"), JointLength("Glenohumeral", bounds=GH_LENGTH_BOUNDS)]

    return [
        MarkerPosition(
            "RSCAPULA", GH_GLENOID, targets=(("Glenohumeral", "parent_point"),), half_range=0.08, prior=0.25
        ),
        MarkerPosition("RHUMERUS", GH_HEAD, targets=(("Glenohumeral", "child_point"),), half_range=0.08, prior=0.25),
        JointLength("Clavicle"),
    ]


def _solve_step(model, markers, parameters, *, regularization, Q_init=None, verbose=True, **extras) -> CalibrationStep:
    """Run one all-frames calibration, write the answer into ``model``, and package the result."""
    calibration = KinematicCalibration(model, markers, parameters=parameters, regularization=regularization)
    Qopt = calibration.solve(Q_init=Q_init, verbose=verbose)
    step = CalibrationStep(model=model, Qopt=Qopt, sol=calibration.sol(), **extras)
    calibration.apply_to(model)
    return step


@dataclass
class _Session:
    """What all three steps share: the pooled trials, the frames, and the ridge weight."""

    train_paths: list[str]
    data: MultiC3dData
    frames: dict
    marker_set: str
    regularization: float
    ellipsoid_joint: str
    glenohumeral: str
    calibrate_orientation: bool
    verbose: bool

    @classmethod
    def build(
        cls,
        train_paths,
        *,
        frames_per_trial,
        marker_set,
        parameter_prior,
        ellipsoid_joint,
        glenohumeral,
        calibrate_orientation,
        verbose,
    ) -> "_Session":
        train_paths = [str(path) for path in train_paths]
        data = MultiC3dData(train_paths)
        base = build_model_constrained(data, marker_set=marker_set)
        frames = select_calibration_frames(base, train_paths, per_trial=frames_per_trial)
        return cls(
            train_paths=train_paths,
            data=data,
            frames=frames,
            marker_set=marker_set,
            # the marker objective grows with the frame count, so the ridge has to as well
            regularization=parameter_prior * sum(len(index) for index in frames.values()),
            ellipsoid_joint=ellipsoid_joint,
            glenohumeral=glenohumeral,
            calibrate_orientation=calibrate_orientation,
            verbose=verbose,
        )

    def markers_for(self, model) -> np.ndarray:
        """The calibration frames, as this model's technical markers."""
        return load_markers_multi(model, self.train_paths, self.frames)

    def base_model(self):
        return build_model_constrained(self.data, marker_set=self.marker_set)

    def free_reconstruction(self):
        """
        An all-FREE model of the calibration frames and its reconstruction, ``(model, Q)``.

        The one reconstruction that imposes none of the geometry being calibrated, so it is what
        anything measuring that geometry has to be read on. ``Q_from_markers`` rather than an IK:
        with every joint free there is nothing to iterate on.
        """
        model = build_model_free(self.data, marker_set=self.marker_set)
        return model, np.asarray(model.Q_from_markers(self.markers_for(model)))


def _calibrate_ellipsoid(session: _Session) -> CalibrationStep:
    """
    Step 1 -- the ellipsoid, on a thorax + scapula chain that has nothing else to hide behind.

    With no clavicle and no humerus, the one-point scapulothoracic joint is the only thing holding
    the two segments together, so the contact-point residual has nowhere else to go.
    """
    base = session.base_model()
    base_markers = session.markers_for(base)

    cloud = contact_point_cloud(base, base_markers)
    reference = thorax_reference(base, base_markers)
    bounds = ellipsoid_bounds(reference)
    warm_start = fit_ellipsoid(cloud, reference, with_orientation=session.calibrate_orientation, bounds=bounds)

    model = build_scapulothoracic_ellipsoid_model(
        session.data,
        (*warm_start["semi_axes"], *warm_start["center"]),
        joint=session.ellipsoid_joint,
        rotation=warm_start["rotation"],
        marker_set=session.marker_set,
        clavicle_constraint=False,
        include_humerus=False,
    )
    return _solve_step(
        model,
        session.markers_for(model),
        _ellipsoid_parameters(session.calibrate_orientation, bounds, reference),
        regularization=session.regularization,
        verbose=session.verbose,
        warm_start=warm_start,
        reference=reference,
        bounds=bounds,
        cloud=cloud,
    )


def _calibrate_joint_centres(session: _Session) -> CalibrationStep:
    """Step 2 -- clavicle length and the two glenohumeral centres, scapulothoracic left free."""
    model = add_glenohumeral_centres(session.base_model())
    step = _solve_step(
        model,
        session.markers_for(model),
        _joint_parameters(),
        regularization=session.regularization,
        verbose=session.verbose,
    )
    step.centres = glenohumeral_centres(model)
    return step


def _glenohumeral_length(session: _Session, step2: CalibrationStep) -> float:
    """
    Warm start [m] for the constant-length glenohumeral radius: how far apart step 2's two centres
    land, on average, when nothing forces them together.

    It has to be measured on an unconstrained reconstruction. Step 2's own ``Qopt`` would give ~0,
    because its spherical joint drives that distance to zero by construction -- the separation only
    becomes visible once the constraint is taken away. This is the same quantity the leave-one-out
    sweep reports out of sample as ``gh_gap_mean_mm``.
    """
    free_model, free_Q = session.free_reconstruction()
    glenoid, head = (
        point_in_global(free_model, segment, scs_to_natural(step2.model, segment, step2.centres[centre]), free_Q)
        for segment, centre in (("RSCAPULA", "glenoid"), ("RHUMERUS", "head"))
    )
    return float(np.mean(np.linalg.norm(glenoid - head, axis=0)))


def _close_the_loop(session: _Session, step1: CalibrationStep, step2: CalibrationStep) -> CalibrationStep:
    """
    Step 3 -- everything at once, warm-started from the two halves that were solved apart.

    With ``glenohumeral="constant_length"`` the two centres are written in and then left alone: they
    are what makes the radius identifiable, so the ellipsoid, the clavicle and that radius are the
    unknowns and the centre drift against step 2 is zero by construction.
    """
    rotation = step1.sol.get(
        "ellipsoid_axes_scs", step1.warm_start["rotation"] if session.calibrate_orientation else None
    )
    constant_length = session.glenohumeral == "constant_length"
    gh_length = float(np.clip(_glenohumeral_length(session, step2), *GH_LENGTH_BOUNDS)) if constant_length else None

    model = build_scapulothoracic_ellipsoid_model(
        session.data,
        (*step1.sol["semi_axes"], *step1.sol["ellipsoid_center_scs"]),
        joint=session.ellipsoid_joint,
        rotation=rotation,
        marker_set=session.marker_set,
        calibratable_gh_centres=True,
        gh_constraint=session.glenohumeral,
        gh_length=gh_length,
    )
    set_glenohumeral_centres(model, glenoid_scs=step2.centres["glenoid"], head_scs=step2.centres["head"])
    model.joints["Clavicle"].length = float(step2.sol["parameters"]["Clavicle.length"])

    step = _solve_step(
        model,
        session.markers_for(model),
        _ellipsoid_parameters(session.calibrate_orientation, step1.bounds, step1.reference)
        + _joint_parameters(session.glenohumeral),
        regularization=session.regularization,
        Q_init=step2.Qopt,
        verbose=session.verbose,
    )
    step.centres = glenohumeral_centres(model)
    return step


def calibrate(
    train_paths,
    *,
    frames_per_trial: int = FRAMES_PER_TRIAL,
    marker_set: str = MARKER_SET,
    calibrate_orientation: bool = False,
    ellipsoid_joint: str = "point",
    glenohumeral: str = GLENOHUMERAL,
    parameter_prior: float = PARAMETER_PRIOR,
    verbose: bool = True,
) -> CalibrationResult:
    """
    Run the three calibration steps on ``train_paths`` and return the calibrated model + diagnostics.

    Every model is built from a :class:`~examples._shared.c3d_data.MultiC3dData` over *these trials
    only*, so the segment geometry and the data-driven joint lengths are trained on the same set as
    the calibrated parameters -- what a held-out evaluation needs to stay honest.

    ``glenohumeral`` selects step 3's glenohumeral joint, ``"spherical"`` or ``"constant_length"``;
    steps 1 and 2 are the same either way, so the two runs differ in exactly one thing.
    """
    session = _Session.build(
        train_paths,
        frames_per_trial=frames_per_trial,
        marker_set=marker_set,
        parameter_prior=parameter_prior,
        ellipsoid_joint=ellipsoid_joint,
        glenohumeral=glenohumeral,
        calibrate_orientation=calibrate_orientation,
        verbose=verbose,
    )

    step1 = _calibrate_ellipsoid(session)
    step2 = _calibrate_joint_centres(session)
    step3 = _close_the_loop(session, step1, step2)

    return CalibrationResult(
        train_paths=session.train_paths, frames=session.frames, step1=step1, step2=step2, step3=step3, model=step3.model
    )


def gh_argument(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The ``--gh`` flag, shared by every driver and figure so one run names one results tree."""
    parser.add_argument(
        "--gh",
        default=GLENOHUMERAL,
        choices=tuple(RESULTS_ROOTS),
        help=f"step-3 glenohumeral joint; each writes its own results tree (default: {GLENOHUMERAL})",
    )
    return parser


def main():
    arguments = gh_argument(argparse.ArgumentParser(description=__doc__.splitlines()[1])).parse_args()
    paths = trials()
    print(f"calibrating on all {len(paths)} trials: {', '.join(trial_label(path) for path in paths)}")
    print(f"glenohumeral joint: {arguments.gh}")
    result = calibrate(paths, glenohumeral=arguments.gh)
    print(result.summary())


if __name__ == "__main__":
    main()
