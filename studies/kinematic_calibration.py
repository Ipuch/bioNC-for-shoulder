"""
All-frames kinematic calibration: an inverse kinematics whose *model parameters* are unknowns too.

The optimisation is

    minimise   sum_over_frames  1/2 ||model_markers(Q_f, p) - xp_markers_f||^2  +  lambda ||p - p0||^2
    over        Q_f (natural coordinates of every frame)  and  the shared parameters p
    subject to  rigid-body + joint constraints on every frame,
                (optionally) direct-frame constraints (positive [u, v, w] determinant),

solved as **one** CasADi NLP (IPOPT, exact Hessian) rather than a frame-by-frame solve wrapped in an
outer loop. Because the objective and the constraints are separable per frame, the frames may come
from several trials: just concatenate the marker arrays (see
:func:`examples._shared.c3d_data.load_markers_multi`).

What is calibrated is described by a list of :class:`CalibrationParameter` objects. Each one knows
its warm start and bounds, how to splice MX symbols into ``model.to_mx()``, and how to write the
result back into a numpy model:

* :class:`EllipsoidSemiAxes`     -- the ``(a, b, c)`` of an ellipsoid joint;
* :class:`MarkerPosition`        -- the location of a virtual point (an ellipsoid centre, a
                                    functional joint centre), in **segment coordinates, in metres**;
* :class:`EllipsoidOrientation`  -- the ellipsoid's principal axes, as an incremental rotation;
* :class:`JointLength`           -- the ``length`` of a CONSTANT_LENGTH joint.

The implementation is distilled from ``bionc``'s ``InverseKinematics`` (CasADi backend), keeping only
the per-frame symbolic Q, the marker objective, and the rigid-body / joint / direct-frame constraints.
"""

import numpy as np
from casadi import MX, Function, cos, dot, horzcat, nlpsol, sin, vertcat

from bionc import NaturalCoordinates as NaturalCoordinatesNumpy
from bionc.bionc_casadi import NaturalCoordinates, SegmentNaturalCoordinates
from bionc.bionc_casadi.natural_marker import NaturalMarker, SegmentNaturalVector
from bionc.bionc_casadi.natural_vector import NaturalVector
from bionc.utils.casadi_utils import sarrus

from examples._shared.frames import natural_to_scs, rodrigues_matrix, segment_transformation_matrix


def scapulothoracic_angles(model, Q: np.ndarray, joint_name: str = "Scapulothoracic") -> np.ndarray:
    """Joint Euler angles [3 x nb_frames] in degrees for a model and its natural coordinates ``Q``."""
    Q = np.asarray(Q)
    column = list(model.joints.joint_names).index(joint_name)
    angles = np.zeros((3, Q.shape[1]))
    for f in range(Q.shape[1]):
        angles[:, f] = model.natural_coordinates_to_joint_angles(NaturalCoordinatesNumpy(Q[:, f]))[:, column]
    return np.degrees(angles)


def labels_at_bounds(values, lower, upper, labels, tolerance: float = 0.01) -> list[str]:
    """
    Names of the unknowns that ended within ``tolerance`` of a bound, relative to their box width.

    A parameter riding a bound is not calibrated -- it is *constrained*, and the number it reports
    means nothing. Every fit in this package reports this, so the test lives here once.

    Unknowns whose box has been collapsed to a point (a rotation frozen at zero, say) are excluded:
    they are trivially "at bounds" and saying so is noise, not a warning.
    """
    values, lower, upper = np.asarray(values), np.asarray(lower), np.asarray(upper)
    width = np.maximum(upper - lower, 1e-12)
    margin = np.minimum(values - lower, upper - values) / width
    return [label for label, close, free in zip(labels, margin <= tolerance, width > 1e-6) if close and free]


def rodrigues(rotation_vector: MX) -> MX:
    """Rotation matrix of an MX Rodrigues (axis-angle) vector, smooth and singularity-free at 0."""
    theta = (dot(rotation_vector, rotation_vector) + 1e-12) ** 0.5
    axis = rotation_vector / theta
    K = MX.zeros(3, 3)
    K[0, 1], K[0, 2] = -axis[2], axis[1]
    K[1, 0], K[1, 2] = axis[2], -axis[0]
    K[2, 0], K[2, 1] = -axis[1], axis[0]
    return MX.eye(3) + sin(theta) * K + (1 - cos(theta)) * (K @ K)


# ---------------------------------------------------------------------------------- parameters
class CalibrationParameter:
    """
    One block of shared unknowns of the calibration.

    Subclasses provide the warm start and bounds read off a numpy model, the splice of MX symbols
    into its ``to_mx()`` twin, and the write-back of the solved values into a numpy model.

    ``prior`` is the weight of a per-block ridge ``prior * ||p - p0||^2`` added to the objective.
    It is per block on purpose: a shared parameter the data pins down well (a functional joint
    centre) must not be dragged toward its warm start just because another one (an ellipsoid radius
    fitted to a small patch) needs help.
    """

    size: int = 0
    prior: float = 0.0

    @property
    def labels(self) -> list[str]:
        """One human-readable name per scalar of the block."""
        raise NotImplementedError

    def initial(self, model) -> np.ndarray:
        raise NotImplementedError

    def bounds(self, model) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def bind(self, model_mx, model, symbols: MX) -> None:
        raise NotImplementedError

    def apply(self, model, values: np.ndarray) -> None:
        raise NotImplementedError


class EllipsoidSemiAxes(CalibrationParameter):
    """The ``(a, b, c)`` semi-axis lengths [m] of an ellipsoid joint."""

    size = 3

    def __init__(self, joint: str = "Scapulothoracic", bounds: tuple = (0.03, 0.50), prior: float = 0.0):
        """``bounds`` is ``(lower, upper)``; each side is a scalar or a per-axis triple [m]."""
        self.joint = joint
        self.bounds_ = bounds
        self.prior = prior

    @property
    def labels(self):
        return [f"{self.joint}.semi_axis_{axis}" for axis in "abc"]

    def initial(self, model):
        return np.array([float(length) for length in model.joints[self.joint].semi_axis_lengths])

    def bounds(self, model):
        lower, upper = self.bounds_
        return (
            np.broadcast_to(np.asarray(lower, dtype=float), (3,)).copy(),
            np.broadcast_to(np.asarray(upper, dtype=float), (3,)).copy(),
        )

    def bind(self, model_mx, model, symbols):
        model_mx.joints[self.joint].semi_axis_lengths = (symbols[0], symbols[1], symbols[2])

    def apply(self, model, values):
        model.joints[self.joint].semi_axis_lengths = tuple(float(value) for value in values)


class MarkerPosition(CalibrationParameter):
    """
    The location of a virtual point, **in its segment's orthonormal coordinate system, in metres**.

    Natural coordinates would make the bounds meaningless (they are dimensionless fractions of the
    segment), so the unknown is the physical offset and the constant ``inv(B)`` maps it back.

    ``targets`` says where the point is *used*: pairs of ``(joint name, attribute)`` such as
    ``("Scapulothoracic", "ellipsoid_center")`` or ``("Glenohumeral", "parent_point")``. Only those
    joint attributes are made symbolic -- the segment's own marker list is left alone, so a tracked
    marker keeps being tracked at its measured location while the virtual point moves.

    ``box_center`` fixes the centre of the bounding box in segment coordinates. Leave it None and
    the box follows the warm start, which is fine for a point that is already roughly right (a
    functional joint centre). Set it for a point whose warm start is itself uncertain -- an
    ellipsoid centre fitted to a patch, say -- so that the bound expresses a standing anatomical
    fact ("inside the thorax") instead of drifting along with the previous step's answer.
    """

    size = 3

    def __init__(
        self, segment: str, marker: str, targets, half_range: float = 0.05, box_center=None, prior: float = 0.0
    ):
        self.segment = segment
        self.marker = marker
        self.targets = tuple(targets)
        self.half_range = half_range
        self.box_center = None if box_center is None else np.asarray(box_center, dtype=float).reshape(3)
        self.prior = prior

    @property
    def labels(self):
        return [f"{self.segment}.{self.marker}.{axis}" for axis in "xyz"]

    def initial(self, model):
        return natural_to_scs(model, self.segment, model.segments[self.segment].marker_from_name(self.marker).position)

    def bounds(self, model):
        center = self.initial(model) if self.box_center is None else self.box_center
        return center - self.half_range, center + self.half_range

    def bind(self, model_mx, model, symbols):
        B_inv = np.linalg.inv(segment_transformation_matrix(model, self.segment))
        marker = NaturalMarker(
            name=self.marker,
            parent_name=self.segment,
            position=NaturalVector(B_inv @ symbols),
            is_technical=False,
            is_anatomical=True,
        )
        for joint_name, attribute in self.targets:
            setattr(model_mx.joints[joint_name], attribute, marker)

    def apply(self, model, values):
        from bionc.bionc_numpy.natural_vector import NaturalVector as NaturalVectorNumpy

        B_inv = np.linalg.inv(segment_transformation_matrix(model, self.segment))
        marker = model.segments[self.segment].marker_from_name(self.marker)
        marker.position = NaturalVectorNumpy(B_inv @ np.asarray(values, dtype=float).reshape(3))
        marker.interpolation_matrix = marker.position.interpolate()
        for joint_name, attribute in self.targets:
            setattr(model.joints[joint_name], attribute, marker)


class EllipsoidOrientation(CalibrationParameter):
    """
    The ellipsoid's principal axes, as a Rodrigues rotation *increment* on the axes already in the
    model (warm start ``0``, so the model's own orientation is the starting point).

    A rotation vector rather than Euler angles: smooth everywhere, no gimbal lock, and a box
    constraint on it is an honest "do not rotate more than this" limit.
    """

    size = 3

    def __init__(
        self, joint: str = "Scapulothoracic", segment: str = "THORAX", half_range: float = 0.6, prior: float = 0.0
    ):
        self.joint = joint
        self.segment = segment
        self.half_range = half_range
        self.prior = prior

    @property
    def labels(self):
        return [f"{self.joint}.rotation_{axis}" for axis in "xyz"]

    def initial(self, model):
        return np.zeros(3)

    def bounds(self, model):
        return np.full(3, -self.half_range), np.full(3, self.half_range)

    def reference_axes(self, model) -> np.ndarray:
        """The model's current principal axes, as unit columns in the segment coordinate system."""
        axes = np.column_stack(
            [natural_to_scs(model, self.segment, axis.position) for axis in model.joints[self.joint].ellipsoid_axes]
        )
        return axes / np.linalg.norm(axes, axis=0, keepdims=True)

    def bind(self, model_mx, model, symbols):
        B_inv = np.linalg.inv(segment_transformation_matrix(model, self.segment))
        axes = self.reference_axes(model)
        rotation = rodrigues(symbols)
        model_mx.joints[self.joint].ellipsoid_axes = [
            SegmentNaturalVector(
                name=f"AXIS_{letter}",
                parent_name=self.segment,
                direction=NaturalVector(B_inv @ (rotation @ axes[:, index])),
            )
            for index, letter in enumerate("ABC")
        ]

    def apply(self, model, values):
        from bionc.bionc_numpy.natural_vector import NaturalVector as NaturalVectorNumpy

        B_inv = np.linalg.inv(segment_transformation_matrix(model, self.segment))
        axes = self.rotation_matrix(model, values) @ self.reference_axes(model)
        for index, axis in enumerate(model.joints[self.joint].ellipsoid_axes):
            axis.position = NaturalVectorNumpy(B_inv @ axes[:, index])
            axis.interpolation_matrix = axis.position.interpolate().rot

    @staticmethod
    def rotation_matrix(model, values) -> np.ndarray:
        """Numpy Rodrigues rotation of the solved values (the numeric twin of :func:`rodrigues`)."""
        return rodrigues_matrix(values)


class JointLength(CalibrationParameter):
    """The ``length`` [m] of a CONSTANT_LENGTH joint, bounded as a fraction of its warm start."""

    size = 1

    def __init__(self, joint: str, scale_bounds: tuple[float, float] = (0.5, 1.5), prior: float = 0.0):
        self.joint = joint
        self.scale_bounds = scale_bounds
        self.prior = prior

    @property
    def labels(self):
        return [f"{self.joint}.length"]

    def initial(self, model):
        return np.array([float(model.joints[self.joint].length)])

    def bounds(self, model):
        start = self.initial(model)
        return start * self.scale_bounds[0], start * self.scale_bounds[1]

    def bind(self, model_mx, model, symbols):
        model_mx.joints[self.joint].length = symbols[0]

    def apply(self, model, values):
        model.joints[self.joint].length = float(values[0])


def default_ellipsoid_parameters(joint: str = "Scapulothoracic", segment: str = "THORAX", **kwargs):
    """The historical parameter set: an ellipsoid's semi-axes and the location of its centre."""
    return [
        EllipsoidSemiAxes(joint, **kwargs),
        MarkerPosition(segment, "ELLIPSOID_CENTER", targets=((joint, "ellipsoid_center"),), half_range=0.15),
    ]


# ------------------------------------------------------------------------------- the calibration
class KinematicCalibration:
    """
    All-frames marker inverse kinematics with model parameters as shared optimisation variables.

    Parameters
    ----------
    model
        A bionc (numpy) model. Whatever the ``parameters`` refer to must exist in it.
    markers
        Experimental technical markers, ``(3, nb_technical_markers, nb_frames)`` in metres. Frames
        from several trials may simply be concatenated along the last axis.
    parameters
        The :class:`CalibrationParameter` blocks to calibrate. Defaults to the semi-axes and centre
        of the ``Scapulothoracic`` ellipsoid joint.
    active_direct_frame_constraints
        If True, add per-frame constraints forcing a positive ``det([u, v, w])`` on every segment.
    regularization
        Global multiplier on the per-parameter ``prior`` ridges (each block contributes
        ``regularization * parameter.prior * ||p - p0||^2``). ``0`` disables every ridge. The
        marker objective grows with the number of frames, so pass something proportional to it --
        the pipelines use ``weight_per_frame * nb_frames`` -- otherwise the same number means
        different things on different frame budgets.
    """

    def __init__(
        self,
        model,
        markers: np.ndarray,
        *,
        parameters: list[CalibrationParameter] = None,
        active_direct_frame_constraints: bool = True,
        regularization: float = 0.0,
    ):
        self.model = model
        self.markers = np.asarray(markers)
        self.nb_frames = self.markers.shape[2]
        self.nb_markers = self.markers.shape[1]
        self.nb_segments = model.nb_segments
        self.active_direct_frame_constraints = active_direct_frame_constraints
        self.regularization = regularization
        self.parameters = list(parameters) if parameters is not None else default_ellipsoid_parameters()

        self._model_mx = model.to_mx()
        self._p_sym, self._p0, self._p_lb, self._p_ub = self._make_parameters_symbolic()
        self._build_nlp()

        self.Qopt = None
        self.theta = None
        self.success = None
        self.objective_value = None

    # ------------------------------------------------------------------ setup
    @property
    def nb_parameters(self) -> int:
        return sum(parameter.size for parameter in self.parameters)

    @property
    def parameter_labels(self) -> list[str]:
        return [label for parameter in self.parameters for label in parameter.labels]

    def _parameter_slices(self):
        offset = 0
        for parameter in self.parameters:
            yield parameter, slice(offset, offset + parameter.size)
            offset += parameter.size

    def _make_parameters_symbolic(self):
        """Splice one MX symbol block per parameter into the MX model; return (sym, p0, lb, ub)."""
        symbols, starts, lower, upper = [], [], [], []
        for parameter, _ in self._parameter_slices():
            block = MX.sym(parameter.labels[0].replace(".", "_"), parameter.size)
            parameter.bind(self._model_mx, self.model, block)

            start = np.asarray(parameter.initial(self.model), dtype=float).reshape(parameter.size)
            lb, ub = parameter.bounds(self.model)
            symbols.append(block)
            starts.append(start)
            lower.append(np.asarray(lb, dtype=float).reshape(parameter.size))
            upper.append(np.asarray(ub, dtype=float).reshape(parameter.size))

        return vertcat(*symbols), np.concatenate(starts), np.concatenate(lower), np.concatenate(upper)

    def _segment_coordinates_sym(self, suffix: str) -> MX:
        return vertcat(*[SegmentNaturalCoordinates.sym(f"{suffix}_{s}") for s in range(self.nb_segments)])

    def _direct_frame_constraints(self, Q_frame: NaturalCoordinates) -> MX:
        determinants = []
        for s in range(self.nb_segments):
            u, v, w = Q_frame.vector(s).to_uvw()
            determinants.append(sarrus(horzcat(u, v, w)))
        return vertcat(*determinants)

    def _build_nlp(self):
        nb_dof = 12 * self.nb_segments
        self._nb_Q = nb_dof * self.nb_frames

        Q_frames = [self._segment_coordinates_sym(f"Q_{f}") for f in range(self.nb_frames)]

        objective = 0
        constraints, lbg, ubg = [], [], []
        for f in range(self.nb_frames):
            Q_f = NaturalCoordinates(Q_frames[f])

            marker_defects = self._model_mx.markers_constraints(self.markers[:3, :, f], Q_f, only_technical=True)
            objective = objective + 0.5 * dot(marker_defects, marker_defects)

            phi_r = self._model_mx.rigid_body_constraints(Q_f)
            phi_k = self._model_mx.joint_constraints(Q_f)
            constraints += [phi_r, phi_k]
            nb_equality = phi_r.shape[0] + phi_k.shape[0]
            lbg += [0.0] * nb_equality
            ubg += [0.0] * nb_equality

            if self.active_direct_frame_constraints:
                constraints.append(self._direct_frame_constraints(Q_f))
                lbg += [0.0] * self.nb_segments
                ubg += [np.inf] * self.nb_segments

        for parameter, block in self._parameter_slices():
            weight = self.regularization * parameter.prior
            if weight:
                drift = self._p_sym[block] - self._p0[block]
                objective = objective + weight * dot(drift, drift)

        x = vertcat(vertcat(*Q_frames), self._p_sym)
        self._nlp = {"x": x, "f": objective, "g": vertcat(*constraints)}
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)

    # ------------------------------------------------------------------ solve
    @staticmethod
    def _default_options() -> dict:
        return {
            # The NLP is large but very sparse (frames couple only through the shared parameters),
            # so the exact Hessian is affordable and converges in far fewer iterations than L-BFGS.
            "ipopt.hessian_approximation": "exact",
            "ipopt.max_iter": 3000,
            "ipopt.tol": 1e-8,
            "ipopt.print_level": 5,
            "print_time": True,
        }

    def solve(self, Q_init: np.ndarray = None, options: dict = None) -> np.ndarray:
        """
        Solve the all-frames calibration.

        Q_init : (12*nb_segments, nb_frames) or (12*nb_segments, 1)
            Initial natural coordinates. Defaults to ``model.Q_from_markers``. A single column is
            broadcast to every frame.
        """
        if Q_init is None:
            Q_init = self.model.Q_from_markers(self.markers)
        Q_init = np.asarray(Q_init, dtype=float)
        if Q_init.shape[1] == 1:
            Q_init = np.repeat(Q_init, self.nb_frames, axis=1)
        self.Q_init = Q_init  # kept for the "before optimisation" reference

        x0 = np.concatenate([Q_init.flatten(order="F"), self._p0])
        lbx = np.concatenate([np.full(self._nb_Q, -np.inf), self._p_lb])
        ubx = np.concatenate([np.full(self._nb_Q, np.inf), self._p_ub])

        solver = nlpsol("kinematic_calibration", "ipopt", self._nlp, options or self._default_options())
        result = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=self._lbg, ubg=self._ubg)

        self.success = bool(solver.stats()["success"])
        x_opt = np.array(result["x"]).reshape(-1)
        self.Qopt = x_opt[: self._nb_Q].reshape((12 * self.nb_segments, self.nb_frames), order="F")
        self.theta = x_opt[self._nb_Q :]
        self.objective_value = float(result["f"])
        return self.Qopt

    def apply_to(self, model):
        """Write the calibrated parameters into ``model`` (a numpy model), in place. Returns it."""
        if self.theta is None:
            raise RuntimeError("call solve() before apply_to()")
        for parameter, block in self._parameter_slices():
            parameter.apply(model, self.theta[block])
        return model

    # ------------------------------------------------------------------ output
    def _build_evaluators(self):
        """CasADi functions giving, for one frame, the marker / joint / rigid-body residuals."""
        Q_sym = self._segment_coordinates_sym("eval")
        Q_f = NaturalCoordinates(Q_sym)
        markers_sym = MX.sym("markers", 3, self.nb_markers)

        marker_defects = self._model_mx.markers_constraints(markers_sym, Q_f, only_technical=True)
        joint_defects = self._model_mx.joint_constraints(Q_f)
        rigid_defects = self._model_mx.rigid_body_constraints(Q_f)

        self._fun_marker = Function("marker_res", [Q_sym, markers_sym, self._p_sym], [marker_defects])
        self._fun_joint = Function("joint_res", [Q_sym, self._p_sym], [joint_defects])
        self._fun_rigid = Function("rigid_res", [Q_sym], [rigid_defects])

    def parameters_at_bounds(self, tolerance: float = 0.01) -> list[str]:
        """
        Labels of the parameters that ended on a bound. Always check this before quoting a
        calibrated value -- see :func:`labels_at_bounds`.
        """
        return labels_at_bounds(self.theta, self._p_lb, self._p_ub, self.parameter_labels, tolerance)

    def sol(self) -> dict:
        """
        Frame-by-frame solution summary (after :meth:`solve`).

        Returns the calibrated parameters (as a ``{label: value}`` mapping and as the raw vector),
        which of them ride a bound, and per-frame residuals: marker RMSE [mm] (global and per-frame),
        joint and rigid-body residuals, and the scapulothoracic Euler angles [deg].
        """
        if self.Qopt is None:
            raise RuntimeError("call solve() before sol()")
        self._build_evaluators()

        marker_xyz = np.zeros((3, self.nb_markers, self.nb_frames))
        joint_residuals = np.zeros(self.nb_frames)
        rigid_residuals = np.zeros(self.nb_frames)
        for f in range(self.nb_frames):
            q = self.Qopt[:, f]
            marker_xyz[:, :, f] = np.array(self._fun_marker(q, self.markers[:3, :, f], self.theta)).reshape(
                3, self.nb_markers, order="F"
            )
            joint_residuals[f] = float(np.max(np.abs(np.array(self._fun_joint(q, self.theta)))))
            rigid_residuals[f] = float(np.max(np.abs(np.array(self._fun_rigid(q)))))

        marker_norm_mm = np.sqrt(np.sum(marker_xyz**2, axis=0)) * 1000  # (nb_markers, nb_frames)
        per_frame_rmse_mm = np.sqrt(np.mean(marker_norm_mm**2, axis=0))
        global_rmse_mm = float(np.sqrt(np.mean(marker_norm_mm**2)))

        summary = dict(
            success=self.success,
            theta=self.theta,
            theta0=self._p0,
            parameters={label: float(value) for label, value in zip(self.parameter_labels, self.theta)},
            parameters_at_bounds=self.parameters_at_bounds(),
            objective=self.objective_value,
            marker_rmse_mm=global_rmse_mm,
            per_frame_marker_rmse_mm=per_frame_rmse_mm,
            marker_residuals_norm_mm=marker_norm_mm,
            max_joint_residual_per_frame=joint_residuals,
            max_rigid_residual_per_frame=rigid_residuals,
        )
        summary.update(self._ellipsoid_summary())
        if "Scapulothoracic" in self.model.joints.joint_names:
            summary["scapulothoracic_angles_deg"] = scapulothoracic_angles(self.model, self.Qopt)
            summary["scapulothoracic_angles_init_deg"] = scapulothoracic_angles(self.model, self.Q_init)
        return summary

    def _ellipsoid_summary(self) -> dict:
        """Ellipsoid parameters under their historical names, when an ellipsoid is being calibrated."""
        from examples._shared.frames import scs_to_natural

        summary = {}
        for parameter, block in self._parameter_slices():
            if isinstance(parameter, EllipsoidSemiAxes):
                summary["semi_axes"] = self.theta[block]
            elif isinstance(parameter, MarkerPosition) and parameter.marker == "ELLIPSOID_CENTER":
                summary["ellipsoid_center_scs"] = self.theta[block]
                summary["ellipsoid_center_natural"] = scs_to_natural(self.model, parameter.segment, self.theta[block])
            elif isinstance(parameter, EllipsoidOrientation):
                summary["ellipsoid_rotation_vector"] = self.theta[block]
                summary["ellipsoid_axes_scs"] = parameter.rotation_matrix(
                    self.model, self.theta[block]
                ) @ parameter.reference_axes(self.model)
        return summary
