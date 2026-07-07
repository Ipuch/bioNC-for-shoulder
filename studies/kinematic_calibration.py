import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root on the path

import numpy as np
from casadi import MX, Function, dot, horzcat, nlpsol, vertcat

from bionc import NaturalCoordinates as NaturalCoordinatesNumpy
from bionc.bionc_casadi import NaturalCoordinates, SegmentNaturalCoordinates
from bionc.bionc_casadi.natural_marker import NaturalMarker
from bionc.bionc_casadi.natural_vector import NaturalVector
from bionc.utils.casadi_utils import sarrus


def scapulothoracic_angles(model, Q: np.ndarray, joint_name: str = "Scapulothoracic") -> np.ndarray:
    """Joint Euler angles [3 x nb_frames] in degrees for a model and its natural coordinates ``Q``."""
    Q = np.asarray(Q)
    column = list(model.joints.joint_names).index(joint_name)
    angles = np.zeros((3, Q.shape[1]))
    for f in range(Q.shape[1]):
        angles[:, f] = model.natural_coordinates_to_joint_angles(NaturalCoordinatesNumpy(Q[:, f]))[:, column]
    return np.degrees(angles)


class KinematicCalibration:
    """
    All-frames marker inverse kinematics with the scapulothoracic ellipsoid parameters
    (semi-axes + centre location in the thorax) as shared optimisation variables.

    Parameters
    ----------
    model
        A bionc (numpy) model whose scapulothoracic joint is an ellipsoid joint
        (``ELLIPSOID_ON_PLANE`` or ``POINT_ON_ELLIPSOID``); both expose ``semi_axis_lengths``
        and ``ellipsoid_center``.
    markers
        Experimental technical markers, ``(3, nb_technical_markers, nb_frames)`` in metres.
    ellipsoid_joint
        Name of the ellipsoid joint to calibrate.
    active_direct_frame_constraints
        If True, add per-frame constraints forcing a positive ``det([u, v, w])`` on every segment.
    semi_axis_bounds
        ``(lower, upper)`` bounds [m] on each ellipsoid semi-axis.
    center_half_ranges
        Half-width of the box (in thorax natural coordinates) the ellipsoid centre may move within,
        around its warm-started value.
    """

    def __init__(
        self,
        model,
        markers: np.ndarray,
        *,
        ellipsoid_joint: str = "Scapulothoracic",
        active_direct_frame_constraints: bool = True,
        semi_axis_bounds: tuple[float, float] = (0.03, 0.50),
        center_half_ranges: float = (0.3, 0.3, 0.25) ,
    ):
        self.model = model
        self.markers = np.asarray(markers)
        self.nb_frames = self.markers.shape[2]
        self.nb_markers = self.markers.shape[1]
        self.nb_segments = model.nb_segments
        self.ellipsoid_joint = ellipsoid_joint
        self.active_direct_frame_constraints = active_direct_frame_constraints

        self._model_mx = model.to_mx()
        self._p_sym, self._p0, self._p_lb, self._p_ub = self._make_ellipsoid_symbolic(
            semi_axis_bounds, center_half_ranges
        )
        self._build_nlp()

        self.Qopt = None
        self.theta = None
        self.success = None
        self.objective_value = None

    # ------------------------------------------------------------------ setup
    def _make_ellipsoid_symbolic(self, semi_axis_bounds, center_half_ranges):
        """Replace the ellipsoid joint's semi-axes and centre by MX symbols; return (p_sym, p0, lb, ub)."""
        joint_np = self.model.joints[self.ellipsoid_joint]
        a0, b0, c0 = (float(length) for length in joint_np.semi_axis_lengths)
        center0 = np.asarray(joint_np.ellipsoid_center.position, dtype=float).reshape(3)

        joint_mx = self._model_mx.joints[self.ellipsoid_joint]
        p_semi = MX.sym("semi_axes", 3)
        p_center = MX.sym("ellipsoid_center", 3)

        joint_mx.semi_axis_lengths = (p_semi[0], p_semi[1], p_semi[2])
        joint_mx.ellipsoid_center = NaturalMarker(
            name=joint_mx.ellipsoid_center.name,
            parent_name=joint_mx.ellipsoid_center.parent_name,
            position=NaturalVector(p_center),
            is_technical=False,
            is_anatomical=True,
        )

        p_sym = vertcat(p_semi, p_center)
        p0 = np.concatenate([[a0, b0, c0], center0])
        lb_center = center0 - np.array(center_half_ranges)
        lb_center[2] = 0
        lb_ellispoid_semi_axis = np.array([0.15, 0.10, 0.05])
        ub_ellispoid_semi_axis = np.array([0.15, 0.30, 0.15])
        lb = np.concatenate([lb_ellispoid_semi_axis, lb_center])
        ub = np.concatenate([ub_ellispoid_semi_axis, center0 + np.array(center_half_ranges)])
        return p_sym, p0, lb, ub

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

        x = vertcat(vertcat(*Q_frames), self._p_sym)
        self._nlp = {"x": x, "f": objective, "g": vertcat(*constraints)}
        self._lbg = np.array(lbg)
        self._ubg = np.array(ubg)

    # ------------------------------------------------------------------ solve
    @staticmethod
    def _default_options() -> dict:
        return {
            "ipopt.hessian_approximation": "exact",  # exact Hessian, as requested
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

    def sol(self) -> dict:
        """
        Frame-by-frame solution summary (after :meth:`solve`).

        Returns a dict with the calibrated parameters and per-frame residuals:
        semi-axes [m], ellipsoid centre (thorax natural coords), marker RMSE [mm] (global and
        per-frame), joint/tangency residual per frame, and the scapulothoracic Euler angles [deg].
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

        return dict(
            success=self.success,
            theta=self.theta,
            semi_axes=self.theta[:3],
            ellipsoid_center_natural=self.theta[3:],
            objective=self.objective_value,
            marker_rmse_mm=global_rmse_mm,
            per_frame_marker_rmse_mm=per_frame_rmse_mm,
            marker_residuals_norm_mm=marker_norm_mm,
            max_joint_residual_per_frame=joint_residuals,
            max_rigid_residual_per_frame=rigid_residuals,
            scapulothoracic_angles_deg=scapulothoracic_angles(self.model, self.Qopt, self.ellipsoid_joint),
            scapulothoracic_angles_init_deg=scapulothoracic_angles(self.model, self.Q_init, self.ellipsoid_joint),
        )

