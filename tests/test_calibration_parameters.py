"""The CalibrationParameter contract.

Every block is bound into an MX model, solved for, and then written back with ``apply``. The
contract that matters and was easiest to break is that ``apply`` writes an *absolute* value: three
of the four blocks obviously do, and the fourth solves for a rotation *increment*, so it has to
capture what the increment is measured from rather than re-reading the model it has just written.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from examples._shared.frames import add_marker_from_scs, add_vector_from_scs, natural_to_scs, rodrigues_matrix
from studies.kinematic_calibration import (
    EllipsoidOrientation,
    EllipsoidSemiAxes,
    JointLength,
    MarkerPosition,
)


class TestSummary:
    """Each block reports its own entries; the engine does not type-switch."""

    def test_base_blocks_report_nothing_by_default(self):
        assert JointLength("Clavicle").summary(None, np.array([0.15])) == {}

    def test_marker_position_is_silent_without_report_as(self):
        marker = MarkerPosition("THORAX", "SOMETHING", targets=())
        assert marker.summary(None, np.zeros(3)) == {}

    def test_semi_axes_report_under_their_historical_name(self):
        values = np.array([0.1, 0.2, 0.3])
        assert EllipsoidSemiAxes().summary(None, values)["semi_axes"] is values


class TestJointLengthBounds:
    """A length is boxed either as a fraction of its warm start or absolutely, never both."""

    def test_scale_bounds_are_relative_to_the_warm_start(self):
        parameter = JointLength("Clavicle", scale_bounds=(0.5, 1.5))
        lower, upper = parameter.bounds(_model_with_joint_length("Clavicle", 0.150))
        np.testing.assert_allclose([lower[0], upper[0]], [0.075, 0.225])

    def test_absolute_bounds_win_and_ignore_the_warm_start(self):
        """
        The glenohumeral radius warm-starts at a few millimetres, where scaling says nothing useful
        and the lower end would sit close enough to zero to make the constraint singular.
        """
        parameter = JointLength("Glenohumeral", bounds=(0.001, 0.050))
        for length in (0.004, 0.020):
            lower, upper = parameter.bounds(_model_with_joint_length("Glenohumeral", length))
            np.testing.assert_allclose([lower[0], upper[0]], [0.001, 0.050])


class TestEllipsoidOrientation:
    """The one block whose unknown is relative rather than absolute."""

    def test_reference_axes_are_captured_once(self, clinical_model):
        """
        The regression: ``apply`` must be idempotent like every other block. It rotates the
        reference axes, so if those were re-read from the model after a write, a second ``apply``
        would measure from the already-rotated triad and compound the two rotations.
        """
        model = _model_with_ellipsoid_axes(clinical_model)
        parameter = EllipsoidOrientation("Scapulothoracic", "THORAX")
        before = parameter.reference_axes(model).copy()

        values = np.array([0.10, -0.05, 0.02])
        parameter.apply(model, values)
        after_once = _axes_in_scs(model)

        parameter.apply(model, values)  # the same answer written a second time
        after_twice = _axes_in_scs(model)

        np.testing.assert_allclose(after_twice, after_once, atol=1e-12)
        np.testing.assert_allclose(parameter.reference_axes(model), before, atol=1e-12)

    def test_apply_rotates_the_reference_triad_by_the_solved_vector(self, clinical_model):
        model = _model_with_ellipsoid_axes(clinical_model)
        parameter = EllipsoidOrientation("Scapulothoracic", "THORAX")
        reference = parameter.reference_axes(model).copy()

        values = np.array([0.2, 0.1, -0.15])
        parameter.apply(model, values)

        np.testing.assert_allclose(_axes_in_scs(model), rodrigues_matrix(values) @ reference, atol=1e-9)

    def test_a_zero_increment_leaves_the_axes_alone(self, clinical_model):
        model = _model_with_ellipsoid_axes(clinical_model)
        parameter = EllipsoidOrientation("Scapulothoracic", "THORAX")
        before = _axes_in_scs(model)

        parameter.apply(model, np.zeros(3))
        np.testing.assert_allclose(_axes_in_scs(model), before, atol=1e-12)


# --------------------------------------------------------------------------------------- helpers
AXIS_NAMES = ("AXIS_A", "AXIS_B", "AXIS_C")


def _model_with_joint_length(joint: str, length: float):
    """The least model ``JointLength`` reads: one joint carrying one length."""
    return SimpleNamespace(joints={joint: SimpleNamespace(length=length)})


def _model_with_ellipsoid_axes(clinical_model):
    """The session model with ellipsoid geometry attached, and a joint object exposing the axes."""
    model = clinical_model
    if not any(name in AXIS_NAMES for name in _vector_names(model)):
        add_marker_from_scs(model, "THORAX", "ELLIPSOID_CENTER", np.zeros(3))
        for name, direction in zip(AXIS_NAMES, np.eye(3)):
            add_vector_from_scs(model, "THORAX", name, direction)

    joint = model.joints["Scapulothoracic"]
    if not hasattr(joint, "ellipsoid_axes"):
        thorax = model.segments["THORAX"]
        joint.ellipsoid_axes = [thorax.vector_from_name(name) for name in AXIS_NAMES]
    return model


def _vector_names(model):
    thorax = model.segments["THORAX"]
    return [name for name in AXIS_NAMES if _has_vector(thorax, name)]


def _has_vector(segment, name) -> bool:
    try:
        segment.vector_from_name(name)
    except Exception:
        return False
    return True


def _axes_in_scs(model) -> np.ndarray:
    joint = model.joints["Scapulothoracic"]
    return np.column_stack([natural_to_scs(model, "THORAX", axis.position) for axis in joint.ellipsoid_axes])
