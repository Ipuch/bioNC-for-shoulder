"""The glenohumeral joint built on the two dedicated centres.

Two things are worth pinning. The results tree a run writes into is chosen by name, and getting
that wrong would silently mix two calibrations into one directory. And the constant-length joint
has a floor under its length that is not a matter of taste: at zero the constraint Jacobian
``2 (p_parent - p_child)^T N`` vanishes, so a zero-length "constant length" is a singular
constraint rather than a tight one, and it has to be refused rather than solved.
"""

import numpy as np
import pytest

from bionc import NaturalCoordinates

from examples.clinical.model import add_glenohumeral_centres
from studies import RESULTS_ROOTS, results_root


class TestResultsRoot:
    """One tree per glenohumeral model, so two runs never overwrite each other."""

    def test_each_model_gets_its_own_directory(self):
        spherical, constant = results_root("spherical"), results_root("constant_length")
        assert spherical.name == "results"
        assert constant.name == "results_gh_constant"
        assert spherical.parent == constant.parent

    def test_an_unknown_model_is_refused_by_name(self):
        with pytest.raises(ValueError, match="constant_length"):
            results_root("ball_and_socket")

    def test_the_default_is_one_of_the_known_models(self):
        assert results_root().name in RESULTS_ROOTS.values()


class TestConstantLengthGlenohumeral:
    """The joint on ``GH_GLENOID`` / ``GH_HEAD``, which is the only place it is not degenerate."""

    def test_a_missing_length_is_refused(self, clinical_model):
        with pytest.raises(ValueError, match="constant-length"):
            add_glenohumeral_centres(clinical_model, constraint="constant_length")

    def test_a_zero_length_is_refused_as_singular(self, clinical_model):
        with pytest.raises(ValueError, match="singular"):
            add_glenohumeral_centres(clinical_model, constraint="constant_length", length=0.0)

    def test_an_unknown_constraint_is_refused(self, clinical_model):
        with pytest.raises(ValueError, match="spherical"):
            add_glenohumeral_centres(clinical_model, constraint="hinge")

    def test_it_costs_one_constraint_where_spherical_costs_three(self, clinical_model):
        spherical = add_glenohumeral_centres(_fresh(clinical_model))
        constant = add_glenohumeral_centres(_fresh(clinical_model), constraint="constant_length", length=0.012)

        assert spherical.joints["Glenohumeral"].nb_constraints == 3
        assert constant.joints["Glenohumeral"].nb_constraints == 1

    def test_the_constraint_vanishes_when_the_centres_are_exactly_that_far_apart(self, clinical_model):
        """
        The constraint is ``||p_glenoid - p_head||^2 - L^2``. Move the humeral-head centre a known
        distance along one scapula-frame axis and the joint should read zero at that distance and
        nowhere else -- which is what says the length is a real radius in metres rather than an
        uninterpreted number.
        """
        length = 0.012
        model = add_glenohumeral_centres(_fresh(clinical_model), constraint="constant_length", length=length)
        joint = model.joints["Glenohumeral"]

        Q = _reference_Q(model)
        offset = _centre_separation(model, Q)

        assert abs(offset - length) > 1e-4, "the fixture already sits at the tested length; pick another"
        joint.length = offset
        residual = joint.constraint(
            NaturalCoordinates(Q).vector(model.segments["RSCAPULA"].index),
            NaturalCoordinates(Q).vector(model.segments["RHUMERUS"].index),
        )
        assert float(np.asarray(residual).reshape(-1)[0]) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------------------- helpers
def _fresh(clinical_model):
    """
    A model that does not yet carry the dedicated centres.

    ``clinical_model`` is session-scoped and ``add_glenohumeral_centres`` mutates in place, so each
    test that adds the centres needs its own copy.
    """
    from copy import deepcopy

    return deepcopy(clinical_model)


def _reference_Q(model) -> np.ndarray:
    """One frame of natural coordinates: each segment in its own reference configuration."""
    return np.concatenate(
        [np.array([1.0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 1.0]) for _ in model.segments_no_ground]
    ).reshape(-1, 1)[:, 0]


def _centre_separation(model, Q: np.ndarray) -> float:
    """Distance [m] between the two glenohumeral centres in that configuration."""
    Q = NaturalCoordinates(Q)
    joint = model.joints["Glenohumeral"]
    glenoid = joint.parent_point.position_in_global(Q.vector(model.segments["RSCAPULA"].index))
    head = joint.child_point.position_in_global(Q.vector(model.segments["RHUMERUS"].index))
    return float(np.linalg.norm(np.asarray(glenoid).reshape(3) - np.asarray(head).reshape(3)))
