"""The rotation conversions, which exist in more than one form and must agree.

The calibration writes a Rodrigues rotation twice -- once symbolically for CasADi, once in numpy
to apply the solved value back onto a model. If those two ever drift apart the optimiser would be
solving for one rotation and the model would receive another, silently. Nothing else in the repo
would catch that, so it is pinned here.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples._shared.frames import rodrigues_matrix
from examples._shared.viz import _axes_to_xyzw, _rotation_to_xyzw


def rotation_vectors():
    """A spread of axis-angle vectors, including the degenerate one at zero."""
    generator = np.random.default_rng(0)
    vectors = [np.zeros(3), np.array([1e-13, 0.0, 0.0]), np.array([np.pi, 0.0, 0.0])]
    vectors += [generator.normal(size=3) * scale for scale in (1e-6, 0.01, 0.3, 1.0, 2.5) for _ in range(4)]
    return vectors


@pytest.mark.parametrize("vector", rotation_vectors())
def test_rodrigues_matches_scipy(vector):
    """The numpy Rodrigues is the rotation scipy says it is."""
    np.testing.assert_allclose(rodrigues_matrix(vector), Rotation.from_rotvec(vector).as_matrix(), atol=1e-12)


@pytest.mark.parametrize("vector", rotation_vectors())
def test_casadi_rodrigues_matches_numpy(vector):
    """The MX Rodrigues the NLP differentiates equals the numpy one that writes the answer back."""
    from casadi import DM, MX, Function

    from studies.kinematic_calibration import rodrigues

    symbol = MX.sym("r", 3)
    evaluate = Function("rodrigues", [symbol], [rodrigues(symbol)])
    symbolic = np.array(evaluate(DM(vector)))

    # the MX form regularises theta with a +1e-12 under the square root, so it cannot be exact at 0
    np.testing.assert_allclose(symbolic, rodrigues_matrix(vector), atol=1e-6)


@pytest.mark.parametrize("vector", rotation_vectors())
def test_rotation_to_quaternion_round_trip(vector):
    """All four Shepperd branches of ``_rotation_to_xyzw`` recover the rotation they were given."""
    rotation = rodrigues_matrix(vector)
    recovered = Rotation.from_quat(_rotation_to_xyzw(rotation)).as_matrix()
    np.testing.assert_allclose(recovered, rotation, atol=1e-9)


def test_rotation_to_quaternion_covers_every_branch():
    """The branch picked depends on which diagonal entry dominates; exercise all four."""
    # trace > 0; then each of the three "largest diagonal element" cases in turn
    rotations = [
        np.eye(3),
        Rotation.from_rotvec([np.pi, 0, 0]).as_matrix(),
        Rotation.from_rotvec([0, np.pi, 0]).as_matrix(),
        Rotation.from_rotvec([0, 0, np.pi]).as_matrix(),
    ]
    for rotation in rotations:
        quaternion = _rotation_to_xyzw(rotation)
        assert np.isfinite(quaternion).all()
        np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
        np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), rotation, atol=1e-9)


def test_axes_to_xyzw_orthonormalises_a_skewed_triad():
    """``_axes_to_xyzw`` takes the nearest rotation, so slightly non-orthogonal axes still work."""
    axes = rodrigues_matrix([0.2, -0.4, 0.1])
    skewed = axes + 0.01 * np.array([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])

    recovered = Rotation.from_quat(_axes_to_xyzw(skewed)).as_matrix()
    np.testing.assert_allclose(recovered @ recovered.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(recovered) > 0
    assert np.linalg.norm(recovered - axes) < 0.05


def test_axes_to_xyzw_rejects_a_reflection():
    """A left-handed triad must come back as a proper rotation, not a reflection."""
    axes = rodrigues_matrix([0.3, 0.1, -0.2]).copy()
    axes[:, 2] *= -1  # determinant now negative

    recovered = Rotation.from_quat(_axes_to_xyzw(axes)).as_matrix()
    assert np.linalg.det(recovered) == pytest.approx(1.0, abs=1e-9)
