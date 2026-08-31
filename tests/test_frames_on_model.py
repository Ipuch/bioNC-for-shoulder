"""The natural <-> segment-coordinate conversions, checked against a real built model.

``examples/_shared/frames.py`` exists because bionc's own ``add_natural_*_from_segment_coordinates``
convert with the transpose of the right matrix, which skews the ellipsoid's principal triad. These
tests pin the property that motivated it: a direction written in segment coordinates must come back
orthonormal, and a round trip must be the identity.
"""

import numpy as np
import pytest

from examples._shared.frames import (
    add_marker_from_scs,
    add_vector_from_scs,
    natural_to_scs,
    scs_to_natural,
    segment_transformation_matrix,
)


@pytest.mark.parametrize("segment", ["THORAX", "RSCAPULA", "RHUMERUS"])
def test_scs_natural_round_trip(clinical_model, segment):
    """``natural_to_scs`` and ``scs_to_natural`` are inverses, on every segment of the chain."""
    generator = np.random.default_rng(0)
    for point_scs in generator.normal(scale=0.1, size=(8, 3)):
        recovered = natural_to_scs(clinical_model, segment, scs_to_natural(clinical_model, segment, point_scs))
        np.testing.assert_allclose(recovered, point_scs, atol=1e-12)


@pytest.mark.parametrize("segment", ["THORAX", "RSCAPULA"])
def test_transformation_matrix_preserves_lengths_and_angles(clinical_model, segment):
    """
    ``M`` maps natural to *orthonormal* segment coordinates, so distances measured in segment
    coordinates are real distances in metres. That is what lets the ellipsoid's ``(a, b, c)`` be
    geometric semi-axes rather than dimensionless fractions of a segment.
    """
    M = segment_transformation_matrix(clinical_model, segment)
    assert M.shape == (3, 3)
    assert abs(np.linalg.det(M)) > 1e-9

    generator = np.random.default_rng(1)
    first, second = generator.normal(size=3), generator.normal(size=3)
    natural_first = scs_to_natural(clinical_model, segment, first)
    natural_second = scs_to_natural(clinical_model, segment, second)

    np.testing.assert_allclose(np.linalg.norm(M @ natural_first), np.linalg.norm(first), atol=1e-12)
    np.testing.assert_allclose(np.dot(M @ natural_first, M @ natural_second), np.dot(first, second), atol=1e-12)


def test_added_vectors_are_orthonormal_in_segment_coordinates(clinical_model):
    """
    The regression that motivated these helpers: three orthonormal directions written in segment
    coordinates must still be orthonormal when read back. Converting with the transpose -- as the
    bionc helpers do -- returns a skewed triad, and the "semi-axis lengths" stop being semi-axes.
    """
    for name, direction in zip(("AX_A", "AX_B", "AX_C"), np.eye(3)):
        add_vector_from_scs(clinical_model, "THORAX", name, direction)

    thorax = clinical_model.segments["THORAX"]
    axes = np.column_stack(
        [natural_to_scs(clinical_model, "THORAX", thorax.vector_from_name(name).position) for name in ("AX_A", "AX_B", "AX_C")]
    )

    np.testing.assert_allclose(np.linalg.norm(axes, axis=0), 1.0, atol=1e-9)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-9)


def test_added_marker_lands_where_it_was_asked_to(clinical_model):
    """A marker placed at a segment-frame position reads back at that position."""
    position = np.array([0.03, -0.05, 0.02])
    add_marker_from_scs(clinical_model, "THORAX", "PROBE", position)

    marker = clinical_model.segments["THORAX"].marker_from_name("PROBE")
    np.testing.assert_allclose(natural_to_scs(clinical_model, "THORAX", marker.position), position, atol=1e-12)
