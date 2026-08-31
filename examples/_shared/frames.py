"""Frame-construction helpers shared by every model definition."""

import numpy as np


def segment_transformation_matrix(model, segment_name: str) -> np.ndarray:
    """
    The map ``M`` from *natural* to orthonormal *segment* coordinates: ``p_scs = M @ p_nat``.

    The columns of ``M`` are the natural basis vectors ``(u, rp - rd, w)`` written in the orthonormal
    segment frame, so ``M.T @ M`` is the Gram matrix of that basis and ``M`` preserves lengths and
    angles between the two descriptions.

    Note the transpose. ``NaturalSegment.compute_transformation_matrix()`` returns ``M.T``, and
    bionc's own ``add_natural_marker_from_segment_coordinates`` /
    ``add_natural_vector_from_segment_coordinates`` then convert with ``inv(M.T)`` where they need
    ``inv(M)``. For a marker that error is a few millimetres; for the ellipsoid axes it is
    structural -- it turns the principal triad into a skewed one (axis norms 0.93 / 1.31 / 1.00 and
    mutual dot products up to 0.56 on this subject) so the "semi-axis lengths" stop being geometric
    semi-axes. Use :func:`add_vector_from_scs` and :func:`add_marker_from_scs` below rather than the
    bionc helpers wherever the geometry has to be exact.
    """
    from bionc import TransformationMatrixType

    return np.asarray(
        model.segments[segment_name].compute_transformation_matrix(TransformationMatrixType.Buv), dtype=float
    ).T


def natural_to_scs(model, segment_name: str, position_natural) -> np.ndarray:
    """Natural segment coordinates -> orthonormal segment coordinates [m]."""
    M = segment_transformation_matrix(model, segment_name)
    return M @ np.asarray(position_natural, dtype=float).reshape(3)


def scs_to_natural(model, segment_name: str, position_scs) -> np.ndarray:
    """Orthonormal segment coordinates [m] -> natural segment coordinates."""
    M = segment_transformation_matrix(model, segment_name)
    return np.linalg.solve(M, np.asarray(position_scs, dtype=float).reshape(3))


def add_marker_from_scs(model, segment_name: str, name: str, position_scs, **flags):
    """Add a natural marker at a position given in orthonormal segment coordinates [m]."""
    from bionc.bionc_numpy.natural_marker import NaturalMarker

    model.segments[segment_name].add_natural_marker(
        NaturalMarker(
            name=name,
            parent_name=segment_name,
            position=scs_to_natural(model, segment_name, position_scs),
            **{"is_technical": False, "is_anatomical": True, **flags},
        )
    )


def add_vector_from_scs(model, segment_name: str, name: str, direction_scs):
    """Add a natural vector for a direction given in orthonormal segment coordinates (normalised)."""
    from bionc.bionc_numpy.natural_marker import SegmentNaturalVector

    direction = np.asarray(direction_scs, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)
    model.segments[segment_name].add_natural_vector(
        SegmentNaturalVector(
            name=name, parent_name=segment_name, direction=scs_to_natural(model, segment_name, direction)
        )
    )


def u_thorax(ij: np.ndarray, centijc7: np.ndarray, centpxt8: np.ndarray) -> np.ndarray:
    """
    Postero-anterior axis ``u`` of the thorax segment.

    The thorax ``u`` axis relies on the segment ``w`` axis, so the whole sequence of
    computations has to be redone here rather than reusing the generic axis templates:

        u = (rp - rd) x w / ||(rp - rd) x w||

    with ``rp`` the midpoint of ``IJ`` and the ``C7`` centroid, ``rd`` the ``PX``/``T5``
    centroid, and ``w`` the plane normal of the three input markers.

    Parameters are ``(4 x Nframes)`` marker trajectories (homogeneous rows). Returns a
    ``(4 x Nframes)`` array.
    """
    rp = (ij + centijc7) / 2
    rd = centpxt8

    v = rp - rd

    w = np.ones((4, ij.shape[1]))
    cross_product = np.ones((4, ij.shape[1]))

    for i, (mk1i, mk2i, mk3i) in enumerate(zip(ij.T, centijc7.T, centpxt8.T)):
        v1 = mk2i[:3] - mk1i[:3]
        v2 = mk3i[:3] - mk1i[:3]
        w[:3, i] = np.cross(v1, v2) / np.linalg.norm(np.cross(v1, v2))
        temp = np.cross(v[:3, i], w[:3, i])
        cross_product[:3, i] = temp / np.linalg.norm(temp)

    return cross_product
