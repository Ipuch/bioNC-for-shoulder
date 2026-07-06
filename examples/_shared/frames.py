"""Frame-construction helpers shared by every model definition."""

import numpy as np


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
