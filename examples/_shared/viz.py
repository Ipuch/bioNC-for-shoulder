"""pyorerun visualization boilerplate shared by the build/IK examples."""

import numpy as np

from examples._shared.ik import load_markers


def animate_model(model, Q, c3d_filename: str, *, show_natural_vectors: bool = False) -> None:
    """
    Animate a bionc model in pyorerun with the experimental markers overlaid.

    Works for any natural coordinates ``Q`` (nb_dof x nb_frames): pass ``Q_from_markers``
    to inspect the calibrated model on the raw data, or the IK solution ``Qopt``.

    Parameters
    ----------
    model
        The bionc (numpy) model.
    Q
        Natural coordinates to animate, shape ``(12 * nb_segments, nb_frames)``.
    c3d_filename
        The trial to read the tracked markers from.
    show_natural_vectors
        If True, also draw the segments' natural base vectors (u, v, w) -- useful to
        visualise the frame conventions while learning.
    """
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun, PyoMarkers

    Q = np.asarray(Q)
    markers = load_markers(model, c3d_filename)
    pyomarkers = PyoMarkers(data=markers, marker_names=model.marker_names_technical)
    pyomarkers.show_labels = False

    prr = PhaseRerun(t_span=np.linspace(0, 1, Q.shape[-1]))
    prr.add_animated_model(BioncModelNoMesh(model), Q, tracked_markers=pyomarkers)
    if show_natural_vectors:
        from bionc.vizualization.pyorerun_natural_vectors import add_natural_vectors

        add_natural_vectors(prr, model, Q, scale_u=0.1, scale_v=1.0, scale_w=0.1)
    prr.rerun()
