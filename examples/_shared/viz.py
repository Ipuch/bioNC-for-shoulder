"""pyorerun visualization boilerplate shared by the examples and the calibration studies."""

import numpy as np

from examples._shared.ik import load_markers


def named_bionc_model(model, display_name: str, options=None):
    """
    ``BioncModelNoMesh`` with a per-instance name.

    The base class hard-codes one name, so two models in the same recording would collide into a
    single rerun entity. Giving each its own name is what makes them separately toggleable.
    """
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh

    class _NamedBioncModel(BioncModelNoMesh):
        def __init__(self, model, display_name, options=None):
            super().__init__(model, options)
            self._display_name = display_name

        @property
        def name(self):
            return self._display_name

    return _NamedBioncModel(model, display_name, options)


def _rotation_to_xyzw(rotation: np.ndarray) -> list:
    """3x3 rotation matrix -> quaternion [x, y, z, w] (Shepperd's method)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2
        x, y, z, w = (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        x, y, z, w = 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        x, y, z, w = (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        x, y, z, w = (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s
    q = np.array([x, y, z, w])
    return (q / np.linalg.norm(q)).tolist()


def _axes_to_xyzw(axes: np.ndarray) -> list:
    """Nearest rotation (SVD) to the (possibly slightly non-orthogonal) axis columns, as xyzw."""
    u, _, vt = np.linalg.svd(axes)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u = u.copy()
        u[:, -1] *= -1
        rotation = u @ vt
    return _rotation_to_xyzw(rotation)


def _plane_triangle(center: np.ndarray, normal: np.ndarray, size: float = 0.12) -> list:
    """Three vertices of an equilateral triangle centred on ``center``, lying in the plane ``normal``."""
    n = normal / np.linalg.norm(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return [
        (center + size * e1).tolist(),
        (center + size * (-0.5 * e1 + 0.8660254 * e2)).tolist(),
        (center + size * (-0.5 * e1 - 0.8660254 * e2)).tolist(),
    ]


def overlay_ellipsoids(ellipsoids, t_span: np.ndarray) -> None:
    """
    Draw calibrated scapulothoracic ellipsoids into the *live* rerun recording.

    Call this **after** ``PhaseRerun.rerun()``: it logs onto the same ``stable_time`` timeline that
    the animation already created, so the ellipsoid moves with the thorax and the contact point with
    the scapula instead of sitting still in the world.

    Each entry of ``ellipsoids`` is
    ``(entity_name, model, Q, semi_axes, centre_in_natural_coordinates, rgba)``. The contact point is
    always drawn; the tangent (ELLIPSOID_ON_PLANE) joint additionally carries a plane, which is drawn
    as a small triangle, while the one-point joint has none.
    """
    import rerun as rr

    from bionc.bionc_numpy.natural_vector import NaturalVector

    for entity, model, Qopt, semi_axes, center_natural, color in ellipsoids:
        solid_color = (color[0], color[1], color[2], 255)  # opaque marker vs translucent surfaces

        radii = np.asarray(semi_axes, dtype=float)
        center_interp = np.asarray(NaturalVector(np.asarray(center_natural, dtype=float)).interpolate(), dtype=float)
        joint = model.joints["Scapulothoracic"]
        axes_interp = [np.asarray(axis.interpolation_matrix, dtype=float) for axis in joint.ellipsoid_axes]
        thorax_slice = slice(12 * model.segments["THORAX"].index, 12 * model.segments["THORAX"].index + 12)
        scapula_slice = slice(12 * model.segments["RSCAPULA"].index, 12 * model.segments["RSCAPULA"].index + 12)

        # the tangent joint carries a plane (point + normal); the one-point joint only a contact point
        plane_point = getattr(joint, "plane_point", None)
        if plane_point is not None:
            contact_interp = np.asarray(plane_point.interpolation_matrix, dtype=float)
            normal_interp = np.asarray(joint.plane_normal.interpolation_matrix, dtype=float)
        else:
            contact_interp = np.asarray(joint.contact_point.interpolation_matrix, dtype=float)
            normal_interp = None

        for k, t in enumerate(t_span):
            q_thorax = np.asarray(Qopt[thorax_slice, k], dtype=float).reshape(-1)
            q_scapula = np.asarray(Qopt[scapula_slice, k], dtype=float).reshape(-1)
            rr.set_time("stable_time", duration=float(t))

            center = center_interp @ q_thorax
            axes = np.column_stack([interp @ q_thorax for interp in axes_interp])
            axes /= np.linalg.norm(axes, axis=0, keepdims=True)
            rr.log(
                entity,
                rr.Ellipsoids3D(
                    half_sizes=[radii.tolist()],
                    centers=[center.tolist()],
                    quaternions=[rr.Quaternion(xyzw=_axes_to_xyzw(axes))],
                    colors=[color],
                    fill_mode="MajorWireframe",
                ),
            )

            contact = contact_interp @ q_scapula
            rr.log(f"{entity}/contact_point", rr.Points3D([contact.tolist()], colors=[solid_color], radii=[0.01]))

            if normal_interp is not None:
                rr.log(
                    f"{entity}/plane",
                    rr.Mesh3D(
                        vertex_positions=_plane_triangle(contact, normal_interp @ q_scapula),
                        triangle_indices=[[0, 1, 2]],
                        vertex_colors=[color, color, color],
                    ),
                )


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
