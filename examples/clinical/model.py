"""Upper-limb model for the clinical marker set.

Same structure as the Henninger model (segments defined once, joints selected by keyword),
plus two clinical specificities:

* ``marker_set`` chooses which markers are *technical* (tracked by the IK): the skin
  ``Cluster_*`` markers, the anatomical bone landmarks, or both;
* :func:`build_ellipsoid_model` replaces the free scapulothoracic joint by a tangent
  ellipsoid-on-plane joint (Naaim 2016/2017), used by the calibration study.

Every builder takes a ``source`` that is either a c3d path or any ``bionc`` ``Data`` object -- pass
an :class:`~examples._shared.c3d_data.MultiC3dData` to calibrate the segment geometry and the
data-driven joint lengths over a whole session instead of a single trial.

Joint chain (proximal -> distal):
    GROUND --Freeflyer--> THORAX --[Clavicle]--> / --Scapulothoracic--> RSCAPULA --Glenohumeral--> RHUMERUS

``include_humerus=False`` stops the chain at the scapula, for the thorax-and-scapula-only model the
scapulothoracic ellipsoid is calibrated on.
"""

from pathlib import Path

import numpy as np

from bionc import (
    AxisTemplate,
    AxisFunctionTemplate,
    BiomechanicalModelTemplate,
    MarkerTemplate,
    SegmentTemplate,
    NaturalSegmentTemplate,
    BiomechanicalModel,
    JointType,
    EulerSequence,
    TransformationMatrixType,
    C3dData,
)

from examples._shared.frames import (
    add_marker_from_scs,
    add_vector_from_scs,
    natural_to_scs,
    scs_to_natural,
    u_thorax,
)
from examples._shared.ik import load_markers

MARKER_SETS = ("cluster", "anatomical", "both")
GLENOHUMERAL_CONSTRAINTS = ("free", "spherical")
ELLIPSOID_JOINTS = ("tangent", "point")

# dedicated, calibratable glenohumeral centres (see add_glenohumeral_centres)
GH_GLENOID = "GH_GLENOID"
GH_HEAD = "GH_HEAD"


def _build_segments(
    model: BiomechanicalModelTemplate, *, use_cluster: bool, use_anatomical: bool, include_humerus: bool = True
) -> None:
    """Add the THORAX, RSCAPULA and (optionally) RHUMERUS segments and their markers (clinical names)."""
    u_axis_thorax = lambda m, bio: u_thorax(m["SJN"], m["CV7"], m["TV8"])

    model["THORAX"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=u_axis_thorax),
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "CV7", "SJN"),
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "SXS", "TV8"),
            w_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "SJN", "CV7", "TV8")),
        )
    )
    model["THORAX"].add_marker(MarkerTemplate(name="SJN", parent_name="THORAX", is_technical=True, is_anatomical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="SXS", parent_name="THORAX", is_technical=True, is_anatomical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="TV8", parent_name="THORAX", is_technical=True, is_anatomical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="CV7", parent_name="THORAX", is_technical=True, is_anatomical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="SME", parent_name="THORAX", is_technical=False, is_anatomical=False))
    model["THORAX"].add_marker(MarkerTemplate(name="RCAS", parent_name="THORAX", is_technical=False, is_anatomical=False))

    model["RSCAPULA"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "RSAA", "RSIA", "RSRS")),
            proximal_point="RSAA",
            distal_point="RSIA",
            w_axis=AxisTemplate(start="RSRS", end="RSAA"),
        )
    )
    model["RSCAPULA"].add_marker(MarkerTemplate(name="RSAA", parent_name="RSCAPULA", is_technical=use_anatomical))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="RSIA", parent_name="RSCAPULA", is_technical=use_anatomical))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="RSRS", parent_name="RSCAPULA", is_technical=use_anatomical))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="RCAJ", parent_name="RSCAPULA", is_technical=use_anatomical))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="Cluster_RS_01", parent_name="RSCAPULA", is_technical=use_cluster))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="Cluster_RS_02", parent_name="RSCAPULA", is_technical=use_cluster))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="Cluster_RS_03", parent_name="RSCAPULA", is_technical=use_cluster))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="RGJC", parent_name="RSCAPULA", is_technical=use_anatomical))

    if not include_humerus:
        return

    model["RHUMERUS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "RHLE", "RHME", "RGJC")),
            proximal_point="RGJC",
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "RHLE", "RHME"),
            w_axis=AxisTemplate(start="RHME", end="RHLE"),
        )
    )
    model["RHUMERUS"].add_marker(MarkerTemplate(name="RGJC", parent_name="RHUMERUS", is_technical=use_anatomical))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="RHME", parent_name="RHUMERUS", is_technical=use_anatomical))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="RHLE", parent_name="RHUMERUS", is_technical=use_anatomical))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="Cluster_RA_01", parent_name="RHUMERUS", is_technical=use_cluster))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="Cluster_RA_02", parent_name="RHUMERUS", is_technical=use_cluster))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="Cluster_RA_03", parent_name="RHUMERUS", is_technical=use_cluster))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="Cluster_RA_04", parent_name="RHUMERUS", is_technical=use_cluster))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="Cluster_RA_05", parent_name="RHUMERUS", is_technical=use_cluster))


def _add_glenohumeral_joint(model: BiomechanicalModelTemplate, glenohumeral: str) -> None:
    """Add the RSCAPULA -> RHUMERUS joint. RGJC is shared by both segments (glenohumeral centre)."""
    if glenohumeral not in GLENOHUMERAL_CONSTRAINTS:
        raise ValueError(f"glenohumeral must be one of {GLENOHUMERAL_CONSTRAINTS}, got {glenohumeral!r}")

    common = dict(
        name="Glenohumeral",
        parent="RSCAPULA",
        child="RHUMERUS",
        projection_basis=EulerSequence.YXY,
        parent_basis=TransformationMatrixType.Bwu,
        child_basis=TransformationMatrixType.Bvu,
    )
    if glenohumeral == "free":
        model.add_joint(joint_type=JointType.FREE, **common)
    else:  # "spherical": the two RGJC points (scapula & humerus) are made coincident
        model.add_joint(joint_type=JointType.SPHERICAL, parent_point="RGJC", child_point="RGJC", **common)


def _as_data(source, first_frame: int, last_frame: int):
    """A c3d path becomes a :class:`~bionc.C3dData`; any ``Data`` object is passed through."""
    if isinstance(source, (str, Path)):
        return C3dData(str(source), first_frame=first_frame, last_frame=last_frame)
    return source


def build_model(
    source,
    *,
    marker_set: str = "both",
    clavicle_constraint: bool = False,
    glenohumeral: str = "free",
    include_humerus: bool = True,
    first_frame: int = 0,
    last_frame: int = 200,
) -> BiomechanicalModel:
    """
    Build and calibrate the clinical upper-limb model.

    Parameters
    ----------
    source
        c3d/trc file used to calibrate (update) the template, or a ``bionc`` ``Data`` object such as
        :class:`~examples._shared.c3d_data.MultiC3dData` to calibrate over several pooled trials.
    marker_set
        Which scapula/humerus markers are technical (tracked by the IK): ``"cluster"``
        (skin cluster markers only), ``"anatomical"`` (bony landmarks only) or ``"both"``.
    clavicle_constraint
        If True, add a CONSTANT_LENGTH "clavicle" joint keeping RCAS (thorax) <-> RCAJ
        (scapula) constant. If False, the scapula is only held by the free scapulothoracic joint.
    glenohumeral
        ``"free"`` or ``"spherical"`` (ball-and-socket at the shared RGJC point). Ignored when
        ``include_humerus`` is False.
    include_humerus
        If False, stop the chain at the scapula: no RHUMERUS segment and no glenohumeral joint.
    first_frame, last_frame
        Frame window used to calibrate the segment geometry. Only used when ``source`` is a path.
    """
    if marker_set not in MARKER_SETS:
        raise ValueError(f"marker_set must be one of {MARKER_SETS}, got {marker_set!r}")
    use_cluster = marker_set in ("cluster", "both")
    use_anatomical = marker_set in ("anatomical", "both")

    model = BiomechanicalModelTemplate()
    _build_segments(
        model, use_cluster=use_cluster, use_anatomical=use_anatomical, include_humerus=include_humerus
    )

    model.add_joint(
        name="Freeflyer",
        joint_type=JointType.GROUND_FREE,
        parent="GROUND",
        child="THORAX",
        projection_basis=EulerSequence.XYZ,
        child_basis=TransformationMatrixType.Bvu,
    )

    if clavicle_constraint:
        model.add_joint(
            name="Clavicle",
            joint_type=JointType.CONSTANT_LENGTH,
            parent="THORAX",
            child="RSCAPULA",
            parent_point="RCAS",
            child_point="RCAJ",
            length=lambda m, bio: MarkerTemplate.distance(m, bio, "RCAS", "RCAJ"),
            projection_basis=EulerSequence.XYZ,
            child_basis=TransformationMatrixType.Bvu,
        )

    model.add_joint(
        name="Scapulothoracic",
        joint_type=JointType.FREE,
        parent="THORAX",
        child="RSCAPULA",
        projection_basis=EulerSequence.YXZ,
        child_basis=TransformationMatrixType.Bvu,
    )

    if include_humerus:
        _add_glenohumeral_joint(model, glenohumeral)

    return model.update(_as_data(source, first_frame, last_frame))


def build_model_free(source, marker_set: str = "both", **kwargs) -> BiomechanicalModel:
    """Fully unconstrained model: every joint FREE (segments tied only by shared markers)."""
    return build_model(source, marker_set=marker_set, clavicle_constraint=False, glenohumeral="free", **kwargs)


def build_model_constrained(source, marker_set: str = "both", **kwargs) -> BiomechanicalModel:
    """Constrained model: constant-length clavicle + spherical (ball-and-socket) glenohumeral."""
    return build_model(source, marker_set=marker_set, clavicle_constraint=True, glenohumeral="spherical", **kwargs)


def first_frame_guess(trc_filename: str, marker_set: str = "anatomical"):
    """
    ``Q_init`` for frame 0 from the anatomical landmarks.

    Those markers are always technical in the anatomical model and define the segment axes,
    so they can bootstrap ``Q_from_markers`` even when a later cluster-only IK does not track
    them. Used as the first-frame initial guess of the IK in the studies.
    """
    model = build_model_constrained(trc_filename, marker_set=marker_set)
    markers = load_markers(model, trc_filename)[:, :, :1]
    return model.Q_from_markers(markers)


def _add_thorax_ellipsoid_geometry(
    model: BiomechanicalModel, cx: float, cy: float, cz: float, rotation: np.ndarray = None
) -> None:
    """
    Attach the ellipsoid geometry to the thorax: its centre (at segment-frame ``(cx, cy, cz)``)
    and its three principal axes. Because they are interpolated from ``Q_THORAX``, the ellipsoid
    moves rigidly with the thorax.

    ``rotation`` is the 3x3 orientation of the ellipsoid in the thorax segment frame; its columns
    become the principal axes. It defaults to the identity, i.e. axes aligned with the thorax
    segment axes.

    The geometry goes in through :func:`~examples._shared.frames.add_marker_from_scs` and
    :func:`~examples._shared.frames.add_vector_from_scs` rather than bionc's
    ``add_natural_*_from_segment_coordinates``, which convert with the transpose of the right matrix
    -- see :func:`~examples._shared.frames.segment_transformation_matrix`. It matters here: only
    with the correct conversion are the three principal axes orthonormal in the global frame, and
    only then do ``(a, b, c)`` mean geometric semi-axes in metres.
    """
    add_marker_from_scs(model, "THORAX", "ELLIPSOID_CENTER", np.array([cx, cy, cz]))
    rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    for index, name in enumerate(("AXIS_A", "AXIS_B", "AXIS_C")):
        add_vector_from_scs(model, "THORAX", name, rotation[:, index])


def add_glenohumeral_centres(model: BiomechanicalModel) -> BiomechanicalModel:
    """
    Give the spherical glenohumeral joint two *dedicated* centres that a calibration can move.

    Out of the box the joint is built on the ``RGJC`` marker of each segment -- but ``RGJC`` is a
    tracked technical marker, and the marker objective must keep pulling it toward its experimental
    trajectory even while the joint centre is being calibrated elsewhere. So we add two
    non-technical natural markers, ``GH_GLENOID`` on the scapula and ``GH_HEAD`` on the humerus,
    initialised at the respective ``RGJC`` local positions, and rebuild the joint on those. The
    tracked ``RGJC`` markers are left untouched.

    Modifies ``model`` in place and returns it.
    """
    from bionc.bionc_numpy.natural_marker import NaturalMarker

    for segment_name, centre_name in (("RSCAPULA", GH_GLENOID), ("RHUMERUS", GH_HEAD)):
        segment = model.segments[segment_name]
        position = np.asarray(segment.marker_from_name("RGJC").position, dtype=float).reshape(3)
        segment.add_natural_marker(
            NaturalMarker(
                name=centre_name,
                parent_name=segment_name,
                position=position,
                is_technical=False,
                is_anatomical=True,
            )
        )

    joint = model.joints["Glenohumeral"]
    model.remove_joint("Glenohumeral")
    model._add_joint(
        dict(
            name="Glenohumeral",
            joint_type=JointType.SPHERICAL,
            parent="RSCAPULA",
            child="RHUMERUS",
            parent_point=GH_GLENOID,
            child_point=GH_HEAD,
            projection_basis=joint.projection_basis,
            parent_basis=joint.parent_basis,
            child_basis=joint.child_basis,
        )
    )
    return model


def glenohumeral_centres(model: BiomechanicalModel) -> dict[str, np.ndarray]:
    """The two glenohumeral centres in their own segment coordinate system [m]."""
    return {
        "glenoid": natural_to_scs(model, "RSCAPULA", model.segments["RSCAPULA"].marker_from_name(GH_GLENOID).position),
        "head": natural_to_scs(model, "RHUMERUS", model.segments["RHUMERUS"].marker_from_name(GH_HEAD).position),
    }


def set_glenohumeral_centres(model: BiomechanicalModel, glenoid_scs=None, head_scs=None) -> BiomechanicalModel:
    """
    Write calibrated glenohumeral centres (segment coordinates [m]) back into ``model``, in place.

    The marker is mutated rather than replaced: the joint keeps a reference to the very object held
    by the segment (``Joint.Spherical.__init__`` does ``parent.marker_from_name(...)``), so updating
    it in place keeps the constraint and the segment in sync. ``interpolation_matrix`` is derived
    from ``position`` at construction, so both have to be refreshed.
    """
    from bionc.bionc_numpy.natural_vector import NaturalVector

    for segment_name, centre_name, position_scs in (
        ("RSCAPULA", GH_GLENOID, glenoid_scs),
        ("RHUMERUS", GH_HEAD, head_scs),
    ):
        if position_scs is None:
            continue
        marker = model.segments[segment_name].marker_from_name(centre_name)
        marker.position = NaturalVector(scs_to_natural(model, segment_name, position_scs))
        marker.interpolation_matrix = marker.position.interpolate()
    return model


def _add_scapula_landmark_centroid(model: BiomechanicalModel, name: str = "SCAP_CENTROID") -> None:
    """
    Add a scapula marker at the centroid of the three anatomical landmarks RSAA, RSIA, RSRS
    (the AA / AI / TS of the ISB scapula), used as the contact point of the one-point ellipsoid
    joint. Its natural position is the mean of the three landmarks' natural positions.
    """
    from bionc.bionc_numpy.natural_marker import NaturalMarker

    scapula = model["RSCAPULA"]
    landmark_positions = [
        np.asarray(scapula.marker_from_name(landmark).position, dtype=float).reshape(3)
        for landmark in ("RSAA", "RSIA", "RSRS")
    ]
    centroid = np.mean(landmark_positions, axis=0)
    scapula.add_natural_marker(
        NaturalMarker(name=name, parent_name="RSCAPULA", position=centroid, is_technical=False, is_anatomical=True)
    )


def build_scapulothoracic_ellipsoid_model(
    source,
    theta,
    *,
    joint: str = "tangent",
    rotation: np.ndarray = None,
    marker_set: str = "anatomical",
    clavicle_constraint: bool = True,
    glenohumeral: str = "spherical",
    include_humerus: bool = True,
    calibratable_gh_centres: bool = False,
    **build_kwargs,
) -> BiomechanicalModel:
    """
    Model whose scapulothoracic FREE joint is replaced by an ellipsoid joint (Naaim 2016/2017).

    Two variants of the joint are available:

    * ``joint="tangent"`` (ELLIPSOID_ON_PLANE) -- the scapula plane stays tangent to a thoracic
      ellipsoid carried by the THORAX (1 holonomic constraint, no penetration by definition);
    * ``joint="point"`` (POINT_ON_ELLIPSOID) -- a single scapula point, the centroid of the
      RSAA/RSIA/RSRS landmarks, lies on the ellipsoid (1 holonomic constraint).

    Parameters
    ----------
    source
        c3d/trc path or ``Data`` object used to build and calibrate the model.
    theta
        The 6 ellipsoid parameters ``(a, b, c, cx, cy, cz)``: the three semi-axis lengths [m] and
        the ellipsoid centre in the THORAX segment coordinate system.
    joint
        ``"tangent"`` or ``"point"``, see above.
    rotation
        3x3 orientation of the ellipsoid in the thorax segment frame (columns = principal axes).
        Defaults to the identity, i.e. axes aligned with the thorax segment axes.
    marker_set, clavicle_constraint, glenohumeral, include_humerus, build_kwargs
        Forwarded to :func:`build_model`. ``include_humerus=False`` with
        ``clavicle_constraint=False`` gives the thorax-and-scapula-only model used to calibrate the
        ellipsoid on its own.
    calibratable_gh_centres
        If True, rebuild the spherical glenohumeral joint on the dedicated ``GH_GLENOID`` /
        ``GH_HEAD`` centres so a calibration can move them (see :func:`add_glenohumeral_centres`).
    """
    if joint not in ELLIPSOID_JOINTS:
        raise ValueError(f"joint must be one of {ELLIPSOID_JOINTS}, got {joint!r}")

    a, b, c, cx, cy, cz = theta
    model = build_model(
        source,
        marker_set=marker_set,
        clavicle_constraint=clavicle_constraint,
        glenohumeral=glenohumeral,
        include_humerus=include_humerus,
        **build_kwargs,
    )
    if calibratable_gh_centres:
        add_glenohumeral_centres(model)
    _add_thorax_ellipsoid_geometry(model, cx, cy, cz, rotation=rotation)

    ellipsoid = dict(
        name="Scapulothoracic",
        parent="THORAX",
        child="RSCAPULA",
        semi_axis_lengths=(a, b, c),
        ellipsoid_center="ELLIPSOID_CENTER",
        ellipsoid_axis_a="AXIS_A",
        ellipsoid_axis_b="AXIS_B",
        ellipsoid_axis_c="AXIS_C",
        projection_basis=EulerSequence.YXZ,  # match the FREE scapulothoracic basis so angles are comparable
        child_basis=TransformationMatrixType.Bvu,
    )

    if joint == "tangent":
        # Scapula plane: the contact point is the centroid of the three anatomical landmarks
        # (RSAA/RSIA/RSRS), which lies on the scapula plane. The plane normal is the scapula u-axis
        # (= normal_to(RSAA, RSIA, RSRS)); the natural direction [-1, 0, 0] makes it point POSTERIORLY
        # (away from the thorax), so the tangent ellipsoid centre sits anterior to the scapula (inside
        # the thorax) instead of behind it.
        _add_scapula_landmark_centroid(model, name="SCAP_CONTACT")
        add_vector_from_scs(model, "RSCAPULA", "SCAP_NORMAL", np.array([-1.0, 0.0, 0.0]))
        ellipsoid |= dict(
            joint_type=JointType.ELLIPSOID_ON_PLANE, plane_point="SCAP_CONTACT", plane_normal="SCAP_NORMAL"
        )
    else:
        _add_scapula_landmark_centroid(model, name="SCAP_CENTROID")
        ellipsoid |= dict(joint_type=JointType.POINT_ON_ELLIPSOID, contact_point="SCAP_CENTROID")

    # Swap the FREE joint for the ellipsoid joint, keeping the same Euler bases so the reported
    # scapulothoracic angles stay comparable to the FREE baseline.
    model.remove_joint("Scapulothoracic")
    model._add_joint(ellipsoid)
    return model


def build_ellipsoid_model(source, theta, marker_set: str = "anatomical", **kwargs) -> BiomechanicalModel:
    """Tangent (ELLIPSOID_ON_PLANE) scapulothoracic model -- see :func:`build_scapulothoracic_ellipsoid_model`."""
    return build_scapulothoracic_ellipsoid_model(source, theta, joint="tangent", marker_set=marker_set, **kwargs)


def build_point_on_ellipsoid_model(source, theta, marker_set: str = "both", **kwargs) -> BiomechanicalModel:
    """One-point (POINT_ON_ELLIPSOID) scapulothoracic model -- see :func:`build_scapulothoracic_ellipsoid_model`."""
    return build_scapulothoracic_ellipsoid_model(source, theta, joint="point", marker_set=marker_set, **kwargs)
