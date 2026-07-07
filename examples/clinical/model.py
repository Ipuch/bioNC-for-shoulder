"""Upper-limb model for the clinical marker set.

Same structure as the Henninger model (segments defined once, joints selected by keyword),
plus two clinical specificities:

* ``marker_set`` chooses which markers are *technical* (tracked by the IK): the skin
  ``Cluster_*`` markers, the anatomical bone landmarks, or both;
* :func:`build_ellipsoid_model` replaces the free scapulothoracic joint by a tangent
  ellipsoid-on-plane joint (Naaim 2016/2017), used by the calibration study.

Joint chain (proximal -> distal):
    GROUND --Freeflyer--> THORAX --[Clavicle]--> / --Scapulothoracic--> RSCAPULA --Glenohumeral--> RHUMERUS
"""

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

from examples._shared.frames import u_thorax
from examples._shared.ik import load_markers

MARKER_SETS = ("cluster", "anatomical", "both")
GLENOHUMERAL_CONSTRAINTS = ("free", "spherical")


def _build_segments(model: BiomechanicalModelTemplate, *, use_cluster: bool, use_anatomical: bool) -> None:
    """Add the THORAX, RSCAPULA and RHUMERUS segments and their markers (clinical names)."""
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


def build_model(
    trc_filename: str,
    *,
    marker_set: str = "both",
    clavicle_constraint: bool = False,
    glenohumeral: str = "free",
    first_frame: int = 0,
    last_frame: int = 200,
) -> BiomechanicalModel:
    """
    Build and calibrate the clinical upper-limb model.

    Parameters
    ----------
    trc_filename
        c3d/trc file used to calibrate (update) the template.
    marker_set
        Which scapula/humerus markers are technical (tracked by the IK): ``"cluster"``
        (skin cluster markers only), ``"anatomical"`` (bony landmarks only) or ``"both"``.
    clavicle_constraint
        If True, add a CONSTANT_LENGTH "clavicle" joint keeping RCAS (thorax) <-> RCAJ
        (scapula) constant. If False, the scapula is only held by the free scapulothoracic joint.
    glenohumeral
        ``"free"`` or ``"spherical"`` (ball-and-socket at the shared RGJC point).
    first_frame, last_frame
        Frame window used to calibrate the segment geometry.
    """
    if marker_set not in MARKER_SETS:
        raise ValueError(f"marker_set must be one of {MARKER_SETS}, got {marker_set!r}")
    use_cluster = marker_set in ("cluster", "both")
    use_anatomical = marker_set in ("anatomical", "both")

    model = BiomechanicalModelTemplate()
    _build_segments(model, use_cluster=use_cluster, use_anatomical=use_anatomical)

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

    _add_glenohumeral_joint(model, glenohumeral)

    data = C3dData(f"{trc_filename}", first_frame=first_frame, last_frame=last_frame)
    return model.update(data)


def build_model_free(trc_filename: str, marker_set: str = "both") -> BiomechanicalModel:
    """Fully unconstrained model: every joint FREE (segments tied only by shared markers)."""
    return build_model(trc_filename, marker_set=marker_set, clavicle_constraint=False, glenohumeral="free")


def build_model_constrained(trc_filename: str, marker_set: str = "both") -> BiomechanicalModel:
    """Constrained model: constant-length clavicle + spherical (ball-and-socket) glenohumeral."""
    return build_model(trc_filename, marker_set=marker_set, clavicle_constraint=True, glenohumeral="spherical")


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


def _add_thorax_ellipsoid_geometry(model: BiomechanicalModel, cx: float, cy: float, cz: float) -> None:
    """
    Attach the ellipsoid geometry to the thorax: its centre (at segment-frame ``(cx, cy, cz)``)
    and its three principal axes (kept aligned with the thorax segment axes). Because they are
    interpolated from ``Q_THORAX``, the ellipsoid moves rigidly with the thorax.
    """
    model["THORAX"].add_natural_marker_from_segment_coordinates(
        name="ELLIPSOID_CENTER", location=np.array([cx, cy, cz]), is_technical=False, is_anatomical=True
    )
    for name, direction in zip(("AXIS_A", "AXIS_B", "AXIS_C"), ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])):
        model["THORAX"].add_natural_vector_from_segment_coordinates(name=name, direction=np.array(direction))


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


def build_ellipsoid_model(trc_filename: str, theta, marker_set: str = "anatomical") -> BiomechanicalModel:
    """
    Constrained model whose scapulothoracic FREE joint is replaced by a tangent
    ellipsoid-on-plane joint (Naaim 2016/2017): the scapula plane stays tangent to a thoracic
    ellipsoid carried by the THORAX (1 holonomic constraint, no penetration by definition).

    Parameters
    ----------
    trc_filename
        The c3d/trc file used to build and calibrate the model.
    theta
        The 6 ellipsoid parameters ``(a, b, c, cx, cy, cz)``: the three semi-axis lengths [m]
        and the ellipsoid centre expressed in the THORAX segment coordinate system. The three
        principal axes stay aligned with the thorax segment axes.
    marker_set
        Forwarded to :func:`build_model` (which markers are technical).
    """
    a, b, c, cx, cy, cz = theta
    model = build_model_constrained(trc_filename, marker_set=marker_set)
    _add_thorax_ellipsoid_geometry(model, cx, cy, cz)

    # Scapula plane: the contact point is the centroid of the three anatomical landmarks
    # (RSAA/RSIA/RSRS), which lies on the scapula plane. The plane normal is the scapula u-axis
    # (= normal_to(RSAA, RSIA, RSRS)); the natural direction [-1, 0, 0] makes it point POSTERIORLY
    # (away from the thorax), so the tangent ellipsoid centre sits anterior to the scapula (inside
    # the thorax) instead of behind it.
    _add_scapula_landmark_centroid(model, name="SCAP_CONTACT")
    model["RSCAPULA"].add_natural_vector_from_segment_coordinates(
        name="SCAP_NORMAL", direction=np.array([-1.0, 0.0, 0.0])
    )

    # Swap the FREE joint for the tangent ellipsoid joint, keeping the same Euler bases so the
    # reported scapulothoracic angles stay comparable to the FREE baseline.
    model.remove_joint("Scapulothoracic")
    model._add_joint(
        dict(
            name="Scapulothoracic",
            joint_type=JointType.ELLIPSOID_ON_PLANE,
            parent="THORAX",
            child="RSCAPULA",
            semi_axis_lengths=(a, b, c),
            ellipsoid_center="ELLIPSOID_CENTER",
            ellipsoid_axis_a="AXIS_A",
            ellipsoid_axis_b="AXIS_B",
            ellipsoid_axis_c="AXIS_C",
            plane_point="SCAP_CONTACT",
            plane_normal="SCAP_NORMAL",
            projection_basis=EulerSequence.YXZ,  # match the FREE scapulothoracic basis so angles are comparable
            child_basis=TransformationMatrixType.Bvu,
        )
    )
    return model


def build_point_on_ellipsoid_model(trc_filename: str, theta, marker_set: str = "both") -> BiomechanicalModel:
    """
    Constrained model whose scapulothoracic FREE joint is replaced by a one-point
    ellipsoid joint: a single scapula point (the centroid of the RSAA/RSIA/RSRS landmarks)
    is constrained to lie on the thoracic ellipsoid (1 holonomic constraint).

    Same ``theta = (a, b, c, cx, cy, cz)`` convention as :func:`build_ellipsoid_model`.
    """
    a, b, c, cx, cy, cz = theta
    model = build_model_constrained(trc_filename, marker_set=marker_set)
    _add_thorax_ellipsoid_geometry(model, cx, cy, cz)
    _add_scapula_landmark_centroid(model, name="SCAP_CENTROID")

    model.remove_joint("Scapulothoracic")
    model._add_joint(
        dict(
            name="Scapulothoracic",
            joint_type=JointType.POINT_ON_ELLIPSOID,
            parent="THORAX",
            child="RSCAPULA",
            semi_axis_lengths=(a, b, c),
            ellipsoid_center="ELLIPSOID_CENTER",
            ellipsoid_axis_a="AXIS_A",
            ellipsoid_axis_b="AXIS_B",
            ellipsoid_axis_c="AXIS_C",
            contact_point="SCAP_CENTROID",
            projection_basis=EulerSequence.YXZ,  # match the FREE scapulothoracic basis so angles are comparable
            child_basis=TransformationMatrixType.Bvu,
        )
    )
    return model
