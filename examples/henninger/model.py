"""Upper-limb model for the Henninger marker set.

The three segments (THORAX, RSCAPULA, RHUMERUS) and their markers are always the same;
only the *joints* change between the "no constraints" and the "with constraints" versions.
That difference is captured by the keyword arguments of :func:`build_model`, so the segment
definition is written exactly once (in :func:`_build_segments`) and shared by every example
and by the glenohumeral-constraint study.

Joint chain (proximal -> distal):
    GROUND --Freeflyer--> THORAX --[Clavicle]--> / --Scapulothoracic--> RSCAPULA --Glenohumeral--> RHUMERUS
"""

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

GLENOHUMERAL_CONSTRAINTS = ("free", "spherical", "constant_length")


def _build_segments(model: BiomechanicalModelTemplate) -> None:
    """Add the THORAX, RSCAPULA and RHUMERUS segments and their markers (Henninger names)."""
    u_axis_thorax = lambda m, bio: u_thorax(m["IJ"], m["C7"], m["T5"])

    model["THORAX"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=u_axis_thorax),
            proximal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "C7", "IJ"),
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "PX", "T5"),
            w_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "IJ", "C7", "T5")),
        )
    )
    model["THORAX"].add_marker(MarkerTemplate(name="IJ", is_technical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="C7", is_technical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="PX", is_technical=True))
    model["THORAX"].add_marker(MarkerTemplate(name="T5", is_technical=True))

    model["RSCAPULA"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "AA", "IA", "TS")),
            proximal_point="AA",
            distal_point="IA",
            w_axis=AxisTemplate(start="TS", end="AA"),
        )
    )
    model["RSCAPULA"].add_marker(MarkerTemplate(name="AA", is_technical=True))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="TS", is_technical=True))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="IA", is_technical=True))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="AC", is_technical=True))
    model["RSCAPULA"].add_marker(MarkerTemplate(name="GSC", is_technical=True))

    model["RHUMERUS"] = SegmentTemplate(
        natural_segment=NaturalSegmentTemplate(
            u_axis=AxisFunctionTemplate(function=lambda m, bio: MarkerTemplate.normal_to(m, bio, "EL", "EM", "GSChum")),
            proximal_point="GSChum",
            distal_point=lambda m, bio: MarkerTemplate.middle_of(m, bio, "EL", "EM"),
            w_axis=AxisTemplate(start="EM", end="EL"),
        )
    )
    model["RHUMERUS"].add_marker(MarkerTemplate(name="GSChum", is_technical=True))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="EL", is_technical=True))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="EM", is_technical=True))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="GT", is_technical=False, is_anatomical=True))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="SCT", is_technical=False, is_anatomical=True))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="SCB", is_technical=False, is_anatomical=True))
    model["RHUMERUS"].add_marker(MarkerTemplate(name="C", is_technical=False, is_anatomical=True))


def _add_glenohumeral_joint(model: BiomechanicalModelTemplate, glenohumeral: str) -> None:
    """Add the RSCAPULA -> RHUMERUS joint with the requested amount of constraint."""
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
        # no coupling: the humerus is only tied to the scapula by its shared markers
        model.add_joint(joint_type=JointType.FREE, **common)
    elif glenohumeral == "spherical":
        # ball-and-socket: GSC (scapula) and GSChum (humerus) are made coincident
        model.add_joint(joint_type=JointType.SPHERICAL, parent_point="GSC", child_point="GSChum", **common)
    else:  # "constant_length"
        # sphere-on-sphere: the GSC <-> GSChum distance is kept constant computed from the mean distance.
        model.add_joint(
            joint_type=JointType.CONSTANT_LENGTH,
            parent_point="GSC",
            child_point="GSChum",
            length=lambda m, bio: MarkerTemplate.distance(m, bio, "GSC", "GSChum"),
            **common,
        )


def build_model(
    trc_filename: str,
    *,
    clavicle_constraint: bool = False,
    glenohumeral: str = "free",
    first_frame: int = 0,
    last_frame: int = 200,
) -> BiomechanicalModel:
    """
    Build and calibrate the Henninger upper-limb model.

    Parameters
    ----------
    trc_filename
        c3d/trc file used to calibrate (update) the template.
    clavicle_constraint
        If True, add a CONSTANT_LENGTH "clavicle" joint keeping the IJ (thorax) <-> AC (scapula)
        distance constant. If False, the scapula is only held by the free scapulothoracic joint.
    glenohumeral
        Amount of constraint on the RSCAPULA -> RHUMERUS joint: one of
        ``"free"``, ``"spherical"`` or ``"constant_length"``.
    first_frame, last_frame
        Frame window used to calibrate the segment geometry.
    """
    model = BiomechanicalModelTemplate()
    _build_segments(model)

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
            parent_point="IJ",
            child_point="AC",
            length=lambda m, bio: MarkerTemplate.distance(m, bio, "IJ", "AC"),
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


def build_model_free(trc_filename: str) -> BiomechanicalModel:
    """Fully unconstrained model: every joint FREE (segments tied only by shared markers)."""
    return build_model(trc_filename, clavicle_constraint=False, glenohumeral="free")


def build_model_constrained(trc_filename: str) -> BiomechanicalModel:
    """Constrained model: constant-length clavicle + spherical (ball-and-socket) glenohumeral."""
    return build_model(trc_filename, clavicle_constraint=True, glenohumeral="spherical")
