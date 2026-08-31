"""
Replay a trial with a calibrated model, in rerun: markers, ellipsoid and contact point.

Any leave-one-out fold can drive any trial. The model is rebuilt from the numbers that fold cached
(:func:`~studies.shoulder_calibration.rebuild_calibrated_model`) rather than re-solved, so this
opens in seconds.

The default is the honest one: ``--trial X`` alone replays trial X with the fold that **held X out**,
so what you are watching is the calibration working on data it never saw. Point ``--fold`` at a
different trial to use a calibration that did train on it, and the script says which case you got.

What is drawn:

* the calibrated model, reconstructed on the trial by differential IK, with the experimental
  markers overlaid;
* the **thoracic ellipsoid**, riding on the thorax, as a wireframe;
* the **contact point**, riding on the scapula — the point the ellipsoid joint constrains. On a
  correct reconstruction it stays on the surface for the whole trial;
* optionally (``--compare-free``) the all-FREE reconstruction of the same trial as a second,
  separately toggleable model, so you can see where the constraints moved the bones.

**Needs a graphical session** (rerun opens a window).

Run:
    python studies/figures/replay_calibrated.py --trial FUNCTIONAL1
    python studies/figures/replay_calibrated.py --trial ANALYTIC2 --fold FUNCTIONAL4 --compare-free
"""

import argparse

import numpy as np

from bionc import InverseKinematics

from examples._shared.c3d_data import MultiC3dData
from examples._shared.ik import load_markers
from examples._shared.viz import named_bionc_model, overlay_ellipsoids
from examples.clinical.model import build_model_free
from studies.shoulder_calibration import MARKER_SET, rebuild_calibrated_model, trial_label, trials
from studies.shoulder_calibration_loo import RESULTS_DIR

ELLIPSOID_RGBA = (220, 70, 70, 90)  # translucent surface, opaque contact point


def load_fold(name: str) -> dict:
    """The cached fold whose held-out trial is ``name``."""
    path = RESULTS_DIR / f"fold_{name}.npz"
    if not path.exists():
        available = sorted(p.stem.replace("fold_", "") for p in RESULTS_DIR.glob("fold_*.npz"))
        raise SystemExit(
            f"no cached fold for {name!r} in {RESULTS_DIR}.\n"
            + (f"available: {', '.join(available)}" if available else
               "run `python studies/shoulder_calibration_loo.py` first.")
        )
    return dict(np.load(path, allow_pickle=True))


def model_from_fold(fold: dict):
    """Rebuild that fold's step-3 model from its cached parameters."""
    return rebuild_calibrated_model(
        [str(path) for path in fold["train"]],
        semi_axes=fold["step3_semi_axes"],
        ellipsoid_center_scs=fold["step3_center_scs"],
        glenoid_scs=fold["step3_glenoid"],
        head_scs=fold["step3_head"],
        clavicle_length=float(fold["step3_clavicle"]),
        rotation=fold.get("step3_axes_scs"),
        marker_set=MARKER_SET,
    )


def solve(model, path: str, stride: int):
    """Differential IK on the whole trial; returns ``(markers, Qopt, rmse_mm)``."""
    markers = load_markers(model, path, stride=stride)
    ik = InverseKinematics(model, markers)
    Qopt = ik.solve(method="dik")
    rmse = float(np.sqrt(np.mean((ik.sol()["marker_residuals_norm"] * 1000) ** 2)))
    return markers, np.asarray(Qopt), rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="ANALYTIC4", help="trial to replay, e.g. ANALYTIC2")
    parser.add_argument("--fold", default=None, help="fold whose calibration to use (default: the one holding out --trial)")
    parser.add_argument("--stride", type=int, default=1, help="frame stride for the replay (default 5)")
    parser.add_argument("--compare-free", action="store_true", help="also show the all-FREE reconstruction")
    arguments = parser.parse_args()

    by_label = {trial_label(path): path for path in trials()}
    if arguments.trial not in by_label:
        raise SystemExit(f"unknown trial {arguments.trial!r}; available: {', '.join(by_label)}")
    path = by_label[arguments.trial]

    fold = load_fold(arguments.fold or arguments.trial)
    trained_on = {trial_label(str(p)) for p in fold["train"]}
    seen = arguments.trial in trained_on
    print(
        f"replaying {arguments.trial} with the calibration that held out {trial_label(str(fold['held_out']))} — "
        + ("this trial WAS in its training set" if seen else "this trial was NOT seen during calibration")
    )

    model = model_from_fold(fold)
    markers, Qopt, rmse = solve(model, path, arguments.stride)
    print(f"calibrated model: {Qopt.shape[1]} frames, marker RMSE {rmse:.2f} mm")

    from pyorerun import PhaseRerun, PyoMarkers

    named = {"calibrated": (model, Qopt)}
    if arguments.compare_free:
        free_model = build_model_free(MultiC3dData([str(p) for p in fold["train"]]), marker_set=MARKER_SET)
        _, free_Q, free_rmse = solve(free_model, path, arguments.stride)
        print(f"all-FREE reference: marker RMSE {free_rmse:.2f} mm")
        named["all-FREE"] = (free_model, free_Q)

    t_span = np.linspace(0, 1, Qopt.shape[1])
    phase = PhaseRerun(t_span=t_span)
    for display_name, (each_model, each_Q) in named.items():
        phase.add_animated_model(named_bionc_model(each_model, display_name), each_Q)

    pyomarkers = PyoMarkers(data=markers, marker_names=list(model.marker_names_technical))
    pyomarkers.show_labels = False
    phase.add_xp_markers("experimental_markers", pyomarkers)
    phase.rerun()

    # the recording is live now, so the ellipsoid can be logged onto the same timeline
    joint = model.joints["Scapulothoracic"]
    overlay_ellipsoids(
        [
            (
                "scapulothoracic_ellipsoid",
                model,
                Qopt,
                np.array([float(length) for length in joint.semi_axis_lengths]),
                np.asarray(joint.ellipsoid_center.position, dtype=float).reshape(3),
                ELLIPSOID_RGBA,
            )
        ],
        t_span,
    )


if __name__ == "__main__":
    main()
