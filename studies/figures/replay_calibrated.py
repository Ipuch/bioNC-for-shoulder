"""
Replay a trial with a calibrated model, in rerun: markers, ellipsoid and contact point.

Any leave-one-out fold can drive any trial. The model is rebuilt from the numbers that fold cached
(:func:`~studies.shoulder_calibration.rebuild_calibrated_model`) rather than re-solved, so this
opens in seconds.

The default is the honest one: ``--trial X`` alone replays trial X with the fold that **held X out**,
so what you are watching is the calibration working on data it never saw. Point ``--fold`` at a
different trial to use a calibration that did train on it, and the script says which case you got.

``--gh`` picks which glenohumeral model is replayed:

* ``spherical`` (default) -- the calibration cached in ``results/loo/``;
* ``constant_length`` -- the one cached in ``results_gh_constant/loo/``;
* ``both`` -- the two side by side, each as its own toggleable model with its own ellipsoid. The two
  sweeps held out the same trials and share steps 1 and 2, so what differs on screen is the
  glenohumeral joint.

What is drawn:

* each calibrated model, reconstructed on the trial by differential IK, with the experimental
  markers overlaid;
* the **thoracic ellipsoid** of each, riding on the thorax, as a wireframe (red for constant length,
  blue for spherical);
* the **contact point**, riding on the scapula — the point the ellipsoid joint constrains. On a
  correct reconstruction it stays on the surface for the whole trial;
* optionally (``--compare-free``) the all-FREE reconstruction of the same trial as a further,
  separately toggleable model, so you can see where the constraints moved the bones.

**Needs a graphical session** (rerun opens a window).

Run:
    python studies/figures/replay_calibrated.py --trial FUNCTIONAL1
    python studies/figures/replay_calibrated.py --trial FUNCTIONAL1 --gh constant_length
    python studies/figures/replay_calibrated.py --trial FUNCTIONAL1 --gh both --compare-free
    python studies/figures/replay_calibrated.py --trial ANALYTIC2 --fold FUNCTIONAL4
"""

import argparse

import numpy as np

from examples._shared.c3d_data import MultiC3dData
from examples._shared.ik import solve_trial
from examples._shared.viz import ELLIPSOID_RGBA, named_bionc_model, overlay_ellipsoids
from examples.clinical.model import build_model_free
from studies import GLENOHUMERAL, RESULTS_ROOTS
from studies.shoulder_calibration import MARKER_SET, rebuild_calibrated_model, trial_label, trials
from studies.shoulder_calibration_loo import loo_dir

# how each glenohumeral model is named and coloured in the viewer
STYLE = {
    "spherical": ("GH spherical", (70, 110, 220, 90)),
    "constant_length": ("GH constant length", ELLIPSOID_RGBA),
}


def load_fold(name: str, glenohumeral: str = GLENOHUMERAL) -> dict:
    """The cached fold whose held-out trial is ``name``."""
    directory = loo_dir(glenohumeral)
    path = directory / f"fold_{name}.npz"
    if not path.exists():
        available = sorted(p.stem.replace("fold_", "") for p in directory.glob("fold_*.npz"))
        raise SystemExit(
            f"no cached fold for {name!r} in {directory}.\n"
            + (
                f"available: {', '.join(available)}"
                if available
                else f"run `python studies/shoulder_calibration_loo.py --gh {glenohumeral}` first."
            )
        )
    return dict(np.load(path, allow_pickle=True))


def model_from_fold(fold: dict):
    """
    Rebuild that fold's step-3 model from its cached parameters.

    The glenohumeral joint comes from the fold too, so a constant-length calibration is replayed
    with the joint it was calibrated with rather than with the spherical default.
    """
    return rebuild_calibrated_model(
        [str(path) for path in fold["train"]],
        semi_axes=fold["step3_semi_axes"],
        ellipsoid_center_scs=fold["step3_center_scs"],
        glenoid_scs=fold["step3_glenoid"],
        head_scs=fold["step3_head"],
        clavicle_length=float(fold["step3_clavicle"]),
        rotation=fold.get("step3_axes_scs"),
        marker_set=MARKER_SET,
        glenohumeral=str(fold.get("glenohumeral", GLENOHUMERAL)),
        gh_length=float(fold.get("step3_gh_length", 0.0)) or None,
    )


def calibrated_replay(glenohumeral: str, fold_name: str, path: str, stride: int) -> dict:
    """One glenohumeral model's calibrated reconstruction of the trial: ``{fold, model, Q, markers}``."""
    fold = load_fold(fold_name, glenohumeral)
    model = model_from_fold(fold)
    reconstruction = solve_trial(model, path, stride=stride)
    label = STYLE[glenohumeral][0]
    length = float(fold.get("step3_gh_length", 0.0)) * 1000
    print(
        f"{label}: {reconstruction['Qopt'].shape[1]} frames, marker RMSE {reconstruction['rmse_mm']:.2f} mm"
        + (f", GH length {length:.2f} mm" if length else "")
    )
    return dict(fold=fold, model=model, Q=reconstruction["Qopt"], markers=reconstruction["markers"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial", default="ANALYTIC4", help="trial to replay, e.g. ANALYTIC2")
    parser.add_argument(
        "--fold", default=None, help="fold whose calibration to use (default: the one holding out --trial)"
    )
    parser.add_argument("--stride", type=int, default=1, help="frame stride for the replay (default 1)")
    parser.add_argument("--compare-free", action="store_true", help="also show the all-FREE reconstruction")
    parser.add_argument(
        "--gh",
        default="constant_length", # spherical
        choices=(*RESULTS_ROOTS, "both"),
        help=f"glenohumeral model to replay, or both side by side (default: {GLENOHUMERAL})",
    )
    arguments = parser.parse_args()

    by_label = {trial_label(path): path for path in trials()}
    if arguments.trial not in by_label:
        raise SystemExit(f"unknown trial {arguments.trial!r}; available: {', '.join(by_label)}")
    path = by_label[arguments.trial]
    fold_name = arguments.fold or arguments.trial
    models = tuple(RESULTS_ROOTS) if arguments.gh == "both" else (arguments.gh,)

    replays = {gh: calibrated_replay(gh, fold_name, path, arguments.stride) for gh in models}
    first = next(iter(replays.values()))
    seen = arguments.trial in {trial_label(str(p)) for p in first["fold"]["train"]}
    print(
        f"replaying {arguments.trial} with the calibration that held out {trial_label(str(first['fold']['held_out']))} — "
        + ("this trial WAS in its training set" if seen else "this trial was NOT seen during calibration")
    )

    from pyorerun import PhaseRerun, PyoMarkers

    named = {STYLE[gh][0]: (replay["model"], replay["Q"]) for gh, replay in replays.items()}
    if arguments.compare_free:
        free_model = build_model_free(MultiC3dData([str(p) for p in first["fold"]["train"]]), marker_set=MARKER_SET)
        free = solve_trial(free_model, path, stride=arguments.stride)
        print(f"all-FREE reference: marker RMSE {free['rmse_mm']:.2f} mm")
        named["all-FREE"] = (free_model, free["Qopt"])

    t_span = np.linspace(0, 1, first["Q"].shape[1])
    phase = PhaseRerun(t_span=t_span)
    for display_name, (each_model, each_Q) in named.items():
        phase.add_animated_model(named_bionc_model(each_model, display_name), each_Q)

    # every model tracks the same technical markers of the same trial, so one set is enough
    pyomarkers = PyoMarkers(data=first["markers"], marker_names=list(first["model"].marker_names_technical))
    pyomarkers.show_labels = False
    phase.add_xp_markers("experimental_markers", pyomarkers)
    phase.rerun()

    # the recording is live now, so the ellipsoids can be logged onto the same timeline
    ellipsoids = []
    for gh, replay in replays.items():
        joint = replay["model"].joints["Scapulothoracic"]
        name, rgba = STYLE[gh]
        ellipsoids.append(
            (
                f"ellipsoid ({name})",
                replay["model"],
                replay["Q"],
                np.array([float(length) for length in joint.semi_axis_lengths]),
                np.asarray(joint.ellipsoid_center.position, dtype=float).reshape(3),
                rgba,
            )
        )
    overlay_ellipsoids(ellipsoids, t_span)


if __name__ == "__main__":
    main()
