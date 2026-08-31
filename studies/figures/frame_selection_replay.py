"""
See the calibration frames as postures, not as points on a scatter plot.

The matplotlib figures in :mod:`studies.figures.frame_selection` say whether the selection covers
the posture space; this one lets you look at what those postures actually *are*. It replays only
the frames the calibration will use -- pooled across the whole session and ordered so that
consecutive frames are far apart in posture -- with the model reconstructed on them and the
experimental markers overlaid.

Two things worth looking for:

* does the arm actually go through distinct configurations, or does the whole selection sit in one
  region of the workspace? Long stretches of near-identical postures mean the frame budget is being
  spent on redundant information.
* does the scapula reconstruction look sane on the extreme postures? Those are the frames that
  carry most of the ellipsoid's information, and also the ones most likely to have soft-tissue
  artefact.

``--all-frames`` replays the whole session instead, for comparison.

**Needs a graphical session** (pyorerun opens a rerun window).

Run:
    python studies/figures/frame_selection_replay.py [--all-frames] [--trial FUNCTIONAL1]
"""

import argparse

import numpy as np

from examples._shared.c3d_data import MultiC3dData, load_markers_multi, select_calibration_frames
from examples._shared.ik import load_markers
from examples.clinical.model import build_model_constrained
from studies.shoulder_calibration import FRAMES_PER_TRIAL, MARKER_SET, trial_label, trials


def replay(model, markers: np.ndarray, name: str) -> None:
    """Animate ``model`` reconstructed on ``markers``, with the experimental markers overlaid."""
    from bionc.vizualization.pyorerun_interface import BioncModelNoMesh
    from pyorerun import PhaseRerun, PyoMarkers

    Q = np.asarray(model.Q_from_markers(markers))
    pyomarkers = PyoMarkers(data=markers, marker_names=list(model.marker_names_technical))
    pyomarkers.show_labels = False

    print(f"replaying {name}: {markers.shape[2]} frames")
    phase = PhaseRerun(t_span=np.linspace(0, 1, Q.shape[1]))
    phase.add_animated_model(BioncModelNoMesh(model), Q, tracked_markers=pyomarkers)
    phase.rerun()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-frames", action="store_true", help="replay every frame instead of the selection")
    parser.add_argument("--trial", default=None, help="restrict to one trial, e.g. FUNCTIONAL1")
    parser.add_argument("--stride", type=int, default=10, help="stride used with --all-frames")
    arguments = parser.parse_args()

    paths = trials()
    if arguments.trial:
        paths = [path for path in paths if trial_label(path) == arguments.trial]
        if not paths:
            raise SystemExit(f"no trial named {arguments.trial!r}")

    model = build_model_constrained(MultiC3dData(paths), marker_set=MARKER_SET)

    if arguments.all_frames:
        markers = np.concatenate([load_markers(model, path, stride=arguments.stride) for path in paths], axis=2)
        replay(model, markers, f"every {arguments.stride}th frame of {len(paths)} trial(s)")
        return

    selection = select_calibration_frames(model, paths, per_trial=FRAMES_PER_TRIAL)
    markers = load_markers_multi(model, paths, selection)
    replay(model, markers, f"the {markers.shape[2]} calibration frames of {len(paths)} trial(s)")


if __name__ == "__main__":
    main()
