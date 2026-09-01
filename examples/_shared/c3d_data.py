"""Pooling several c3d trials into one dataset, for calibrations that span a whole session.

A single-trial calibration only ever sees one posture family. The scapulothoracic ellipsoid and the
functional joint centres are identified by *variety* of posture, so they want every trial of the
session at once. Two things are needed for that:

* :class:`MultiC3dData` -- a ``bionc`` ``Data`` (see ``bionc.model_creation.protocols.Data``, which
  only requires a ``values`` dict) that concatenates the frames of several c3d files. Handing it to
  ``BiomechanicalModelTemplate.update`` averages the marker local positions and the data-driven joint
  ``length`` callables over all of those trials instead of one.
* :func:`select_calibration_frames` -- picks the handful of frames the (expensive) all-frames
  calibration will actually run on, spread over the *workspace* rather than over the timeline.
"""

from pathlib import Path

import numpy as np

from bionc import C3dData, NaturalCoordinates
from pyomeca import Markers

from examples._shared.ik import c3d_length_factor, load_markers

# Stride the calibration scans a trial on to build its candidate set. One truth: the figures that
# draw the selection must scan exactly what the selection scanned, or they mislabel which frames
# were picked.
COARSE_STRIDE = 10


class MultiC3dData:
    """
    Several c3d trials seen as one :class:`~bionc.model_creation.protocols.Data`.

    ``values[marker_name]`` is the ``(4, sum_of_frames)`` concatenation of the per-trial arrays, in
    metres (each trial is converted by :class:`~bionc.C3dData`, which reads its own ``POINT:UNITS``,
    so mixing millimetre and metre files is safe).

    Only the markers present in *every* trial are kept: the ``ANALYTIC`` files of the 99007140
    dataset carry six extra channels (``RHT``/``RGH``/``RST`` and their left counterparts) that the
    ``FUNCTIONAL`` files do not, and those are Euler-angle channels rather than points anyway.
    """

    def __init__(self, paths, first_frame: int = 0, last_frame: int = None):
        self.paths = [str(path) for path in paths]
        if not self.paths:
            raise ValueError("MultiC3dData needs at least one c3d path")

        trials = [C3dData(path, first_frame=first_frame, last_frame=last_frame) for path in self.paths]
        self.nb_frames_per_trial = [next(iter(trial.values.values())).shape[1] for trial in trials]

        shared = set(trials[0].values)
        for trial in trials[1:]:
            shared &= set(trial.values)

        # Iterate the first trial rather than the set: set order over strings varies between
        # processes with PYTHONHASHSEED, and a calibration that documents itself as deterministic
        # should not leave the key order of its own dataset to chance.
        self.values = {
            name: np.concatenate([trial.values[name] for trial in trials], axis=1)
            for name in trials[0].values
            if name in shared
        }

    @property
    def nb_frames(self) -> int:
        return int(sum(self.nb_frames_per_trial))

    def __repr__(self) -> str:
        names = ", ".join(Path(path).stem[-14:] for path in self.paths)
        return f"MultiC3dData({len(self.paths)} trials, {self.nb_frames} frames: {names})"


def load_markers_multi(model, paths, frame_index: dict[str, np.ndarray]) -> np.ndarray:
    """
    Technical markers of ``model`` gathered from several trials, ``(3, nb_technical, nb_selected)``.

    ``frame_index`` maps a trial path to the frame numbers to take from it (as returned by
    :func:`select_calibration_frames`). Columns come out in ``paths`` order, then frame order.
    """
    blocks = [load_markers(model, str(path))[:, :, np.asarray(frame_index[str(path)], dtype=int)] for path in paths]
    return np.concatenate(blocks, axis=2)


def load_named_markers(paths, names, frame_index: dict[str, np.ndarray] = None) -> np.ndarray:
    """
    Named markers straight from the c3d files, ``(3, len(names), nb_frames)`` in metres.

    Unlike :func:`load_markers_multi` this does not go through a model, so it reaches markers the
    model does not track (``RCAS``, say, which is a non-technical thorax marker but one end of the
    clavicle). ``frame_index`` selects frames per trial; omit it to take every frame.
    """
    blocks = []
    for path in paths:
        path = str(path)
        block = Markers.from_c3d(path, usecols=list(names)).to_numpy()[:3] / c3d_length_factor(path)
        if frame_index is not None:
            block = block[:, :, np.asarray(frame_index[path], dtype=int)]
        blocks.append(block)
    return np.concatenate(blocks, axis=2)


def posture_features(model, markers: np.ndarray) -> np.ndarray:
    """
    One row per frame describing the shoulder posture, used to spread the frame selection.

    The features are the raw (pre-optimisation) scapulothoracic Euler angles plus, when the model
    carries a humerus, the humerothoracic ones -- i.e. exactly the degrees of freedom the ellipsoid
    and the glenohumeral centres have to be identified from. Columns are standardised so no angle
    dominates the distances.
    """
    Q = np.asarray(model.Q_from_markers(markers))
    joint_names = list(model.joints.joint_names)
    wanted = [name for name in ("Scapulothoracic", "Glenohumeral") if name in joint_names]
    columns = [joint_names.index(name) for name in wanted]

    features = np.zeros((Q.shape[1], 3 * len(columns)))
    for frame in range(Q.shape[1]):
        angles = model.natural_coordinates_to_joint_angles(NaturalCoordinates(Q[:, frame]))
        features[frame] = np.concatenate([np.asarray(angles[:, column]).reshape(3) for column in columns])

    features = np.unwrap(features, axis=0)
    spread = features.std(axis=0)
    spread[spread < 1e-9] = 1.0
    return (features - features.mean(axis=0)) / spread


def farthest_point_sample(features: np.ndarray, nb_samples: int) -> np.ndarray:
    """
    Greedy farthest-point sampling: indices of ``nb_samples`` rows that are as spread out as possible.

    Seeded on the row farthest from the mean posture (deterministic), then each new pick maximises
    the distance to the closest already-picked row.
    """
    nb_rows = features.shape[0]
    if nb_samples >= nb_rows:
        return np.arange(nb_rows)

    picked = [int(np.argmax(np.linalg.norm(features - features.mean(axis=0), axis=1)))]
    distances = np.linalg.norm(features - features[picked[0]], axis=1)
    for _ in range(nb_samples - 1):
        candidate = int(np.argmax(distances))
        picked.append(candidate)
        distances = np.minimum(distances, np.linalg.norm(features - features[candidate], axis=1))
    return np.sort(np.array(picked))


def posture_scan(model, paths, coarse_stride: int = COARSE_STRIDE) -> dict[str, dict]:
    """
    The candidate set the frame selection chooses from: per trial, ``{candidates, features}``.

    ``candidates`` are frame numbers relative to the full trial, ``features`` the matching rows of
    :func:`posture_features`. Scanning is the expensive half (it reads the c3d and reconstructs
    every candidate frame), so it is exposed separately: the figures want the same scan the
    selection ran on, and recomputing it would both cost a second pass and risk disagreeing with it.
    """
    scan = {}
    for path in paths:
        path = str(path)
        markers = load_markers(model, path, stride=coarse_stride)
        scan[path] = dict(
            candidates=np.arange(markers.shape[2]) * coarse_stride,
            features=posture_features(model, markers),
        )
    return scan


def select_calibration_frames(
    model, paths, per_trial: int = 35, coarse_stride: int = COARSE_STRIDE, scan: dict = None
) -> dict[str, np.ndarray]:
    """
    Frames each trial contributes to the all-frames calibration, spread over the *workspace*.

    A uniform stride spreads frames over *time*, which for a slow sweep means many near-duplicate
    postures and a badly conditioned fit. Instead the trial is scanned on ``coarse_stride`` and
    farthest-point sampled in the posture space of :func:`posture_features`, so the selection
    covers the range of scapular (and glenohumeral) configurations the trial actually visited.

    Pass ``scan`` (from :func:`posture_scan`) to reuse a scan you already have; otherwise one is
    made here.

    Returns ``{path: frame_numbers}`` with frame numbers relative to the full trial. Deterministic.
    """
    scan = posture_scan(model, paths, coarse_stride) if scan is None else scan
    return {
        path: entry["candidates"][farthest_point_sample(entry["features"], per_trial)] for path, entry in scan.items()
    }
