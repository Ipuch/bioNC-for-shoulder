"""
Is the calibration subsample diverse enough?

The all-frames calibration can only afford a few hundred frames out of the session's 25 312, and
which ones it gets decides what the calibration can identify. This script looks at that choice from
four angles:

1. **the posture latent space** -- every candidate frame projected onto the first two principal
   components of the scapulothoracic + glenohumeral angles, with the selected frames on top. If the
   selection hugs one lobe of the cloud, the calibration only ever sees one kind of posture.
2. **coverage vs frame budget** -- how far the worst-covered posture is from its nearest selected
   frame, as the budget grows, for farthest-point sampling against a plain uniform stride. The knee
   of that curve is the honest answer to "how many frames do I need"; the gap between the two curves
   is what the sampling strategy buys.
3. **range covered per angle** -- per trial and per Euler angle, the span the selection reproduces
   against the span the trial actually visited.
4. **the contact-point patch** -- the scapula contact point in the thorax frame, all frames versus
   selected. This is the figure that explains the identifiability limit: the patch is only about
   55 x 52 x 67 mm, and a patch that small carries essentially no curvature signal.

Run:
    python studies/figures/frame_selection.py [--save]
"""

import numpy as np
from matplotlib import pyplot as plt

from examples._shared.c3d_data import (
    COARSE_STRIDE,
    MultiC3dData,
    farthest_point_sample,
    load_markers_multi,
    posture_scan,
    select_calibration_frames,
)
from examples._shared.ik import load_markers
from examples.clinical.model import build_model_constrained
from studies.shoulder_calibration import FRAMES_PER_TRIAL, MARKER_SET, contact_point_cloud, trial_kind, trial_label, trials
from studies.figures import KIND_COLORS, finish, parse_args

ANGLE_LABELS = ["ST Y", "ST X", "ST Z", "GH Y1", "GH X", "GH Y2"]


def gather(paths, per_trial: int = FRAMES_PER_TRIAL) -> dict:
    """Candidate and selected posture features per trial, plus the pooled contact-point clouds."""
    model = build_model_constrained(MultiC3dData(paths), marker_set=MARKER_SET)

    # one scan, shared with the selection: this figure has to draw the candidate set the
    # calibration actually chose from, not an independently recomputed lookalike
    scan = posture_scan(model, paths)
    selection = select_calibration_frames(model, paths, per_trial=per_trial, scan=scan)

    per_trial_data = {
        path: dict(
            features=scan[str(path)]["features"],
            candidates=scan[str(path)]["candidates"],
            picked=np.searchsorted(scan[str(path)]["candidates"], selection[str(path)]),
        )
        for path in paths
    }

    candidate_markers = np.concatenate([load_markers(model, path, stride=COARSE_STRIDE) for path in paths], axis=2)
    return dict(
        model=model,
        paths=list(paths),
        selection=selection,
        per_trial=per_trial_data,
        cloud_all=contact_point_cloud(model, candidate_markers),
        cloud_selected=contact_point_cloud(model, load_markers_multi(model, paths, selection)),
    )


def _pooled_features(data: dict):
    """All candidate rows stacked, with the trial each came from and whether it was selected."""
    rows, trial_of, selected = [], [], []
    for path in data["paths"]:
        entry = data["per_trial"][path]
        rows.append(entry["features"])
        trial_of += [path] * entry["features"].shape[0]
        mask = np.zeros(entry["features"].shape[0], dtype=bool)
        mask[entry["picked"]] = True
        selected.append(mask)
    return np.vstack(rows), np.array(trial_of), np.concatenate(selected)


def plot_latent_space(data: dict):
    """PCA of the posture features: where the selected frames sit inside the session's postures."""
    features, trial_of, selected = _pooled_features(data)
    centred = features - features.mean(axis=0)
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    scores = centred @ components[:2].T
    explained = singular**2 / np.sum(singular**2)

    figure, (axis, axis_scree) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True,
                                              gridspec_kw={"width_ratios": [2, 1]})
    figure.suptitle("Posture latent space: what the calibration frames actually sample", fontsize=14, fontweight="bold")

    for kind, color in KIND_COLORS.items():
        mask = np.array([trial_kind(path) == kind for path in trial_of])
        axis.scatter(scores[mask, 0], scores[mask, 1], s=6, color=color, alpha=0.16, label=f"{kind} (all frames)")
    axis.scatter(
        scores[selected, 0], scores[selected, 1], s=42, facecolor="none", edgecolor="black", lw=1.1,
        label="selected for calibration",
    )
    axis.set_xlabel(f"PC1 ({explained[0]:.0%} of the posture variance)")
    axis.set_ylabel(f"PC2 ({explained[1]:.0%})")
    axis.set_title("candidates vs selection")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=9, loc="best")

    axis_scree.bar(np.arange(1, len(explained) + 1), explained, color="tab:gray")
    axis_scree.plot(np.arange(1, len(explained) + 1), np.cumsum(explained), "o-", color="tab:red", label="cumulative")
    axis_scree.set_xlabel("principal component")
    axis_scree.set_ylabel("share of posture variance")
    axis_scree.set_title("how many directions the postures really span")
    axis_scree.grid(True, axis="y", alpha=0.25)
    axis_scree.legend(fontsize=9)
    return figure


def plot_coverage_curve(data: dict, budgets=(5, 10, 15, 20, 25, 30, 35, 45, 60, 80)):
    """
    Worst-case distance from a candidate posture to its nearest selected one, as the budget grows.

    Farthest-point sampling minimises exactly this quantity, so the curve is the direct answer to
    "how many frames per trial is enough": where it flattens, more frames stop adding postures.
    """
    figure, (axis, axis_gain) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    figure.suptitle("Coverage of the posture space vs frame budget", fontsize=14, fontweight="bold")

    def coverage(features, index):
        distances = np.linalg.norm(features[:, None, :] - features[None, index, :], axis=2)
        return float(np.max(np.min(distances, axis=1)))

    farthest_all, uniform_all = [], []
    for path in data["paths"]:
        features = data["per_trial"][path]["features"]
        farthest = [coverage(features, farthest_point_sample(features, budget)) for budget in budgets]
        uniform = [
            coverage(features, np.linspace(0, features.shape[0] - 1, budget).astype(int)) for budget in budgets
        ]
        color = KIND_COLORS[trial_kind(path)]
        axis.plot(budgets, farthest, "-", color=color, alpha=0.85, lw=1.6)
        axis.plot(budgets, uniform, ":", color=color, alpha=0.6, lw=1.4)
        farthest_all.append(farthest)
        uniform_all.append(uniform)

    axis.axvline(FRAMES_PER_TRIAL, color="black", ls="--", lw=1)
    axis.annotate(f"budget in use\n({FRAMES_PER_TRIAL}/trial)", (FRAMES_PER_TRIAL, axis.get_ylim()[1]),
                  textcoords="offset points", xytext=(6, -28), fontsize=9)
    axis.set_xlabel("frames selected per trial")
    axis.set_ylabel("coverage radius (standardised posture units)")
    axis.set_title("solid = farthest-point, dotted = uniform stride")
    axis.grid(True, alpha=0.25)
    axis.legend(
        handles=[plt.Line2D([], [], color=color, label=kind) for kind, color in KIND_COLORS.items()], fontsize=9
    )

    farthest_all, uniform_all = np.array(farthest_all), np.array(uniform_all)
    axis_gain.plot(budgets, uniform_all.mean(axis=0), "o:", color="tab:gray", label="uniform stride")
    axis_gain.plot(budgets, farthest_all.mean(axis=0), "o-", color="tab:green", label="farthest-point")
    axis_gain.fill_between(budgets, farthest_all.min(axis=0), farthest_all.max(axis=0), color="tab:green", alpha=0.15)
    axis_gain.axvline(FRAMES_PER_TRIAL, color="black", ls="--", lw=1)
    axis_gain.set_xlabel("frames selected per trial")
    axis_gain.set_ylabel("coverage radius")
    axis_gain.set_title("mean over the 8 trials (band = min/max)")
    axis_gain.grid(True, alpha=0.25)
    axis_gain.legend(fontsize=9)
    return figure


def plot_angle_ranges(data: dict):
    """Per trial and per angle: the span the selection reproduces vs the span the trial visited."""
    paths = data["paths"]
    nb_angles = data["per_trial"][paths[0]]["features"].shape[1]
    labels = ANGLE_LABELS[:nb_angles]

    figure, axes = plt.subplots(1, nb_angles, figsize=(2.4 * nb_angles, 5), constrained_layout=True, sharey=False)
    figure.suptitle("Range of each joint angle: whole trial vs selected frames", fontsize=14, fontweight="bold")

    for column, axis in enumerate(np.atleast_1d(axes)):
        for row, path in enumerate(paths):
            entry = data["per_trial"][path]
            values = entry["features"][:, column]
            picked = values[entry["picked"]]
            color = KIND_COLORS[trial_kind(path)]
            axis.plot([row, row], [values.min(), values.max()], color=color, lw=6, alpha=0.28)
            axis.plot([row, row], [picked.min(), picked.max()], color=color, lw=2.2)
        axis.set_xticks(range(len(paths)), [trial_label(path) for path in paths], rotation=90, fontsize=7)
        axis.set_title(labels[column], fontsize=10)
        axis.grid(True, axis="y", alpha=0.25)
        if column == 0:
            axis.set_ylabel("standardised angle (thick = trial, thin = selection)")
    return figure


def plot_contact_patch(data: dict):
    """
    The scapula contact point in the thorax frame -- the patch the ellipsoid has to be fitted to.

    Small is the point: a patch of this size has almost no curvature signal, which is why the
    ellipsoid radius needs a prior (see ``shoulder_calibration.PARAMETER_PRIOR``).
    """
    all_cloud, selected_cloud = data["cloud_all"] * 1000, data["cloud_selected"] * 1000
    planes = [(0, 1, "X (antero-posterior)", "Y"), (0, 2, "X", "Z (medio-lateral)"), (2, 1, "Z", "Y")]

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    extent = np.ptp(all_cloud, axis=1)
    figure.suptitle(
        f"Scapula contact point in the thorax frame — the patch spans only "
        f"{extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} mm",
        fontsize=14,
        fontweight="bold",
    )

    for axis, (first, second, xlabel, ylabel) in zip(axes, planes):
        axis.scatter(all_cloud[first], all_cloud[second], s=5, color="tab:gray", alpha=0.25, label="all frames")
        axis.scatter(
            selected_cloud[first], selected_cloud[second], s=26, color="tab:red", alpha=0.85, label="selected"
        )
        axis.set_xlabel(f"{xlabel} (mm)")
        axis.set_ylabel(f"{ylabel} (mm)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)
    return figure


def main():
    arguments = parse_args(__doc__)
    paths = trials()
    print(f"gathering postures from {len(paths)} trials ...")
    data = gather(paths)

    figures = {
        "frame_selection_latent_space": plot_latent_space(data),
        "frame_selection_coverage": plot_coverage_curve(data),
        "frame_selection_angle_ranges": plot_angle_ranges(data),
        "frame_selection_contact_patch": plot_contact_patch(data),
    }
    finish(figures, arguments.save)


if __name__ == "__main__":
    main()
