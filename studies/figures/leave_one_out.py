"""
Does the calibration generalise? The leave-one-trial-out figures.

Reads the folds cached by :mod:`studies.shoulder_calibration_loo` under ``results/loo/`` -- run that
first; it is the part that takes the better part of an hour. Four figures:

* **held-out RMSE per fold**, with the two references it has to sit between, and split by marker
  group. Calibrating the parameters improves the held-out fit; adding the scapulothoracic ellipsoid
  removes a degree of freedom and costs some of it back, so both are drawn.
* **parameter spread across folds** -- read with the caveat the driver prints: the ellipsoid
  parameters are held near the subject's thorax by a ridge, so their small scatter is mostly
  inherited from that prior rather than earned from the data.
* **ellipsoid: parameters vs surface** -- the semi-axes fold by fold next to how well the held-out
  trial's contact point still lands on the calibrated surface. Read the two together: the semi-axes
  look tight because a ridge holds them there, while the signed surface distance changes sign
  between folds by roughly the contact point's own scatter. The ellipsoid transfers loosely.
* **step 1 / step 2 -> step 3 drift** -- how far closing the loop moved what the two halves had
  each decided on their own.

Run:
    python studies/figures/leave_one_out.py [--save]
"""

import numpy as np
from matplotlib import pyplot as plt

from studies.shoulder_calibration import trial_kind, trial_label
from studies.shoulder_calibration_loo import LOO_DIR, held_out_mask
from studies.figures import finish, parse_args


def load_folds() -> list[dict]:
    """Every cached fold, in trial order. Raises if the sweep has not been run."""
    files = sorted(LOO_DIR.glob("fold_*.npz"))
    if not files:
        raise SystemExit(f"no folds cached in {LOO_DIR}. Run `python studies/shoulder_calibration_loo.py` first.")
    print(f"loaded {len(files)} folds from {LOO_DIR}")
    return [dict(np.load(path, allow_pickle=True)) for path in files]


def plot_rmse(folds: list[dict]):
    """Held-out RMSE per fold against the two references it has to sit between."""
    labels = [trial_label(str(fold["held_out"])) for fold in folds]
    pick = lambda key: [float(fold[key][held_out_mask(fold)][0]) for fold in folds]

    figure, (axis, axis_group) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    figure.suptitle("Held-out marker RMSE, leave-one-trial-out", fontsize=14, fontweight="bold")

    position = np.arange(len(labels))
    axis.bar(
        position - 0.2,
        pick("eval_calibrated_free_st_rmse_mm"),
        0.4,
        color="tab:green",
        label="calibrated, scapulothoracic free (step 2)",
    )
    axis.bar(position + 0.2, pick("eval_rmse_mm"), 0.4, color="tab:red", label="+ ellipsoid (step 3)")
    axis.plot(position, pick("eval_free_rmse_mm"), "o--", color="tab:gray", label="all-FREE floor")
    axis.plot(position, pick("eval_reference_rmse_mm"), "s--", color="tab:blue", label="uncalibrated constrained")
    axis.plot(
        position,
        [float(fold["train_rmse_mm"]) for fold in folds],
        "^:",
        color="tab:orange",
        label="step 3, training fit",
    )
    axis.set_xticks(position, labels, rotation=45, ha="right")
    axis.set_ylabel("marker RMSE (mm)")
    axis.set_title("per fold — calibration gain vs the cost of the extra constraint")
    axis.grid(True, axis="y", alpha=0.25)
    axis.set_ylim(0, max(pick("eval_rmse_mm")) * 1.45)  # headroom so the legend clears the bars
    axis.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.95)

    groups = [key[len("eval_rmse_") :] for key in folds[0] if key.startswith("eval_rmse_") and key != "eval_rmse_mm"]
    for offset, group in enumerate(groups):
        values = [float(fold[f"eval_rmse_{group}"][held_out_mask(fold)][0]) for fold in folds]
        axis_group.bar(position + (offset - len(groups) / 2 + 0.5) * 0.25, values, 0.25, label=group)
    axis_group.set_xticks(position, labels, rotation=45, ha="right")
    axis_group.set_ylabel("marker RMSE (mm)")
    axis_group.set_title("held-out RMSE by marker group — the scapula is the hard part")
    axis_group.grid(True, axis="y", alpha=0.25)
    axis_group.legend(fontsize=9)
    return figure


def plot_parameter_spread(folds: list[dict]):
    """Every calibrated parameter, fold by fold, normalised so they share one axis."""
    labels = list(folds[0]["parameter_labels"])
    values = np.array([fold["parameters"] for fold in folds]) * 1000
    kinds = [trial_kind(str(fold["held_out"])) for fold in folds]

    figure, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    figure.suptitle(
        "Calibrated parameters across the 8 folds (deviation from their mean)", fontsize=14, fontweight="bold"
    )
    for index, label in enumerate(labels):
        column = values[:, index]
        for fold_index, value in enumerate(column):
            axis.scatter(
                index,
                value - column.mean(),
                color="tab:red" if kinds[fold_index] == "ANALYTIC" else "tab:blue",
                alpha=0.75,
                s=45,
            )
        axis.errorbar(index, 0, yerr=column.std(), color="black", capsize=6, lw=1.5)
    axis.axhline(0, color="tab:gray", lw=1, ls="--")
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    axis.set_ylabel("deviation from the across-fold mean (mm)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color="tab:red", label="ANALYTIC held out"),
            plt.Line2D([], [], marker="o", ls="", color="tab:blue", label="FUNCTIONAL held out"),
            plt.Line2D([], [], color="black", label="across-fold SD"),
        ],
        fontsize=9,
    )
    return figure


def plot_ellipsoid_agreement(folds: list[dict]):
    """
    The two ellipsoid answers side by side, and they do not agree.

    Left: the semi-axes barely move between folds -- but a ridge is holding them near the subject's
    thorax, so that is mostly the prior talking. Right: on the trial the fold never saw, the signed
    distance from the contact point to the calibrated surface swings from about -8 to +8 mm, which
    is the size of the contact point's own scatter. Tight parameters, loose surface.
    """
    labels = [trial_label(str(fold["held_out"])) for fold in folds]
    semi_axes = np.array([fold["step3_semi_axes"] for fold in folds]) * 1000
    surface = [float(fold["eval_ellipsoid_surface_rms_mm"][held_out_mask(fold)][0]) for fold in folds]
    bias = [float(fold["eval_ellipsoid_surface_bias_mm"][held_out_mask(fold)][0]) for fold in folds]

    figure, (axis_parameters, axis_surface) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    figure.suptitle("Ellipsoid: parameter spread vs surface agreement", fontsize=14, fontweight="bold")

    position = np.arange(len(labels))
    for index, name in enumerate("abc"):
        axis_parameters.plot(position, semi_axes[:, index], "o-", label=f"semi-axis {name}")
    axis_parameters.set_xticks(position, labels, rotation=45, ha="right")
    axis_parameters.set_ylabel("semi-axis (mm)")
    axis_parameters.set_title("calibrated semi-axes per fold")
    axis_parameters.grid(True, alpha=0.25)
    axis_parameters.legend()

    axis_surface.bar(position - 0.2, surface, 0.4, color="tab:green", label="RMS |distance|")
    axis_surface.bar(position + 0.2, bias, 0.4, color="tab:olive", label="mean signed distance")
    axis_surface.axhline(0, color="tab:gray", lw=1)
    axis_surface.set_xticks(position, labels, rotation=45, ha="right")
    axis_surface.set_ylabel("contact point to calibrated surface (mm)")
    axis_surface.set_title("held-out trial: does the scapula still ride the surface?")
    axis_surface.grid(True, axis="y", alpha=0.25)
    axis_surface.legend()
    return figure


def plot_step_drift(folds: list[dict]):
    """How far closing the loop in step 3 had to move what steps 1 and 2 had decided."""
    labels = [trial_label(str(fold["held_out"])) for fold in folds]
    position = np.arange(len(labels))
    series = {
        "ellipsoid semi-axes": np.linalg.norm(
            np.array([fold["step3_semi_axes"] - fold["step1_semi_axes"] for fold in folds]), axis=1
        ),
        "ellipsoid centre": np.linalg.norm(
            np.array([fold["step3_center_scs"] - fold["step1_center_scs"] for fold in folds]), axis=1
        ),
        "glenoid centre": np.linalg.norm(
            np.array([fold["step3_glenoid"] - fold["step2_glenoid"] for fold in folds]), axis=1
        ),
        "humeral head centre": np.linalg.norm(
            np.array([fold["step3_head"] - fold["step2_head"] for fold in folds]), axis=1
        ),
        "clavicle length": np.abs(np.array([fold["step3_clavicle"] - fold["step2_clavicle"] for fold in folds])),
    }

    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    figure.suptitle("Step 1 / step 2 -> step 3: what closing the loop changed", fontsize=14, fontweight="bold")
    width = 0.8 / len(series)
    for index, (name, values) in enumerate(series.items()):
        axis.bar(position + (index - len(series) / 2 + 0.5) * width, np.asarray(values) * 1000, width, label=name)
    axis.set_xticks(position, labels, rotation=45, ha="right")
    axis.set_ylabel("displacement (mm)")
    axis.set_xlabel("held-out trial")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=9)
    return figure


def main():
    arguments = parse_args(__doc__)
    folds = load_folds()
    figures = {
        "loo_rmse": plot_rmse(folds),
        "loo_parameter_spread": plot_parameter_spread(folds),
        "loo_ellipsoid_agreement": plot_ellipsoid_agreement(folds),
        "loo_step_drift": plot_step_drift(folds),
    }
    finish(figures, arguments.save)


if __name__ == "__main__":
    main()
