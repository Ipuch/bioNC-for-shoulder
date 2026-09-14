"""
Spherical vs constant-length glenohumeral joint, side by side.

Reads both leave-one-out sweeps -- ``results/loo/`` and ``results_gh_constant/loo/`` -- so both have
to have run. The two runs share steps 1 and 2 and differ only in step 3's glenohumeral joint, so
every difference drawn here is that joint's doing. Three panels:

* **the glenohumeral distance, fold by fold.** The spherical joint holds the two centres at 0 mm by
  construction; the constant-length joint holds them at a calibrated radius. Both are drawn against
  the separation the two centres actually show on the held-out trial, measured on an unconstrained
  reconstruction -- the one number neither calibration controls.
* **held-out marker RMSE of step 3**, spherical against constant length, with the shared step-2
  model and the all-FREE floor for scale.
* **where the fit changed**, constant length minus spherical, per marker group.

Run:
    python studies/figures/gh_comparison.py [--save]

``--save`` writes ``results_gh_constant/figures/gh_comparison.png``.
"""

import argparse

import numpy as np
from matplotlib import pyplot as plt

from studies.figures import finish
from studies.figures.leave_one_out import load_folds
from studies.shoulder_calibration import trial_label
from studies.shoulder_calibration_loo import held_out_mask

MODELS = {"spherical": "tab:blue", "constant_length": "tab:red"}
LABELS = {"spherical": "GH spherical", "constant_length": "GH constant length"}
GROUPS = ("THORAX", "RSCAPULA", "RHUMERUS")


def held_out(folds: list[dict], key: str) -> np.ndarray:
    """One value per fold: ``key`` on the trial that fold held out."""
    return np.array([float(fold[key][held_out_mask(fold)][0]) for fold in folds])


def gh_length_mm(folds: list[dict]) -> np.ndarray:
    """The distance step 3 imposes between the two centres; 0 for a spherical joint."""
    return np.array([float(fold.get("step3_gh_length", 0.0)) * 1000 for fold in folds])


def plot_comparison(runs: dict[str, list[dict]]):
    labels = [trial_label(str(fold["held_out"])) for fold in runs["spherical"]]
    if labels != [trial_label(str(fold["held_out"])) for fold in runs["constant_length"]]:
        raise SystemExit("the two sweeps do not hold out the same trials; re-run both")
    position = np.arange(len(labels))
    width = 0.38

    figure, (axis_length, axis_rmse, axis_group) = plt.subplots(
        1, 3, figsize=(18, 5.6), constrained_layout=True, gridspec_kw=dict(width_ratios=(1.25, 1.25, 0.8))
    )
    figure.suptitle(
        "Glenohumeral joint: spherical vs constant length (leave-one-trial-out)", fontsize=14, fontweight="bold"
    )

    # --- 1. the distance each joint imposes, against what the held-out trial shows
    for offset, (model, color) in zip((-width / 2, width / 2), MODELS.items()):
        lengths = gh_length_mm(runs[model])
        axis_length.bar(
            position + offset,
            lengths,
            width,
            color=color,
            label=f"{LABELS[model]}: imposed {lengths.mean():.2f} ± {lengths.std():.2f} mm",
        )
        gap = held_out(runs[model], "eval_gh_gap_mean_mm")
        axis_length.plot(
            position + offset,
            gap,
            "D",
            ms=8,
            mfc="white",
            mec="black" if model == "constant_length" else color,
            mew=2,
            zorder=3,  # above the bars, or the constant-length gap vanishes into its own bar
            label=f"{LABELS[model]}: held-out gap {gap.mean():.2f} ± {gap.std():.2f} mm",
        )
    for index in position:  # a zero-height bar is invisible, so say it
        axis_length.text(index - width / 2, 0.15, "0", ha="center", va="bottom", color=MODELS["spherical"], fontsize=8)
    axis_length.set_xticks(position, labels, rotation=45, ha="right")
    axis_length.set_ylabel("glenoid centre to humeral-head centre (mm)")
    axis_length.set_title("bars: distance step 3 imposes — diamonds: what the held-out trial shows")
    axis_length.grid(True, axis="y", alpha=0.25)
    axis_length.legend(fontsize=8, loc="upper left")
    axis_length.set_ylim(0, max(held_out(run, "eval_gh_gap_mean_mm").max() for run in runs.values()) * 1.45)

    # --- 2. held-out fit of step 3
    for offset, (model, color) in zip((-width / 2, width / 2), MODELS.items()):
        rmse = held_out(runs[model], "eval_rmse_mm")
        axis_rmse.bar(position + offset, rmse, width, color=color, label=f"{LABELS[model]} ({rmse.mean():.2f} mm)")
    step2 = held_out(runs["spherical"], "eval_calibrated_free_st_rmse_mm")
    floor = held_out(runs["spherical"], "eval_free_rmse_mm")
    axis_rmse.plot(position, step2, "s--", color="tab:green", label=f"step 2, shared ({step2.mean():.2f} mm)")
    axis_rmse.plot(position, floor, "o--", color="tab:gray", label=f"all-FREE floor ({floor.mean():.2f} mm)")
    axis_rmse.set_xticks(position, labels, rotation=45, ha="right")
    axis_rmse.set_ylabel("held-out marker RMSE (mm)")
    axis_rmse.set_title("step 3 on the trial each fold never saw")
    axis_rmse.grid(True, axis="y", alpha=0.25)
    axis_rmse.legend(fontsize=8, loc="lower right")

    # --- 3. which markers the looser joint helped
    for index, group in enumerate(GROUPS):
        delta = held_out(runs["constant_length"], f"eval_rmse_{group}") - held_out(
            runs["spherical"], f"eval_rmse_{group}"
        )
        axis_group.bar(index, delta.mean(), 0.6, color="tab:purple", alpha=0.35)
        axis_group.scatter(np.full(len(delta), index), delta, color="tab:purple", s=25, zorder=3)
        axis_group.text(index + 0.32, delta.mean(), f"{delta.mean():+.2f}", ha="left", va="center", fontsize=9)
    total = held_out(runs["constant_length"], "eval_rmse_mm") - held_out(runs["spherical"], "eval_rmse_mm")
    axis_group.axhline(0, color="black", lw=1)
    axis_group.set_xticks(range(len(GROUPS)), GROUPS)
    axis_group.set_xlim(-0.5, len(GROUPS) - 0.1)  # room for the value printed beside the last bar
    axis_group.set_ylabel("constant length − spherical (mm)")
    axis_group.set_title(f"held-out RMSE change by marker group\n(all markers: {total.mean():+.2f} mm)")
    axis_group.grid(True, axis="y", alpha=0.25)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", action="store_true", help="write the PNG to results_gh_constant/figures")
    arguments = parser.parse_args()

    runs = {model: load_folds(model) for model in MODELS}
    finish({"gh_comparison": plot_comparison(runs)}, arguments.save, "constant_length")


if __name__ == "__main__":
    main()
