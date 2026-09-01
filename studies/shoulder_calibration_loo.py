"""
Study - does the shoulder calibration generalise? Leave-one-trial-out cross-validation.

A calibration that is only ever scored on the trials it was fitted to proves nothing: with 13 free
parameters it can always improve its own training fit. So the session's 8 trials are split 8 ways.
Each fold calibrates on 7 trials -- **including the segment geometry**, which
:class:`~examples._shared.c3d_data.MultiC3dData` rebuilds from those 7 only -- and is then scored on
the 8th, which the fold has never seen.

What the folds are asked:

* **does it transfer?** marker RMSE on the held-out trial, per marker group. Four reconstructions
  are scored so that two effects that pull in opposite directions can be told apart: calibrating the
  parameters should improve the fit, while adding the scapulothoracic ellipsoid removes a degree of
  freedom and can only worsen it. See :func:`evaluate`. The train-minus-test gap says whether the
  fit is memorising.
* **is the ellipsoid stable?** the honest answer needs both halves. The *surface*, sampled over the
  patch the scapula actually visits, is what the model uses, and the folds measure it directly. The
  *parameters* look stable too -- but they are held near the subject's thorax by an explicit ridge,
  so their small across-fold SD is largely inherited from that prior rather than earned from the
  data. The identifiability evidence is the flatness of the objective, not the fold spread.
* **do the joint centres transfer?** the glenoid and humeral-head centres calibrated on 7 trials,
  measured on the 8th: how far apart do they reconstruct when nothing forces them together? Nothing
  in the prior determines this, so it is a real out-of-sample test.
* **what did closing the loop cost?** step 1 and step 2 are re-solved jointly in step 3; the drift
  from one to the other says how much the two halves disagree.

Run (from the repo root, inside the ``bionc`` conda env):
    python studies/shoulder_calibration_loo.py

Folds are cached under ``results/loo/`` and skipped on a re-run, so an interrupted sweep resumes.
This script only computes and reports; the figures live in :mod:`studies.figures.leave_one_out`,
which reads that cache, so you can redraw them without re-solving anything.
"""

from pathlib import Path


import numpy as np

from examples._shared.c3d_data import MultiC3dData
from examples._shared.frames import scs_to_natural
from examples._shared.ik import solve_trial
from examples.clinical.model import build_model_constrained, build_model_free
from studies.shoulder_calibration import (
    FRAMES_PER_TRIAL,
    MARKER_SET,
    calibrate,
    ellipsoid_surface_distance_mm,
    to_segment_frame,
    trial_kind,
    trial_label,
    trials,
)

LOO_DIR = Path(__file__).resolve().parents[1] / "results" / "loo"
# Every 5th frame of each full trial (20 Hz). The metric is a per-frame spatial residual, not a
# temporal signal, so decimating costs nothing but turns three differential IK solves per
# (fold, trial) from ~20 s into ~8 s -- the difference between a 75 and a 35 minute sweep.
EVAL_STRIDE = 5


# --------------------------------------------------------------------------------- evaluation
def point_in_global(model, segment_name: str, position_natural, Q: np.ndarray) -> np.ndarray:
    """Trajectory ``(3, nb_frames)`` of a segment-fixed point, given natural coordinates ``Q``."""
    from bionc.bionc_numpy.natural_vector import NaturalVector

    interpolation = np.asarray(NaturalVector(np.asarray(position_natural).reshape(3)).interpolate(), dtype=float)
    segment = model.segments[segment_name]
    block = slice(12 * segment.index, 12 * segment.index + 12)
    return interpolation @ np.asarray(Q)[block, :]


def evaluate(result, free_model, reference_model, path: str, stride: int = EVAL_STRIDE) -> dict:
    """
    Score one fold on one whole trial, against three references chosen to separate two effects.

    Calibrating the parameters should *improve* the fit; adding the scapulothoracic ellipsoid
    *removes a degree of freedom* and can only worsen it. Comparing the final model straight to the
    uncalibrated one confounds the two, so four reconstructions are scored on every trial:

    * ``free_model`` -- everything FREE: the achievable floor. Because it constrains nothing it is
      also the unbiased reconstruction to measure the calibrated *geometry* against (where it puts
      the contact point, how far apart it leaves the two glenohumeral centres).
    * ``reference_model`` -- the uncalibrated constrained model: clavicle + spherical GH on ``RGJC``,
      scapulothoracic free. What you get without any of this.
    * ``result.step2.model`` -- same constraint set as the reference, calibrated parameters. The
      difference between these two is the value of the calibration alone.
    * ``result.model`` (step 3) -- the above plus the closed-loop ellipsoid. The difference from
      step 2 is what that extra constraint costs.
    """
    calibrated = solve_trial(result.model, path, stride=stride)
    free = solve_trial(free_model, path, stride=stride)
    reference = solve_trial(reference_model, path, stride=stride)
    calibrated_free_st = solve_trial(result.step2.model, path, stride=stride)

    # Where the scapula actually goes on this trial, versus the ellipsoid calibrated without it.
    # The point measured is the model's own contact point (SCAP_CENTROID), carried by the *FREE*
    # reconstruction: on the calibrated model the ellipsoid constraint is enforced by the IK, so
    # its distance to the surface would be zero by construction and the metric would say nothing.
    contact_natural = result.model.segments["RSCAPULA"].marker_from_name("SCAP_CENTROID").position
    contact_global = point_in_global(free_model, "RSCAPULA", contact_natural, free["Qopt"])
    cloud = to_segment_frame(free_model, free["Qopt"], contact_global, segment="THORAX")

    joint = result.model.joints["Scapulothoracic"]
    semi_axes = np.array([float(length) for length in joint.semi_axis_lengths])
    center_scs = result.step3.sol["ellipsoid_center_scs"]
    rotation = result.step3.sol.get("ellipsoid_axes_scs")
    surface_mm = ellipsoid_surface_distance_mm(cloud, semi_axes, center_scs, rotation)

    # how far apart the two calibrated centres land when nothing forces them together
    centres = result.step3.centres
    glenoid = point_in_global(free_model, "RSCAPULA", scs_to_natural(result.model, "RSCAPULA", centres["glenoid"]), free["Qopt"])
    head = point_in_global(free_model, "RHUMERUS", scs_to_natural(result.model, "RHUMERUS", centres["head"]), free["Qopt"])
    gh_gap_mm = np.linalg.norm(glenoid - head, axis=0) * 1000

    return dict(
        trial=trial_label(path),
        kind=trial_kind(path),
        rmse_mm=calibrated["rmse_mm"],
        rmse_by_group_mm=calibrated["rmse_by_group_mm"],
        free_rmse_mm=free["rmse_mm"],
        reference_rmse_mm=reference["rmse_mm"],
        calibrated_free_st_rmse_mm=calibrated_free_st["rmse_mm"],
        ellipsoid_surface_rms_mm=float(np.sqrt(np.mean(surface_mm**2))),
        ellipsoid_surface_bias_mm=float(np.mean(surface_mm)),
        gh_gap_mean_mm=float(np.mean(gh_gap_mm)),
        gh_gap_max_mm=float(np.max(gh_gap_mm)),
    )


# ------------------------------------------------------------------------------------- folds
def _fold_payload(result, evaluations, held_out: str) -> dict:
    """Flatten a fold to the arrays and scalars worth keeping on disk."""
    payload = {
        "held_out": held_out,
        "train": np.array(result.train_paths),
        "parameter_labels": np.array(list(result.parameters)),
        "parameters": np.array(list(result.parameters.values())),
        "step1_semi_axes": result.step1.sol["semi_axes"],
        "step1_center_scs": result.step1.sol["ellipsoid_center_scs"],
        "step3_semi_axes": result.step3.sol["semi_axes"],
        "step3_center_scs": result.step3.sol["ellipsoid_center_scs"],
        # identity unless the orientation was calibrated; kept so a model can be rebuilt from the cache
        "step3_axes_scs": result.step3.sol.get("ellipsoid_axes_scs", np.eye(3)),
        "step2_glenoid": result.step2.centres["glenoid"],
        "step2_head": result.step2.centres["head"],
        "step3_glenoid": result.step3.centres["glenoid"],
        "step3_head": result.step3.centres["head"],
        "step2_clavicle": result.step2.sol["parameters"]["Clavicle.length"],
        "step3_clavicle": result.step3.sol["parameters"]["Clavicle.length"],
        "train_rmse_mm": result.step3.sol["marker_rmse_mm"],
        "at_bounds": np.array(result.step3.sol["parameters_at_bounds"]),
    }
    for key in ("trial", "kind"):
        payload[f"eval_{key}"] = np.array([evaluation[key] for evaluation in evaluations])
    for key in (
        "rmse_mm",
        "free_rmse_mm",
        "reference_rmse_mm",
        "calibrated_free_st_rmse_mm",
        "ellipsoid_surface_rms_mm",
        "ellipsoid_surface_bias_mm",
        "gh_gap_mean_mm",
        "gh_gap_max_mm",
    ):
        payload[f"eval_{key}"] = np.array([evaluation[key] for evaluation in evaluations])
    for group in evaluations[0]["rmse_by_group_mm"]:
        payload[f"eval_rmse_{group}"] = np.array([evaluation["rmse_by_group_mm"][group] for evaluation in evaluations])
    return payload


def run_fold(held_out: str, paths: list[str], *, frames_per_trial: int, stride: int, cache: bool = True) -> dict:
    """Calibrate on every trial but ``held_out``, then score the fold on all of them."""
    LOO_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = LOO_DIR / f"fold_{trial_label(held_out)}.npz"
    if cache and cache_file.exists():
        print(f"[{trial_label(held_out)}] cached")
        return dict(np.load(cache_file, allow_pickle=True))

    train = [path for path in paths if path != held_out]
    print(f"[{trial_label(held_out)}] calibrating on {', '.join(trial_label(path) for path in train)}")
    result = calibrate(train, frames_per_trial=frames_per_trial, marker_set=MARKER_SET, verbose=False)

    # the baselines have to share the fold's geometry, or the comparison is not like for like
    data = MultiC3dData(train)
    free_model = build_model_free(data, marker_set=MARKER_SET)
    reference_model = build_model_constrained(data, marker_set=MARKER_SET)

    evaluations = [evaluate(result, free_model, reference_model, path, stride=stride) for path in paths]
    payload = _fold_payload(result, evaluations, held_out)
    np.savez(cache_file, **payload)
    for line in result.summary().splitlines()[1:4]:  # the three step lines, not the parameter dump
        print(f"[{trial_label(held_out)}]{line}")
    return payload


# ------------------------------------------------------------------------------------ report
def held_out_mask(fold: dict) -> np.ndarray:
    return fold["eval_trial"] == trial_label(str(fold["held_out"]))


def summarise(folds: list[dict]) -> str:
    lines = ["", "===== leave-one-trial-out: marker RMSE on the held-out trial (mm) =====", ""]
    lines.append(
        f"{'held out':<12s} {'FREE':>7s} {'uncal.':>8s} {'calib.':>8s} {'+ellips':>8s}"
        f" {'train':>7s} {'gap':>7s} {'surf RMS':>9s} {'GH gap':>8s}"
    )
    columns = {key: [] for key in ("free", "reference", "step2", "step3", "gap", "surface", "bias", "gh_gap")}
    for fold in folds:
        mask = held_out_mask(fold)
        row = dict(
            free=float(fold["eval_free_rmse_mm"][mask][0]),
            reference=float(fold["eval_reference_rmse_mm"][mask][0]),
            step2=float(fold["eval_calibrated_free_st_rmse_mm"][mask][0]),
            step3=float(fold["eval_rmse_mm"][mask][0]),
            surface=float(fold["eval_ellipsoid_surface_rms_mm"][mask][0]),
            bias=float(fold["eval_ellipsoid_surface_bias_mm"][mask][0]),
            gh_gap=float(fold["eval_gh_gap_mean_mm"][mask][0]),
        )
        train = float(fold["train_rmse_mm"])
        row["gap"] = row["step3"] - train
        lines.append(
            f"{trial_label(str(fold['held_out'])):<12s} {row['free']:7.2f} {row['reference']:8.2f}"
            f" {row['step2']:8.2f} {row['step3']:8.2f} {train:7.2f} {row['gap']:7.2f}"
            f" {row['surface']:9.2f} {row['gh_gap']:8.2f}"
        )
        for key, value in row.items():
            columns[key].append(value)

    mean = {key: float(np.mean(value)) for key, value in columns.items()}
    sd = {key: float(np.std(value)) for key, value in columns.items()}
    lines += [
        "",
        "  FREE    = every joint free, the achievable floor",
        "  uncal.  = clavicle + spherical GH on RGJC, scapulothoracic free, nothing calibrated",
        "  calib.  = same constraints, calibrated clavicle length and glenohumeral centres (step 2)",
        "  +ellips = and the closed-loop scapulothoracic ellipsoid (step 3), which removes a DoF",
        "",
        f"FREE floor          {mean['free']:.2f} +- {sd['free']:.2f} mm",
        f"uncalibrated        {mean['reference']:.2f} +- {sd['reference']:.2f} mm",
        f"calibrated (step 2) {mean['step2']:.2f} +- {sd['step2']:.2f} mm"
        f"   -> calibration alone: {mean['step2'] - mean['reference']:+.2f} mm vs uncalibrated",
        f"calibrated (step 3) {mean['step3']:.2f} +- {sd['step3']:.2f} mm"
        f"   -> the ellipsoid costs a further {mean['step3'] - mean['step2']:+.2f} mm",
        f"train-test gap      {mean['gap']:+.2f} +- {sd['gap']:.2f} mm",
        "",
        f"ellipsoid surface RMS on held-out trial   {mean['surface']:.2f} +- {sd['surface']:.2f} mm",
        f"  signed bias ranges {min(columns['bias']):+.2f} to {max(columns['bias']):+.2f} mm across folds"
        " -- it changes sign,",
        "  so the calibrated surface sits inside the held-out contact path on some folds and outside",
        "  on others, by as much as the within-fold scatter of the contact point itself (~5 mm).",
        "  The ellipsoid transfers loosely; it is not pinned down to a millimetre by 7 trials.",
        f"glenohumeral centre gap on held-out trial {mean['gh_gap']:.2f} +- {sd['gh_gap']:.2f} mm",
    ]

    lines += ["", "--- calibrated parameters across folds (mm) ---",
              f"{'parameter':<38s} {'mean':>9s} {'sd':>8s} {'min':>9s} {'max':>9s}"]
    labels = list(folds[0]["parameter_labels"])
    values = np.array([fold["parameters"] for fold in folds]) * 1000
    for index, label in enumerate(labels):
        column = values[:, index]
        lines.append(f"{label:<38s} {column.mean():9.2f} {column.std():8.2f} {column.min():9.2f} {column.max():9.2f}")

    at_bounds = sorted({str(name) for fold in folds for name in fold["at_bounds"]})
    lines.append("")
    lines.append("parameters riding a bound in at least one fold: " + (", ".join(at_bounds) if at_bounds else "none"))
    lines.append(
        "\nHow to read the parameter table: the across-fold SD is small, but do NOT read that as the"
        "\ndata pinning the ellipsoid down. The semi-axes are held near the subject's thorax size by"
        "\nan explicit ridge (see shoulder_calibration.PARAMETER_PRIOR), and the thorax reference is"
        "\nitself nearly identical from fold to fold -- so most of that stability is inherited from"
        "\nthe prior, not earned from the data. The identifiability evidence is the flatness of the"
        "\nobjective (a 0.05 mm change in surface residual across a ~4x change in radius, against a"
        "\n5.1 mm noise floor), not this SD. What the folds *do* measure out of sample is the"
        "\nsurface RMS and the glenohumeral centres, neither of which the prior determines -- and"
        "\nthe surface bias above shows the ellipsoid transfers only loosely."
    )
    return "\n".join(lines)


def write_csv(folds: list[dict], path: Path) -> None:
    """One row per (fold, evaluated trial), for whatever comes next."""
    columns = [key for key in folds[0] if key.startswith("eval_")]
    with open(path, "w") as stream:
        stream.write("held_out,is_held_out," + ",".join(key[len("eval_") :] for key in columns) + "\n")
        for fold in folds:
            mask = held_out_mask(fold)
            for row in range(len(fold["eval_trial"])):
                values = ",".join(
                    f"{fold[key][row]:.4f}" if fold[key].dtype.kind == "f" else str(fold[key][row]) for key in columns
                )
                stream.write(f"{trial_label(str(fold['held_out']))},{int(mask[row])},{values}\n")


def main():
    paths = trials()
    folds = [run_fold(path, paths, frames_per_trial=FRAMES_PER_TRIAL, stride=EVAL_STRIDE) for path in paths]

    print(summarise(folds))
    write_csv(folds, LOO_DIR / "summary.csv")
    print(f"\nper-(fold, trial) rows written to {LOO_DIR / 'summary.csv'}")
    print("figures: python studies/figures/leave_one_out.py")


if __name__ == "__main__":
    main()
