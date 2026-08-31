# Studies — small shoulder investigations

These are short analysis scripts that go **beyond the tutorial** in
[`examples/`](../examples/). Each one reuses the dataset model builders from
`examples/<dataset>/model.py` and the shared IK/plotting helpers, so no modelling code is
duplicated here — a study only adds its specific experiment on top.

Run from the **repo root**, inside the `bionc` conda environment:

```bash
conda activate bionc
PYTHONPATH=$HOME/ProjetsPython/bioNC python studies/gh_constraint_comparison.py
```

(`bionc` itself is not installed in that environment — it resolves through the sibling source tree.
Either set `PYTHONPATH` as above or run `pip install -e ../bioNC` once.)

| Script | Dataset | Question |
| --- | --- | --- |
| [`gh_constraint_comparison.py`](gh_constraint_comparison.py) | Henninger | How does constraining only the glenohumeral joint (free vs spherical vs constant-length) affect the marker fit (RMSE) and the joint angles, relative to a fully-free "raw" reconstruction? |
| [`marker_set_comparison.py`](marker_set_comparison.py) | clinical | How sensitive are the reconstructed joint angles to the IK marker set (skin clusters only vs anatomical landmarks only vs both)? |
| [`scapulothoracic_ellipsoid_calibration.py`](scapulothoracic_ellipsoid_calibration.py) | clinical | Calibrate a thoracic ellipsoid on which the scapula glides (Naaim 2016/2017) by solving one all-frames inverse kinematics whose shared variables include the ellipsoid semi-axes and centre; supports both the tangent (ELLIPSOID_ON_PLANE) and one-point (POINT_ON_ELLIPSOID) joints, and compares the scapulothoracic angles to the FREE baseline. |
| [`shoulder_calibration.py`](shoulder_calibration.py) | `99007140-*` (8 trials) | Calibrate the whole chain from a *whole session* at once: the thoracic ellipsoid, the clavicle length, and the glenoid / humeral-head centres. Three steps, importable as `calibrate(train_paths)`. |
| [`shoulder_calibration_loo.py`](shoulder_calibration_loo.py) | `99007140-*` (8 trials) | Does that calibration generalise? Leave-one-trial-out: 8 folds, each calibrated on 7 trials and scored on the 8th. |

## The all-frames calibration engine

[`kinematic_calibration.py`](kinematic_calibration.py) builds a single CasADi/IPOPT NLP over every
(subsampled) frame at once — the natural coordinates of all frames **and** the shared model
parameters are optimised together, with the rigid-body/joint constraints enforced on each frame.
`KinematicCalibration` is distilled from bionc's `InverseKinematics`.

What gets calibrated is a list of parameter objects, so the same engine covers every study:

| Parameter | What it makes symbolic |
| --- | --- |
| `EllipsoidSemiAxes` | an ellipsoid joint's `(a, b, c)` |
| `MarkerPosition` | a virtual point — an ellipsoid centre, a functional joint centre — in **segment coordinates, in metres** |
| `EllipsoidOrientation` | the principal axes, as a Rodrigues rotation increment |
| `JointLength` | a `CONSTANT_LENGTH` joint's length |

Because the objective and constraints are separable per frame, frames from several trials are just
concatenated. Each parameter carries an optional ridge weight (`prior`) toward its warm start, and
`sol()` reports **which parameters ended on a bound** — a parameter riding a bound is constrained,
not calibrated, and its value means nothing.

## The three-step session calibration

`shoulder_calibration.calibrate(train_paths)`:

1. **THORAX + RSCAPULA only**, joined *only* by a one-point scapulothoracic ellipsoid joint. Nothing
   else can absorb the residual → semi-axes and centre (orientation optionally, off by default).
2. **Full chain, scapulothoracic FREE**, clavicle `CONSTANT_LENGTH`, glenohumeral `SPHERICAL` on two
   dedicated centres → clavicle length, glenoid centre, humeral-head centre. A SCoRE-style
   functional joint-centre estimation written as a constrained IK.
3. **Everything at once**, closing the loop, warm-started from steps 1 and 2.

Two identifiability facts drive the design, and both are measured rather than assumed:

* **The glenohumeral joint gets no calibrated length.** With the two centres free as well the
  constraint is rank deficient: if `(c_glenoid, c_head, L)` fits, so does `(c_glenoid, c_head + d,
  ‖d‖)` for *any* `d`, since `‖P_s − P_h‖ = ‖R_h d‖` is then constant. The centres are the
  identifiable half. Add `JointLength("Glenohumeral")` once other data pins the centres.
* **A contact patch does not determine an ellipsoid.** The scapula sweeps roughly 55 × 52 × 67 mm on
  this subject; across that patch the best-fit surface residual moves by **0.05 mm** as the radius
  goes from 80 mm to 300 mm, against a **5.1 mm** noise floor — a curvature signal two orders of
  magnitude below the noise. The semi-axes are therefore held near the subject's own thorax size by
  an explicit, reported ridge. `studies/figures/frame_selection.py` draws the patch; the
  leave-one-out study shows what the folds can and cannot establish about it.

## Figures

Computation and figures are kept apart: the calibrations take minutes to an hour, figures are what
you iterate on. Everything under [`figures/`](figures/) reads cached arrays and redraws in seconds.

```bash
python studies/figures/frame_selection.py          # is the calibration subsample diverse enough?
python studies/figures/calibration_steps.py        # what each of the three steps did
python studies/figures/leave_one_out.py            # does it generalise?
python studies/figures/replay_calibrated.py --trial FUNCTIONAL1   # rerun: model + ellipsoid + markers
python studies/figures/frame_selection_replay.py   # rerun: the calibration frames as postures
```

Each script runs either as a plain file (above) or as `python -m studies.figures.<name>`. `--save`
writes PNGs to `results/figures/` instead of opening windows; `--refresh` recomputes a script's own
cache (`calibration_steps.py` keeps one in `results/calibration/steps.npz`). `leave_one_out.py` and
`replay_calibrated.py` need `shoulder_calibration_loo.py` to have run first.

| Script | Answers |
| --- | --- |
| [`figures/frame_selection.py`](figures/frame_selection.py) | Posture latent space (PCA) with the selected frames on top; coverage radius vs frame budget, farthest-point against a uniform stride; per-angle range covered; and the contact-point patch that limits the ellipsoid. |
| [`figures/calibration_steps.py`](figures/calibration_steps.py) | One figure per step: the ellipsoid cut through the contact patch with before/after surface residuals; the two glenohumeral centres and the clavicle scatter a constant length has to absorb; the per-frame fit of every step against the FREE floor plus the step 1/2 → step 3 drift. |
| [`figures/leave_one_out.py`](figures/leave_one_out.py) | The four cross-validation figures, read from `results/loo/`. |
| [`figures/replay_calibrated.py`](figures/replay_calibrated.py) | **rerun replay of any trial with any fold's calibrated model**, with the experimental markers, the thoracic ellipsoid riding on the thorax and the contact point riding on the scapula. **Needs a graphical session.** |
| [`figures/frame_selection_replay.py`](figures/frame_selection_replay.py) | rerun replay of *only* the calibration frames, to look at the postures themselves. **Needs a graphical session.** |

### Replaying a calibrated model

`replay_calibrated.py` rebuilds the model from the numbers a fold cached
(`shoulder_calibration.rebuild_calibrated_model`) rather than re-solving, so it opens in seconds.
The segment geometry is rebuilt from that fold's *training* trials, which is what makes the cached
natural coordinates mean the same thing again.

```bash
python studies/figures/replay_calibrated.py --trial FUNCTIONAL1                    # held-out replay
python studies/figures/replay_calibrated.py --trial ANALYTIC2 --fold FUNCTIONAL4   # cross-fold
python studies/figures/replay_calibrated.py --trial ANALYTIC3 --compare-free       # vs all-FREE
```

`--trial X` on its own uses the fold that **held X out**, so you are watching the calibration work
on data it never saw; the script prints which case you got. `--compare-free` adds the all-FREE
reconstruction as a second toggleable model. The contact point stays on the ellipsoid surface to
~1e-10 because the IK enforces that constraint — if it visibly leaves the surface, something is
wrong with the rebuild, not with the calibration.

The frame-selection figures answer the "is there enough data" question directly: farthest-point
sampling at 35 frames/trial covers the posture space better than a uniform stride at 80, and the
coverage curve flattens around 25–35 frames, which is where the budget was set.

## Reading the leave-one-out results

`shoulder_calibration_loo.py` caches each fold under `results/loo/` (gitignored) and skips folds
already computed, so an interrupted sweep resumes. It writes `results/loo/summary.csv` with one row
per (fold, evaluated trial); the figures come from `figures/leave_one_out.py`.

**Score four models, not two.** Calibrating the parameters should improve the held-out fit, while
adding the scapulothoracic ellipsoid removes a degree of freedom and can only worsen it. Comparing
the final model straight to the uncalibrated one confounds the two, and makes the calibration look
worse than it is. So every held-out trial is reconstructed four ways — all-FREE (the floor),
uncalibrated constrained, calibrated with the scapulothoracic still free (step 2), and calibrated
with the ellipsoid closed (step 3). The step 2 − uncalibrated difference is the value of the
calibration; the step 3 − step 2 difference is the price of the extra constraint.

**Do not read the small parameter SD as identifiability.** The semi-axes are held near the
subject's thorax by an explicit ridge, and the thorax reference barely moves between folds, so most
of that stability is inherited from the prior rather than earned from the data. The identifiability
evidence is the flatness of the objective described above.

What the folds *do* measure out of sample — because the prior does not determine them — are the
held-out surface agreement and the glenohumeral centres. Read the surface honestly: the RMS is
6.1 ± 2.8 mm and the **mean signed distance changes sign between folds, −7.7 to +8.2 mm**, so the
calibrated surface sits inside the held-out contact path on some folds and outside on others, by
about as much as the contact point's own within-fold scatter (~5 mm). The ellipsoid transfers
*loosely*; seven trials of one subject do not pin it to the millimetre. The tight semi-axis plot in
`loo_ellipsoid_agreement.png` and this bias are the same story told from two ends.

Every number is trial-to-trial generalisation on **one subject**. It says nothing about how the
calibration transfers between people.

## A note on `bionc`'s segment-coordinate conversion

`NaturalSegment.compute_transformation_matrix()` returns `Mᵀ`, where `M` maps natural to orthonormal
segment coordinates. bionc's `add_natural_marker_from_segment_coordinates` and
`add_natural_vector_from_segment_coordinates` then convert with `inv(Mᵀ)` where they need `inv(M)`.
For a marker that is a few millimetres; for the ellipsoid axes it is structural — it turns the
principal triad into a skewed one (axis norms 0.93 / 1.31 / 1.00 and mutual dot products up to 0.56
on this subject), so the reported "semi-axis lengths" stop being geometric semi-axes.

These studies therefore build that geometry through `examples/_shared/frames.py`
(`add_marker_from_scs`, `add_vector_from_scs`, `natural_to_scs`, `scs_to_natural`), which use `M`.
With the correct conversion the axes come out orthonormal and the calibrated surface-distance metric
agrees exactly with bionc's own joint constraint. **This is an upstream bug worth fixing in
`bioNC`.**
