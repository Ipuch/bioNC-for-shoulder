# Studies — small shoulder investigations

These are short analysis scripts that go **beyond the tutorial** in
[`examples/`](../examples/). Each one reuses the dataset model builders from
`examples/<dataset>/model.py` and the shared IK/plotting helpers, so no modelling code is
duplicated here — a study only adds its specific experiment on top.

Run from the **repo root**, inside the `bionc` conda environment:

```bash
conda activate bionc
python studies/gh_constraint_comparison.py
```

| Script | Dataset | Question |
| --- | --- | --- |
| [`gh_constraint_comparison.py`](gh_constraint_comparison.py) | Henninger | How does constraining only the glenohumeral joint (free vs spherical vs constant-length) affect the marker fit (RMSE) and the joint angles, relative to a fully-free "raw" reconstruction? |
| [`marker_set_comparison.py`](marker_set_comparison.py) | clinical | How sensitive are the reconstructed joint angles to the IK marker set (skin clusters only vs anatomical landmarks only vs both)? |
| [`scapulothoracic_ellipsoid_calibration.py`](scapulothoracic_ellipsoid_calibration.py) | clinical | Calibrate a thoracic ellipsoid on which the scapula glides (Naaim 2016/2017) by solving one all-frames inverse kinematics whose shared variables include the ellipsoid semi-axes and centre; supports both the tangent (ELLIPSOID_ON_PLANE) and one-point (POINT_ON_ELLIPSOID) joints, and compares the scapulothoracic angles to the FREE baseline. |

> `scapulothoracic_ellipsoid_calibration.py` builds a single CasADi/IPOPT NLP over every
> (subsampled) frame at once — the natural coordinates of all frames **and** the six ellipsoid
> parameters `(a, b, c, cx, cy, cz)` are optimised together, with the rigid-body/joint constraints
> enforced on each frame. The `KinematicCalibration` class in that file is distilled from bionc's
> `InverseKinematics`. Because the whole trial is one NLP, it is the heaviest study; reduce
> `STRIDE`/frame count if it is slow, and switch `ELLIPSOID_MODELS` to pick the joint(s) to calibrate.
