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
| [`scapulothoracic_ellipsoid_calibration.py`](scapulothoracic_ellipsoid_calibration.py) | clinical | Calibrate, by bilevel optimisation, a thoracic ellipsoid on which the scapula glides (tangent ellipsoid-on-plane joint, Naaim 2016/2017), and compare its scapulothoracic angles to the free baseline. |

> `scapulothoracic_ellipsoid_calibration.py` runs a nested `scipy.optimize.least_squares` around
> the differential IK, so it is markedly slower than the other scripts.
