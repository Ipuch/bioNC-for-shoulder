# Examples — a progressive tour of bioNC for the shoulder

These examples build up, step by step, how to model the shoulder complex with
[bioNC](https://github.com/Ipuch/bioNC) and run an inverse kinematics (IK) on real motion
capture data. They are meant to be **read and run in order**.

Run everything from the **repo root**, inside the `bionc` conda environment:

```bash
conda activate bionc
python examples/henninger/01_build_model_no_constraints.py
```

## Two datasets, two parallel tutorials

The same three-step progression is provided for two **different** experimental datasets, which
use different marker conventions. Treat them as two separate examples of the same workflow.

| Folder | Data file | Marker convention |
| --- | --- | --- |
| [`henninger/`](henninger/) | `data/testFlorent_HenningerData.c3d` | `IJ, C7, PX, T5` (thorax), `AA, TS, IA, AC, GSC` (scapula), `GSChum, EL, EM` (humerus) |
| [`clinical/`](clinical/) | `data/testFlorent_clinicalData.c3d` | `SJN, CV7, SXS, TV8` (thorax), `RSAA, RSIA, RSRS, RCAJ` + `Cluster_RS_*` (scapula), `RGJC, RHME, RHLE` + `Cluster_RA_*` (humerus) |

## The three steps (per dataset)

1. **`01_build_model_no_constraints.py`** — build the model with **every joint FREE**. The
   segments are only tied together by the markers they share. The model is reconstructed
   directly from the markers (`Q_from_markers`, no optimisation) and shown against the raw data.
2. **`02_build_model_with_constraints.py`** — same segments, but the joints now carry anatomy:
   a constant-length *clavicle* and a *spherical* (ball-and-socket) glenohumeral joint; the
   scapulothoracic joint stays free. Still reconstructed directly from markers, to *see* the
   constrained model on the data.
3. **`03_inverse_kinematics.py`** — solve the IK on the constrained model: the optimiser
   **enforces** the joint constraints while staying close to the markers. Prints the marker
   RMSE, animates the reconstruction, and plots the joint angles (each joint in its own Euler
   sequence).

## How the code is organised (minimal duplication)

- **`_shared/`** holds everything that is not dataset-specific: the thorax frame helper
  (`frames.py`), the IK / joint-angle / RMSE / plotting helpers (`ik.py`) and the pyorerun
  animation boilerplate (`viz.py`). Written once, imported everywhere.
- Each dataset folder has a single **`model.py`** that defines its segments once and selects the
  joints by keyword (`build_model_free`, `build_model_constrained`, …). The numbered scripts are
  thin: they just call a builder and a shared helper.

Deeper analyses that go beyond the tutorial (comparing constraints, marker sets, calibrating a
scapulothoracic ellipsoid) live in the sibling [`studies/`](../studies/) folder and reuse these
same `model.py` builders.
