
# bioNC for Shoulder

This repository contains examples and modeling scripts built on top of [bioNC](https://github.com/Ipuch/bioNC), a Python library for biomechanics and natural coordinates.

The recommended workflow is:
1. create a Conda environment from `environment.yml`
2. install `bionc` from source in editable mode
3. run the examples from this repository

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd bioNC-for-shoulder
```

### 2. Create the Conda environment

The environment file includes the scientific stack and the dependencies needed to run the examples.

```bash
conda env create -f environment.yml
conda activate bionc
```

If you already have an existing environment, you can update it with:

```bash
conda env update -f environment.yml --prune
```

### 3. Install bioNC from source

This repository expects the local source tree to be used directly.

```bash
pip install -e .
```
or directly from the repo:
```bash
pip install -e git+https://github.com/Ipuch/bioNC.git#egg=bionc
```

Editable installation is the safest option while developing because changes in the source tree are immediately reflected without reinstalling.

### 4. Verify the installation

```bash
python -c "import bionc; print(bionc.__version__)"
```

If the import succeeds, the environment is ready.

## Environment Notes

The Conda environment is defined in [environment.yml](environment.yml). It currently includes:

- Python 3.11
- `numpy`, `scipy`, `casadi`
- `biorbd`
- `ezc3d`, `pyomeca`, `pyorerun`
- `matplotlib`, `plotly`, `numba`, `proxsuite`, `dill`

If `biorbd` is not available in your setup, install it from Conda Forge before running the examples.

## Running the examples

The [examples/](examples/) folder is a **progressive tutorial**: for each of the two datasets
(Henninger and clinical), you build the model without constraints, then with constraints, then
run the inverse kinematics. See [examples/README.md](examples/README.md) for the full tour.

Run everything from the repo root, inside the `bionc` conda environment:

```bash
python examples/henninger/01_build_model_no_constraints.py
python examples/henninger/02_build_model_with_constraints.py
python examples/henninger/03_inverse_kinematics.py
```

Deeper analyses (comparing joint constraints, marker sets, calibrating a scapulothoracic
ellipsoid) live in [studies/](studies/) — see [studies/README.md](studies/README.md).

Some scripts open visualisation windows, so run them from a graphical session.

## Why source installation

Installing from source is useful when you want to keep this repository and the library in sync during development

## Project Structure

- [examples/](examples/) — the progressive, per-dataset tutorial (build model → add constraints → IK), with shared helpers in `examples/_shared/`
- [studies/](studies/) — short shoulder investigations built on top of the example model builders
- [environment.yml](environment.yml) defines the Conda environment used for development

## License

This project is distributed under the MIT License. See [LICENSE.md](LICENSE.md).
