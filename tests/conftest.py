"""Shared fixtures.

Most tests here are pure and run anywhere. The few that need a real model must skip rather than
fail when the c3d files are absent: ``examples/data/`` is gitignored, so a fresh clone has none.
"""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "examples" / "data"
CLINICAL_C3D = DATA_DIR / "testFlorent_clinicalData.c3d"


@pytest.fixture(scope="session")
def clinical_c3d() -> str:
    """Path to the single-trial clinical c3d, or a skip if this clone has no data."""
    if not CLINICAL_C3D.exists():
        pytest.skip(f"no c3d data at {CLINICAL_C3D} (examples/data/ is gitignored)")
    return str(CLINICAL_C3D)


@pytest.fixture(scope="session")
def clinical_model(clinical_c3d):
    """A built clinical model. Session-scoped: building it reads and fits the whole c3d."""
    from examples.clinical.model import build_model_constrained

    return build_model_constrained(clinical_c3d, marker_set="anatomical")
