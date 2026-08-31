"""Frame selection and the small naming helpers the studies key everything off.

``farthest_point_sample`` is the claim the frame-selection figure is built to defend -- that
spreading the budget over the *workspace* beats spreading it over time -- and it has to be
deterministic, because the calibration is only reproducible if the frames it picks are.
"""

import numpy as np
import pytest

from examples._shared.c3d_data import farthest_point_sample
from examples._shared.ik import euler_axis_labels
from studies.kinematic_calibration import labels_at_bounds
from studies.shoulder_calibration import trial_kind, trial_label


def coverage_radius(features, index):
    """Worst-case distance from any row to its nearest selected row -- what the sampling minimises."""
    distances = np.linalg.norm(features[:, None, :] - features[None, index, :], axis=2)
    return float(np.max(np.min(distances, axis=1)))


class TestFarthestPointSample:
    def test_is_deterministic(self):
        features = np.random.default_rng(3).normal(size=(200, 4))
        first = farthest_point_sample(features, 20)
        np.testing.assert_array_equal(first, farthest_point_sample(features, 20))

    def test_returns_sorted_unique_indices_in_range(self):
        features = np.random.default_rng(4).normal(size=(150, 3))
        picked = farthest_point_sample(features, 25)

        assert len(picked) == 25
        assert len(set(picked.tolist())) == 25
        np.testing.assert_array_equal(picked, np.sort(picked))
        assert picked.min() >= 0 and picked.max() < 150

    @pytest.mark.parametrize("nb_samples", [150, 400])
    def test_asking_for_everything_returns_everything(self, nb_samples):
        features = np.random.default_rng(5).normal(size=(150, 3))
        np.testing.assert_array_equal(farthest_point_sample(features, nb_samples), np.arange(150))

    def test_covers_the_space_better_than_a_uniform_stride(self):
        """
        The reason the frame selection exists. A trial dwells *unevenly*: it lingers in some
        postures and passes quickly through others. Uniform-in-time then spends most of the budget
        re-sampling the slow regions and misses the brief ones entirely, while farthest-point
        sampling spends it on distinct postures.

        (Were the dwells all the same length, a uniform stride would be optimal by construction --
        which is exactly the case this test must not accidentally set up.)
        """
        generator = np.random.default_rng(6)
        postures = generator.normal(size=(8, 3)) * 3
        dwells = [200, 100, 40, 20, 10, 10, 10, 10]  # 400 frames, wildly uneven
        features = np.repeat(postures, dwells, axis=0) + generator.normal(size=(400, 3)) * 0.05

        budget = 8
        farthest = coverage_radius(features, farthest_point_sample(features, budget))
        uniform = coverage_radius(features, np.linspace(0, len(features) - 1, budget).astype(int))

        assert farthest < uniform
        assert farthest < 0.5  # it found all eight postures; the noise floor is ~0.15

    def test_picks_the_extremes_of_an_elongated_cloud(self):
        features = np.linspace(-1, 1, 101).reshape(-1, 1)
        picked = farthest_point_sample(features, 2).tolist()
        assert picked == [0, 100]


class TestTrialNaming:
    @pytest.mark.parametrize(
        "path, label, kind",
        [
            ("/data/99007140-40.19107308-20260825-PROTOCOL01-FUNCTIONAL2-01.c3d", "FUNCTIONAL2", "FUNCTIONAL"),
            ("/data/99007140-40.19107308-20260825-PROTOCOL01-ANALYTIC4-01.c3d", "ANALYTIC4", "ANALYTIC"),
        ],
    )
    def test_label_and_kind(self, path, label, kind):
        assert trial_label(path) == label
        assert trial_kind(path) == kind


class TestEulerAxisLabels:
    def test_repeated_axes_are_numbered(self):
        assert euler_axis_labels("yxy") == ["Y1", "X", "Y2"]

    def test_distinct_axes_are_left_alone(self):
        assert euler_axis_labels("xyz") == ["X", "Y", "Z"]


class TestLabelsAtBounds:
    def test_names_only_the_unknown_riding_a_bound(self):
        found = labels_at_bounds([5.0, 1.0], lower=[0.0, 0.0], upper=[5.0, 10.0], labels=["riding", "interior"])
        assert found == ["riding"]

    def test_ignores_an_unknown_frozen_to_a_point(self):
        """A box collapsed to a point is trivially 'at bounds'; saying so would be noise."""
        assert labels_at_bounds([0.0], lower=[-1e-9], upper=[1e-9], labels=["frozen"]) == []

    def test_tolerance_is_relative_to_the_box_width(self):
        # 0.5 % into a box of width 100 -> within the default 1 % tolerance
        assert labels_at_bounds([0.5], lower=[0.0], upper=[100.0], labels=["near"]) == ["near"]
        assert labels_at_bounds([0.5], lower=[0.0], upper=[100.0], labels=["near"], tolerance=0.001) == []
