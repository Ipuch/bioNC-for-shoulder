"""The ellipsoid geometry the calibration reports its headline numbers from.

``ellipsoid_surface_distance_mm`` is the metric the leave-one-out study quotes, and
``fit_ellipsoid`` produces the warm start. Both are pure functions of a point cloud, so they can be
checked against a surface whose answer is known exactly.
"""

import numpy as np
import pytest

from examples._shared.frames import rodrigues_matrix
from studies.shoulder_calibration import ellipsoid_surface_distance_mm, fit_ellipsoid


def ellipsoid_patch(semi_axes, center, nb_points=400, spread=0.6, seed=0):
    """Points exactly on the ellipsoid, over a patch rather than the whole surface."""
    generator = np.random.default_rng(seed)
    theta = generator.uniform(np.pi / 2 - spread, np.pi / 2 + spread, nb_points)
    phi = generator.uniform(-spread, spread, nb_points)
    unit = np.vstack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    return np.asarray(semi_axes).reshape(3, 1) * unit + np.asarray(center).reshape(3, 1)


class TestSurfaceDistance:
    def test_sphere_distance_is_signed_and_radial(self):
        """On a sphere the answer is exactly ``|p - c| - r``, positive outside."""
        radius, center = 0.1, np.array([0.3, -0.2, 0.05])
        directions = rodrigues_matrix([0.4, 0.2, -0.1])  # three orthonormal directions
        for factor, expected_mm in ((1.0, 0.0), (2.0, 100.0), (0.5, -50.0)):
            cloud = center.reshape(3, 1) + factor * radius * directions
            distances = ellipsoid_surface_distance_mm(cloud, [radius] * 3, center)
            np.testing.assert_allclose(distances, expected_mm, atol=1e-9)

    def test_points_on_the_surface_are_at_zero(self):
        semi_axes, center = (0.14, 0.11, 0.19), (0.02, -0.03, 0.01)
        cloud = ellipsoid_patch(semi_axes, center)
        np.testing.assert_allclose(ellipsoid_surface_distance_mm(cloud, semi_axes, center), 0.0, atol=1e-9)

    def test_rotation_is_applied_about_the_centre(self):
        """A rotated ellipsoid still contains the correspondingly rotated cloud."""
        semi_axes, center = np.array([0.2, 0.1, 0.15]), np.array([0.05, 0.0, -0.02])
        rotation = rodrigues_matrix([0.3, -0.5, 0.2])
        cloud = ellipsoid_patch(semi_axes, np.zeros(3))
        rotated = rotation @ cloud + center.reshape(3, 1)

        np.testing.assert_allclose(ellipsoid_surface_distance_mm(rotated, semi_axes, center, rotation), 0.0, atol=1e-9)
        # and ignoring the rotation gives a genuinely different answer, i.e. it is really used
        assert np.abs(ellipsoid_surface_distance_mm(rotated, semi_axes, center)).max() > 1.0


class TestFitEllipsoid:
    """``reference`` is the subject-sized box; the ridge pulls the semi-axes toward its ``scale``."""

    def test_recovers_a_noiseless_sphere_the_prior_agrees_with(self):
        """When the truth matches the reference scale, the ridge costs nothing and the fit is exact."""
        radius, center = 0.15, np.array([0.01, -0.02, 0.03])
        cloud = ellipsoid_patch([radius] * 3, center, spread=1.0)
        fit = fit_ellipsoid(cloud, dict(center=center, scale=radius))

        np.testing.assert_allclose(fit["semi_axes"], radius, atol=1e-4)
        np.testing.assert_allclose(fit["center"], center, atol=1e-4)
        assert fit["residual_mm"] == pytest.approx(0.0, abs=1e-3)
        assert fit["prior_pull_mm"] == pytest.approx(0.0, abs=1e-3)
        assert fit["at_bounds"] == []

    def test_residual_excludes_the_ridge(self):
        """
        The regression this pins: ``residual_mm`` is the surface RMS, not the RMS of the whole
        minimised vector. Force the ridge to be stretched by handing a reference scale far from the
        truth, and check the reported number still describes the surface alone.
        """
        semi_axes = np.array([0.20, 0.20, 0.20])
        center = np.zeros(3)
        cloud = ellipsoid_patch(semi_axes, center, spread=1.0)
        reference = dict(center=center, scale=0.30)  # deliberately wrong by 100 mm

        fit = fit_ellipsoid(cloud, reference, prior=0.05)

        recomputed = ellipsoid_surface_distance_mm(cloud, fit["semi_axes"], fit["center"], fit["rotation"])
        assert fit["residual_mm"] == pytest.approx(float(np.sqrt(np.mean(recomputed**2))), abs=1e-9)
        assert fit["prior_pull_mm"] > 10.0  # the ridge really is stretched, so the two differ

    def test_reports_semi_axes_riding_a_bound(self):
        """A semi-axis pinned by its box must be named, because its value then means nothing."""
        cloud = ellipsoid_patch([0.15, 0.15, 0.15], np.zeros(3))
        bounds = dict(semi_axes=(0.28, 0.40), center_half_range=0.15)  # truth is below the lower bound
        fit = fit_ellipsoid(cloud, dict(center=np.zeros(3), scale=0.15), bounds=bounds)

        assert set("abc") & set(fit["at_bounds"])

    def test_orientation_is_frozen_unless_asked_for(self):
        cloud = ellipsoid_patch([0.14, 0.11, 0.19], np.zeros(3))
        reference = dict(center=np.zeros(3), scale=0.15)

        np.testing.assert_allclose(fit_ellipsoid(cloud, reference)["rotation"], np.eye(3), atol=1e-8)
        # a frozen unknown must never be reported as riding its (collapsed) bound
        assert not {"rx", "ry", "rz"} & set(fit_ellipsoid(cloud, reference)["at_bounds"])
