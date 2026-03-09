"""Tests for D4 geometric symmetry — verify transforms and inverses are correct."""

import numpy as np
import pytest

from mezzanine.symmetries.depth_geometric import (
    apply_d4,
    inverse_d4,
    DepthGeometricSymmetry,
    DepthGeometricSymmetryConfig,
)


def _make_asymmetric(h=8, w=12):
    """Create an asymmetric image so we can detect any transform errors."""
    img = np.zeros((h, w), dtype=np.float32)
    img[0, 0] = 1.0  # top-left marker
    img[1, 2] = 2.0  # another unique marker
    return img


class TestD4Transforms:
    """Verify that apply_d4 followed by inverse_d4 is identity for all 8 elements."""

    @pytest.mark.parametrize("idx", range(8))
    def test_roundtrip_2d(self, idx):
        img = _make_asymmetric()
        transformed = apply_d4(img, idx)
        recovered = inverse_d4(transformed, idx)
        np.testing.assert_array_equal(recovered, img)

    @pytest.mark.parametrize("idx", range(8))
    def test_roundtrip_3d_hwc(self, idx):
        img = np.random.default_rng(42).random((8, 12, 3)).astype(np.float32)
        transformed = apply_d4(img, idx)
        recovered = inverse_d4(transformed, idx)
        np.testing.assert_allclose(recovered, img)

    def test_identity_is_noop(self):
        img = _make_asymmetric()
        np.testing.assert_array_equal(apply_d4(img, 0), img)

    def test_vflip(self):
        img = _make_asymmetric(4, 6)
        flipped = apply_d4(img, 4)
        # Top-left marker should now be at bottom-left
        assert flipped[-1, 0] == 1.0
        assert flipped[0, 0] == 0.0

    def test_rot180_is_double_rot90(self):
        img = _make_asymmetric()
        r180 = apply_d4(img, 2)
        r90_twice = apply_d4(apply_d4(img, 1), 1)
        np.testing.assert_array_equal(r180, r90_twice)


class TestDepthGeometricSymmetry:
    def test_batch_returns_all_elements(self):
        sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="d4"))
        img = _make_asymmetric()
        views = sym.batch(img)
        assert len(views) == 8

    def test_orbit_average_identity_subgroup(self):
        sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="identity"))
        img = _make_asymmetric()
        views = sym.batch(img)
        avg = sym.orbit_average(views)
        np.testing.assert_array_equal(avg, img)

    def test_orbit_average_recovers_symmetric(self):
        """For a perfectly symmetric input, orbit average should equal input."""
        sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="d4"))
        # Constant image is D4-invariant
        img = np.ones((8, 8), dtype=np.float32) * 5.0
        views = sym.batch(img)
        # Simulate: model predicts same constant for all views
        predictions = [np.ones((8, 8), dtype=np.float32) * 5.0 for _ in views]
        avg = sym.orbit_average(predictions)
        np.testing.assert_allclose(avg, img)

    def test_vflip_subgroup(self):
        sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="vflip"))
        assert sym.k == 2
        assert sym.elements == [0, 4]

    def test_orbit_average_cancels_vertical_bias(self):
        """Key test: if model adds a vertical gradient bias, orbit averaging should cancel it."""
        sym = DepthGeometricSymmetry(DepthGeometricSymmetryConfig(subgroup="vflip"))
        h, w = 8, 8

        # True depth is uniform
        true_depth = np.ones((h, w), dtype=np.float32) * 5.0

        # Model adds a vertical bias: bottom rows get +bias
        bias = np.linspace(0, 1, h).reshape(h, 1) * np.ones((1, w))

        img = np.zeros((h, w, 3), dtype=np.uint8)  # dummy image
        sym.batch(img)  # [original, vflipped]

        # Simulate biased model predictions for each view
        # Original view: bias goes top(0) -> bottom(1)
        pred_original = true_depth + bias.astype(np.float32)
        # Vflipped view: model still thinks bottom=closer, so bias is same direction
        # but in the flipped frame
        pred_vflipped = true_depth + bias.astype(np.float32)

        avg = sym.orbit_average([pred_original, pred_vflipped])
        # The vertical bias should be reduced (not perfectly cancelled unless
        # the bias is exactly linear, which it is here)
        vertical_var_original = np.var(pred_original.mean(axis=1))
        vertical_var_averaged = np.var(avg.mean(axis=1))
        assert vertical_var_averaged < vertical_var_original
