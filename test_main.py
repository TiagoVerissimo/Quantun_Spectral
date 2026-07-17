import unittest
from fractions import Fraction

import numpy as np
import scipy.sparse as sp

from main import (
    _analytic_weight_bounds,
    conditional_sweep,
    hybrid_profile,
    uniform_grid_sweep,
    weyl_endpoint_cert,
)


class DensePath:
    def __init__(self, diagonal):
        self.matrix = sp.diags(np.asarray(diagonal, dtype=float), format="csr")
        self.matvecs = 0

    def H(self, _s):
        return self.matrix


class AnalyticBoundTests(unittest.TestCase):
    def test_gap_bound_is_rounded_up_from_exact_binary64_weights(self):
        edges = [(0, 1, 0.1), (0, 2, 0.2), (1, 2, 0.3)]
        bounds = _analytic_weight_bounds(edges, Lambda_I=4)
        exact_target = sum(
            (Fraction.from_float(weight) for _, _, weight in edges),
            Fraction(4),
        )
        self.assertGreaterEqual(
            Fraction.from_float(bounds["K_gap_cert"]), exact_target
        )

    def test_endpoint_and_profile_use_direct_gap_slope(self):
        self.assertEqual(weyl_endpoint_cert(0.25, 2.0, 1.0, 4.0), 1.0)
        records = [(0.0, 0.0, 2.0, 2.0)]
        profile = hybrid_profile(records, 4.0, np.asarray([0.0, 0.25]))
        np.testing.assert_allclose(profile, [2.0, 1.0])


class ContinuationTests(unittest.TestCase):
    def test_zero_diameter_uses_one_interval(self):
        path = DensePath([0.0, 1.0, 2.0, 3.0])
        records, windows, intervals = conditional_sweep(path, 0.0)
        self.assertEqual(len(records), 1)
        self.assertEqual(windows, [])
        self.assertEqual(len(intervals), 1)
        self.assertEqual(intervals[0]["h_actual"], 1.0)
        self.assertTrue(intervals[0]["is_conditionally_resolved"])

    def test_floor_probes_cover_unresolved_path_without_microsteps(self):
        path = DensePath([0.0, 0.0, 2.0, 3.0])
        records, windows, intervals = conditional_sweep(
            path, 1.0, eta=0.9, h_floor=1e-6
        )
        self.assertLess(len(records), 40)
        self.assertEqual(windows, [(0.0, 1.0)])
        self.assertAlmostEqual(sum(item["h_actual"] for item in intervals), 1.0)
        self.assertTrue(all(not item["is_conditionally_resolved"] for item in intervals))
        self.assertTrue(all(item["B_k"] <= 0.0 for item in intervals))

    def test_invalid_parameters_are_rejected(self):
        path = DensePath([0.0, 1.0, 2.0, 3.0])
        with self.assertRaises(ValueError):
            conditional_sweep(path, -1.0)
        with self.assertRaises(ValueError):
            conditional_sweep(path, 1.0, eta=1.0)
        with self.assertRaises(ValueError):
            conditional_sweep(path, 1.0, h_floor=0.0)

    def test_uniform_grid_uses_direct_K_over_delta_count(self):
        path = DensePath([0.0, 1.0, 2.0, 3.0])
        count, _, _, _ = uniform_grid_sweep(
            path, K_gap=4.0, delta_target=0.25, s_grid=np.linspace(0.0, 1.0, 5)
        )
        self.assertEqual(count, 17)


if __name__ == "__main__":
    unittest.main()
