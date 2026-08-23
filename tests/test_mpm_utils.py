import unittest

import numpy as np

from utils.mpm_utils import collider_state, column_fill_points, lame_parameters


class MPMUtilsTest(unittest.TestCase):
    def test_column_fill_points(self):
        surface = np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=float,
        )
        filled = column_fill_points(surface, pitch=1.0)
        actual = {tuple(point) for point in filled}
        self.assertEqual(actual, {(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 0, 0), (1, 1, 0)})

    def test_column_fill_to_global_bottom_builds_heightfield_solid(self):
        surface = np.array([[0, 0, 0], [0, 2, 0], [1, 1, 0]], dtype=float)
        filled = column_fill_points(surface, pitch=1.0, fill_to_global_bottom=True)
        actual = {tuple(point) for point in filled}
        self.assertIn((1, 0, 0), actual)
        self.assertIn((1, 1, 0), actual)

    def test_collider_trajectory_contains_press_slide_and_recovery(self):
        start = np.array([0.0, 1.0, 0.0])
        pressed = collider_state(0.4, start, 0.2, 0.3, 1.0)
        slid = collider_state(0.6, start, 0.2, 0.3, 1.0)
        final = collider_state(1.0, start, 0.2, 0.3, 1.0)
        np.testing.assert_allclose(pressed.center, [0.0, 0.8, 0.0])
        np.testing.assert_allclose(slid.center, [0.3, 0.8, 0.0])
        np.testing.assert_allclose(final.center, [0.3, 1.0, 0.0])
        np.testing.assert_allclose(final.velocity, 0.0)

    def test_lame_parameters(self):
        mu, lam = lame_parameters(1000.0, 0.2)
        self.assertGreater(mu, 0.0)
        self.assertGreater(lam, 0.0)


if __name__ == "__main__":
    unittest.main()
