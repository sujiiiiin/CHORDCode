import unittest

import numpy as np

from utils.mpm_utils import lame_parameters
from utils.warp_mpm import MPMConfig, WarpMPMSolver


class WarpMPMSmokeTest(unittest.TestCase):
    def test_cpu_step_preserves_fixed_particles_and_finite_state(self):
        points = np.array(
            [[x, y, z] for x in (0.0, 0.1) for y in (0.0, 0.1) for z in (0.0, 0.1)],
            dtype=np.float32,
        )
        fixed = (points[:, 1] == 0.0).astype(np.int32)
        mu, lam = lame_parameters(100.0, 0.2)
        solver = WarpMPMSolver(
            points,
            fixed,
            MPMConfig(
                dx=0.1,
                dt=1.0e-4,
                density=1.0,
                particle_volume=0.001,
                mu=mu,
                lam=lam,
            ),
            device="cpu",
        )
        solver.step(np.array([10, 10, 10]), np.zeros(3), 0.1, 0.0, False)
        result = solver.positions()
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result[fixed == 1], points[fixed == 1])


if __name__ == "__main__":
    unittest.main()
