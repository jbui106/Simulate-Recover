import unittest
import numpy as np
from models.ez_diffusion import EZDiffusionModel
from utils.data_simulation import simulate_and_recover

class TestEZDiffusionModel(unittest.TestCase):

    def test_bias_and_mse(self):
        """Test if bias is close to 0 and MSE decreases with N"""
        N_values = [10, 40, 4000]
        for N in N_values:
            true_params, estimated_params = simulate_and_recover(N, num_iterations=100)
            bias = np.mean(estimated_params - true_params, axis=0)
            mse = np.mean((estimated_params - true_params) ** 2, axis=0)

            # Check that bias approaches 0
            for b in bias:
                self.assertAlmostEqual(b, 0, delta=0.05)

            # Check that MSE decreases as N increases
            if N > 10:
                self.assertLess(np.mean(mse), 0.1)

if __name__ == "__main__":
    unittest.main()
