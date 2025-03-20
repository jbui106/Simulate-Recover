import unittest
import numpy as np
from src.EZDiffusionModel import EZDiffusionModel
from src.SimulateAndRecover import SimulateAndRecover


class TestSimulateAndRecover(unittest.TestCase):

    def test_initialization(self):
        # Test initialization with default parameters
        simulator = SimulateAndRecover()
        
        # Check that random params are generated correctly
        self.assertTrue(0.5 <= simulator.true_v <= 2.0)
        self.assertTrue(0.5 <= simulator.true_a <= 2.0)
        self.assertTrue(0.1 <= simulator.true_tau <= 0.5)

        # Test initialization with specific parameters
        true_params = (1.0, 1.5, 0.3)
        simulator = SimulateAndRecover(true_params=true_params)
        
        self.assertEqual(simulator.true_v, 1.0)
        self.assertEqual(simulator.true_a, 1.5)
        self.assertEqual(simulator.true_tau, 0.3)

    def test_bias_and_squared_error(self):
        # Test the calculation of bias and squared error
        simulator = SimulateAndRecover(true_params=(1.0, 1.5, 0.3))
        
        # Simulating data
        N = 40
        simulator.simulate_and_recover()

        # Check if the biases and squared errors are calculated correctly
        biases = simulator.biases[N]
        squared_errors = simulator.squared_errors[N]

        # Assert that the lists are not empty
        self.assertGreater(len(biases['v']), 0)
        self.assertGreater(len(biases['a']), 0)
        self.assertGreater(len(biases['tau']), 0)

        self.assertGreater(len(squared_errors['v']), 0)
        self.assertGreater(len(squared_errors['a']), 0)
        self.assertGreater(len(squared_errors['tau']), 0)

    def test_prediction(self):
        # Test that the EZDiffusionModel's predicted statistics are correct
        model = EZDiffusionModel(1.0, 1.5, 0.3)
        R_pred, M_pred, V_pred = model.forward_equations()

        # Test that predictions are returned as expected (accuracy, mean RT, variance RT)
        self.assertIsInstance(R_pred, float)
        self.assertIsInstance(M_pred, float)
        self.assertIsInstance(V_pred, float)

        # Ensure that the predicted accuracy is between 0 and 1
        self.assertGreaterEqual(R_pred, 0)
        self.assertLessEqual(R_pred, 1)

    def test_parameter_estimation(self):
        # Test the parameter recovery using the inverse equations
        model = EZDiffusionModel(1.0, 1.5, 0.3)
        
        # Simulate some noisy data
        N = 40
        R_pred, M_pred, V_pred = model.forward_equations()
        R_obs, M_obs, V_obs = model.simulate_noisy_data(N, R_pred, M_pred, V_pred)

        # Estimate parameters from the observed data
        v_est, a_est, tau_est = model.inverse_equations(R_obs, M_obs, V_obs)

        # Ensure that the estimated parameters are close to the true values
        self.assertAlmostEqual(v_est, 1.0, delta=0.2)
        self.assertAlmostEqual(a_est, 1.5, delta=0.2)
        self.assertAlmostEqual(tau_est, 0.3, delta=0.1)

    def test_integration(self):
        # Test the full simulation and recovery integration
        simulator = SimulateAndRecover(true_params=(1.0, 1.5, 0.3))
        
        # Run the full simulation and recovery process
        results = simulator.simulate_and_recover()

        # Ensure the results contain average biases and squared errors
        self.assertIn('avg_bias', results)
        self.assertIn('avg_squared_error', results)

        # Check that the averages for each parameter (v, a, tau) are available
        self.assertIn('v', results['avg_bias'])
        self.assertIn('a', results['avg_bias'])
        self.assertIn('tau', results['avg_bias'])

        self.assertIn('v', results['avg_squared_error'])
        self.assertIn('a', results['avg_squared_error'])
        self.assertIn('tau', results['avg_squared_error'])

    def test_corruption(self):
        # Test if the model handles corrupted data (e.g., NaN, extreme values)
        model = EZDiffusionModel(1.0, 1.5, 0.3)

        # Create corrupted (NaN) data
        R_obs, M_obs, V_obs = np.nan, np.nan, np.nan

        # Try to recover parameters and ensure it raises an error or handles NaN values
        with self.assertRaises(ValueError):
            model.inverse_equations(R_obs, M_obs, V_obs)

        # Now test with extreme values
        R_obs, M_obs, V_obs = np.inf, -np.inf, np.inf
        with self.assertRaises(ValueError):
            model.inverse_equations(R_obs, M_obs, V_obs)

if __name__ == '__main__':
    unittest.main()
