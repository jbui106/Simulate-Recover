import unittest
import numpy as np
from src.SimulateAndRecover import SimulateAndRecover
from src.EZDiffusionModel import EZDiffusionModel

class TestSimulateAndRecover(unittest.TestCase):

    # 1. Setup and Initialization Test
    def test_initialization_with_random_params(self):
        # Create an instance of SimulateAndRecover without providing any parameters
        sar = SimulateAndRecover()

        # Check that the true parameters are set (should be random values)
        self.assertTrue(0.5 <= sar.true_v <= 2.0)
        self.assertTrue(0.5 <= sar.true_a <= 2.0)
        self.assertTrue(0.1 <= sar.true_tau <= 0.5)

    def test_initialization_with_given_params(self):
        # Provide specific parameters
        true_params = (1.2, 1.5, 0.3)
        sar = SimulateAndRecover(true_params=true_params)

        # Check that the true parameters match the provided values
        self.assertEqual(sar.true_v, 1.2)
        self.assertEqual(sar.true_a, 1.5)
        self.assertEqual(sar.true_tau, 0.3)

    # 2. Prediction Test
    def test_simulate_and_recover(self):
        # Using arbitrary parameters for the test
        true_params = (1.2, 1.5, 0.3)
        sar = SimulateAndRecover(true_params=true_params, N_values=[10])

        # Run the simulation and recovery
        results = sar.simulate_and_recover()

        # Ensure that results are returned as expected
        self.assertIn('avg_bias', results)
        self.assertIn('avg_squared_error', results)

        # Check if results for N=10 are available
        self.assertIn(10, results['avg_bias'])
        self.assertIn(10, results['avg_squared_error'])

        # Check the results contain valid numbers
        self.assertIsInstance(results['avg_bias'][10], dict)
        self.assertIsInstance(results['avg_squared_error'][10], dict)

    # 3. Parameter Estimation Test
    def test_parameter_estimation(self):
        true_params = (1.2, 1.5, 0.3)
        sar = SimulateAndRecover(true_params=true_params, N_values=[10])

        # Run the simulation
        results = sar.simulate_and_recover()

        # Extract the bias and squared errors
        avg_bias = results['avg_bias'][10]
        avg_squared_error = results['avg_squared_error'][10]

        # Test if bias and squared errors are computed (they should be numeric)
        self.assertIsInstance(avg_bias['v'], float)
        self.assertIsInstance(avg_bias['a'], float)
        self.assertIsInstance(avg_bias['tau'], float)

        self.assertIsInstance(avg_squared_error['v'], float)
        self.assertIsInstance(avg_squared_error['a'], float)
        self.assertIsInstance(avg_squared_error['tau'], float)

    # 4. Integration Test
    def test_integration(self):
        true_params = (1.2, 1.5, 0.3)
        sar = SimulateAndRecover(true_params=true_params, N_values=[10])

        # Check if the EZDiffusionModel is correctly instantiated
        model = EZDiffusionModel(sar.true_a, sar.true_v, sar.true_tau)
        self.assertIsInstance(model, EZDiffusionModel)

        # Check if the model can produce the forward equations without error
        R_pred, M_pred, V_pred = model.forward_equations()
        self.assertIsInstance(R_pred, np.ndarray)
        self.assertIsInstance(M_pred, np.ndarray)
        self.assertIsInstance(V_pred, np.ndarray)

        # Check if the model can simulate noisy data
        R_obs, M_obs, V_obs = model.simulate_noisy_data(10, R_pred, M_pred, V_pred)
        self.assertIsInstance(R_obs, np.ndarray)
        self.assertIsInstance(M_obs, np.ndarray)
        self.assertIsInstance(V_obs, np.ndarray)

        # Run the parameter estimation with the model
        v_est, a_est, tau_est = model.inverse_equations(R_obs, M_obs, V_obs)
        self.assertIsInstance(v_est, float)
        self.assertIsInstance(a_est, float)
        self.assertIsInstance(tau_est, float)

    # 5. Corruption Test
    def test_invalid_parameters(self):
        # Test if the system handles invalid parameters
        with self.assertRaises(ValueError):
            SimulateAndRecover(true_params=(-1.0, -1.5, -0.3))  # Negative params

if __name__ == "__main__":
    unittest.main()
