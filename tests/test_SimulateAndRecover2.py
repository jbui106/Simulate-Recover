import unittest
import numpy as np
from EZDiffusionModel import EZDiffusionModel
from SimulateAndRecover import SimulateAndRecover

class TestSimulateAndRecoverSecond(unittest.TestCase):

    def setUp(self):
        # Test parameters
        self.true_params = (1.0, 1.0, 0.3)  # (v, a, tau)
        
        # Test with small N values and fewer trials for faster tests
        self.N_values = [10, 40, 100]
        self.num_trials = 100
        
        # Create simulator
        self.simulator = SimulateAndRecover(self.true_params, self.N_values, self.num_trials)
    
    def test_initialization(self):
        #Test proper initialization of simulator parameters
        self.assertEqual(self.simulator.true_v, 1.0)
        self.assertEqual(self.simulator.true_a, 1.0)
        self.assertEqual(self.simulator.true_tau, 0.3)
        self.assertEqual(self.simulator.N_values, self.N_values)
        self.assertEqual(self.simulator.num_trials, self.num_trials)
    
    def test_initialization_with_defaults(self):
        #Test initialization with default values
        simulator = SimulateAndRecover(self.true_params)
        self.assertEqual(simulator.true_v, 1.0)
        self.assertEqual(simulator.true_a, 1.0)
        self.assertEqual(simulator.true_tau, 0.3)
        self.assertEqual(simulator.N_values, [10, 40, 4000])
        self.assertEqual(simulator.num_trials, 1000)
    
    def test_random_parameter_generation(self):
        # Test random parameter generation if the class has this method
        if hasattr(self.simulator, 'generate_random_params'):
            for _ in range(10):
                v, a, tau = self.simulator.generate_random_params()
                
                # Check ranges
                self.assertTrue(0.5 <= v <= 2.0)
                self.assertTrue(0.5 <= a <= 2.0)
                self.assertTrue(0.1 <= tau <= 0.5)
    
    def test_simulate_and_recover_runs(self):
        results = self.simulator.simulate_and_recover()
        
        # Check that results contain expected keys
        self.assertIn('avg_bias', results)
        self.assertIn('avg_squared_error', results)
        
        # Check that results contain data for each N value
        for N in self.N_values:
            self.assertIn(N, results['avg_bias'])
            self.assertIn(N, results['avg_squared_error'])
    
    def test_bias_close_to_zero(self):
        simulator = SimulateAndRecover(self.true_params, [1000], 100)
        results = simulator.simulate_and_recover()
        
        # Check that average bias is close to zero
        avg_bias_v = results['avg_bias'][1000]['v']
        avg_bias_a = results['avg_bias'][1000]['a']
        avg_bias_tau = results['avg_bias'][1000]['tau']
        
        # Allow for some statistical variation
        self.assertAlmostEqual(avg_bias_v, 0.0, places=1)
        self.assertAlmostEqual(avg_bias_a, 0.0, places=1)
        self.assertAlmostEqual(avg_bias_tau, 0.0, places=1)
    
    def test_squared_error_decreases_with_n(self):
        N_values = [10, 100, 1000]
        simulator = SimulateAndRecover(self.true_params, N_values, 50)
        results = simulator.simulate_and_recover()
        
        # Check that squared error decreases as N increases
        se_v_10 = results['avg_squared_error'][10]['v']
        se_v_100 = results['avg_squared_error'][100]['v']
        se_v_1000 = results['avg_squared_error'][1000]['v']
        
        se_a_10 = results['avg_squared_error'][10]['a']
        se_a_100 = results['avg_squared_error'][100]['a']
        se_a_1000 = results['avg_squared_error'][1000]['a']
        
        se_tau_10 = results['avg_squared_error'][10]['tau']
        se_tau_100 = results['avg_squared_error'][100]['tau']
        se_tau_1000 = results['avg_squared_error'][1000]['tau']
        
        # Check that errors decrease with increasing N
        self.assertGreater(se_v_10, se_v_100)
        self.assertGreater(se_v_100, se_v_1000)
        
        self.assertGreater(se_a_10, se_a_100)
        self.assertGreater(se_a_100, se_a_1000)
        
        self.assertGreater(se_tau_10, se_tau_100)
        self.assertGreater(se_tau_100, se_tau_1000)
    
    def test_inverse_proportional_relationship(self):
        """Test that squared error is roughly inversely proportional to N."""
        # Use a larger range of N values
        N_values = [10, 100, 1000]
        simulator = SimulateAndRecover(self.true_params, N_values, 50)
        results = simulator.simulate_and_recover()
        
        # Check that squared error * N is roughly constant
        se_v_10 = results['avg_squared_error'][10]['v'] * 10
        se_v_100 = results['avg_squared_error'][100]['v'] * 100
        se_v_1000 = results['avg_squared_error'][1000]['v'] * 1000
        
        # Allow for some statistical variation
        ratio1 = se_v_100 / se_v_10
        ratio2 = se_v_1000 / se_v_100
        
        # Check that the ratios are roughly 1
        self.assertTrue(0.5 <= ratio1 <= 2.0)
        self.assertTrue(0.5 <= ratio2 <= 2.0)
    
    def test_compute_statistics(self):
        """Test that compute_statistics method works correctly."""
        # Populate the biases and squared_errors with known values
        for N in self.N_values:
            self.simulator.biases[N]['v'] = [0.1, -0.1, 0.2, -0.2, 0.0]
            self.simulator.biases[N]['a'] = [0.05, -0.05, 0.1, -0.1, 0.0]
            self.simulator.biases[N]['tau'] = [0.02, -0.02, 0.03, -0.03, 0.0]
            
            self.simulator.squared_errors[N]['v'] = [0.01, 0.01, 0.04, 0.04, 0.0]
            self.simulator.squared_errors[N]['a'] = [0.0025, 0.0025, 0.01, 0.01, 0.0]
            self.simulator.squared_errors[N]['tau'] = [0.0004, 0.0004, 0.0009, 0.0009, 0.0]
        
        # Compute statistics
        results = self.simulator.compute_statistics()
        
        # Check that average bias is computed correctly
        for N in self.N_values:
            self.assertAlmostEqual(results['avg_bias'][N]['v'], 0.0, places=6)
            self.assertAlmostEqual(results['avg_bias'][N]['a'], 0.0, places=6)
            self.assertAlmostEqual(results['avg_bias'][N]['tau'], 0.0, places=6)
            
            self.assertAlmostEqual(results['avg_squared_error'][N]['v'], 0.02, places=6)
            self.assertAlmostEqual(results['avg_squared_error'][N]['a'], 0.005, places=6)
            self.assertAlmostEqual(results['avg_squared_error'][N]['tau'], 0.00052, places=6)
    
    def test_integration(self):
        """Test full integration of forward equations, simulation, and inverse equations."""
        # Create a model with known parameters
        model = EZDiffusionModel(boundary=1.0, drift_rate=1.0, non_decision_time=0.3)
        
        # Get predicted values
        R_pred, M_pred, V_pred = model.forward_equations()
        
        # Simulate observed data with very large N
        N = 10000
        R_obs, M_obs, V_obs = model.simulate_noisy_data(N, R_pred, M_pred, V_pred)
        
        # Recover parameters
        v_est, a_est, tau_est = model.inverse_equations(R_obs, M_obs, V_obs)
        
        # Check that recovered parameters are close to original
        self.assertAlmostEqual(v_est, 1.0, places=1)
        self.assertAlmostEqual(a_est, 1.0, places=1)
        self.assertAlmostEqual(tau_est, 0.3, places=1)
    
    def test_corruption_resistance(self):
        model = EZDiffusionModel(boundary=1.0, drift_rate=1.0, non_decision_time=0.3)
        
        R_pred, M_pred, V_pred = model.forward_equations()
        
        R_corrupt = 0.0  # Completely corrupt accuracy
        
        try:
            v_est, a_est, tau_est = model.inverse_equations(R_corrupt, M_pred, V_pred)
            # If no exception, check for NaN or unrealistic values
            self.assertTrue(np.isnan(v_est) or v_est < 0 or v_est > 10)
        except:
            # Exception is expected
            pass
        
        # Test with corrupt mean RT
        M_corrupt = 0.0  # Impossible mean RT
        
        # Expect errors or NaN values
        try:
            v_est, a_est, tau_est = model.inverse_equations(R_pred, M_corrupt, V_pred)
            # If no exception, check for NaN or unrealistic values
            self.assertTrue(np.isnan(tau_est) or tau_est < 0)
        except:
            # Exception is expected
            pass
    
    def test_extreme_values(self):
        # Create simulator with extreme but valid parameter values
        true_params = (2.0, 2.0, 0.5)  # Max values
        simulator = SimulateAndRecover(true_params, [1000], 10)
        results = simulator.simulate_and_recover()
        
        # Check that results exist
        self.assertIn('avg_bias', results)
        self.assertIn('avg_squared_error', results)
        
        # Create simulator with extreme but valid parameter values
        true_params = (0.5, 0.5, 0.1)  # Min values
        simulator = SimulateAndRecover(true_params, [1000], 10)
        results = simulator.simulate_and_recover()
        
        # Check that results exist
        self.assertIn('avg_bias', results)
        self.assertIn('avg_squared_error', results)

if __name__ == '__main__':
    unittest.main()
