import unittest
import numpy as np
from src.EZDiffusionModel import EZDiffusionModel
from src.SimulateAndRecover import SimulateAndRecover

class TestEZDiffusionModel(unittest.TestCase):
    
    def generate_random_parameters(self):
        """Generates random values within the specified ranges."""
        a = np.random.uniform(0.5, 2)  # Boundary separation
        v = np.random.uniform(0.5, 2)  # Drift rate
        t = np.random.uniform(0.1, 0.5)  # Nondecision time
        return a, v, t
    
    # Initialization Tests
    def test_initialization_valid(self):
        # Test with valid parameters
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        self.assertEqual(model.v, v)
        self.assertEqual(model.a, a)
        self.assertEqual(model.t, t)

    def test_initialization_invalid(self):
        # Test for invalid initialization (negative values)
        with self.assertRaises(ValueError):
            EZDiffusionModel(v=-1.0, a=1.0, t=0.2)
        
        with self.assertRaises(ValueError):
            EZDiffusionModel(v=1.5, a=0, t=0.2)

    def test_access_before_fit(self):
        # Ensure that parameter estimates cannot be accessed before fitting
        model = EZDiffusionModel(v=1.5, a=1.0, t=0.2)
        with self.assertRaises(ValueError):
            model.nu_est

        with self.assertRaises(ValueError):
            model.alpha_est
        
        with self.assertRaises(ValueError):
            model.tau_est

    # Prediction Tests
    def test_prediction_no_noise(self):
        # Test that the bias is close to zero when no noise is introduced
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        model.fit(N=10)
        bias = model.bias()
        self.assertTrue(np.allclose(bias, np.array([0, 0, 0]), atol=0.01))

    def test_squared_error_decreases_with_N(self):
        # Test that the squared error decreases as sample size N increases
        model = EZDiffusionModel(v=1.5, a=1.0, t=0.2)
        errors = []
        
        for N in [10, 40, 4000]:
            model.fit(N=N)
            errors.append(model.squared_error())
        
        self.assertTrue(errors[0] > errors[1] > errors[2])

    def test_prediction_with_known_values(self):
        # Ensure that prediction matches the expected result with known parameters
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        model.fit(N=10)
        bias = model.bias()
        
        # Check that bias should be close to zero for a correct fit
        self.assertTrue(np.allclose(bias, np.array([0, 0, 0]), atol=0.05))

    # Parameter Estimation Tests
    def test_parameter_estimation_before_fit(self):
        # Test that user cannot request parameter estimates before fitting
        model = EZDiffusionModel(v=1.5, a=1.0, t=0.2)
        
        with self.assertRaises(ValueError):
            model.bias()
        
        with self.assertRaises(ValueError):
            model.squared_error()
        
    def test_parameter_estimation_with_fit(self):
        # Test that after fitting, parameters are estimated
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        model.fit(N=10)
        self.assertIsNotNone(model.nu_est)
        self.assertIsNotNone(model.alpha_est)
        self.assertIsNotNone(model.tau_est)

    # Integration Tests
    def test_stability_of_fitting_process(self):
        # Ensure that fitting multiple times leads to stable parameter estimates
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        
        previous_values = None
        for _ in range(10):
            model.fit(N=40)
            current_values = (model.nu_est, model.alpha_est, model.tau_est)
            if previous_values:
                self.assertTrue(np.allclose(previous_values, current_values, atol=0.01))
            previous_values = current_values

    def test_prediction_alignment(self):
        # Verify that predictions for each condition align with the observed response patterns for each sample size
        for _ in range(1000):
            a, v, t = self.generate_random_parameters()
            model = EZDiffusionModel(v=v, a=a, t=t)
            conditions = [10, 40, 4000]
            for N in conditions:
                model.fit(N=N)
                bias = model.bias()
                squared_error = model.squared_error()
                
                # Ensure that bias approaches zero and squared error decreases with N
                self.assertTrue(np.allclose(bias, np.array([0, 0, 0]), atol=0.05))
                self.assertLess(squared_error, 0.1)

    # Corruption Tests
    def test_private_attribute_access(self):
        # Test that private attributes cannot be accessed directly
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        
        with self.assertRaises(AttributeError):
            model._nu_est
        
        with self.assertRaises(AttributeError):
            model._alpha_est
        
        with self.assertRaises(AttributeError):
            model._tau_est

    def test_inconsistent_object_state(self):
        # Test that the model cannot be in an inconsistent state due to improper modification
        a, v, t = self.generate_random_parameters()
        model = EZDiffusionModel(v=v, a=a, t=t)
        
        # Simulate a scenario where user tries to manually alter internal state
        model._nu_est = None  # Directly manipulating private attribute
        
        with self.assertRaises(ValueError):
            model.bias()  # Should raise error because the model state is inconsistent

if __name__ == '__main__':
    unittest.main()
