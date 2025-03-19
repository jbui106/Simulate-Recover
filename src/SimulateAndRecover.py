import numpy as np
import scipy.stats as stats
from src.EZDiffusionModel import EZDiffusionModel

class SimulateAndRecover:
    def __init__(self, true_params, N_values, num_trials=1000):
        """
        Initialize the simulator with true model parameters, sample sizes, and number of trials.
        :param true_params: Tuple of (v, a, tau), the true model parameters.
        :param N_values: List of sample sizes (e.g., [10, 40, 4000]).
        :param num_trials: The number of trials to run for each sample size (default is 1000).
        """
        self.true_v, self.true_a, self.true_tau = true_params
        self.N_values = N_values
        self.num_trials = num_trials
        self.biases = {'v': [], 'a': [], 'tau': []}
        self.squared_errors = {'v': [], 'a': [], 'tau': []}

    def simulate_and_recover(self):
        """
        Simulate the data, recover the parameters, and compute bias and squared error for each trial and N.
        :return: A tuple containing average bias and squared error for each parameter.
        """
        for N in self.N_values:
            for _ in range(self.num_trials):
                # Step 1: Create model and get predicted summary statistics
                model = EZDiffusionModel(self.true_a, self.true_v, self.true_tau)
                R_pred, M_pred, V_pred = model.forward_equations()

                # Step 2: Simulate noisy observed summary statistics
                R_obs, M_obs, V_obs = model.simulate_noisy_data(N, R_pred, M_pred, V_pred)

                # Step 3: Recover parameters using inverse equations
                v_est, a_est, tau_est = model.inverse_equations(R_obs, M_obs, V_obs)

                # Step 4: Compute biases and squared errors
                bias_v = self.true_v - v_est
                bias_a = self.true_a - a_est
                bias_tau = self.true_tau - tau_est

                squared_error_v = bias_v ** 2
                squared_error_a = bias_a ** 2
                squared_error_tau = bias_tau ** 2

                # Store the results
                self.biases['v'].append(bias_v)
                self.biases['a'].append(bias_a)
                self.biases['tau'].append(bias_tau)

                self.squared_errors['v'].append(squared_error_v)
                self.squared_errors['a'].append(squared_error_a)
                self.squared_errors['tau'].append(squared_error_tau)

        # Return average bias and squared error
        return self.compute_statistics()

    def compute_statistics(self):
        """
        Calculate the average bias and squared error for each parameter.
        :return: A dictionary with average biases and squared errors.
        """
        avg_bias_v = np.mean(self.biases['v'])
        avg_bias_a = np.mean(self.biases['a'])
        avg_bias_tau = np.mean(self.biases['tau'])

        avg_squared_error_v = np.mean(self.squared_errors['v'])
        avg_squared_error_a = np.mean(self.squared_errors['a'])
        avg_squared_error_tau = np.mean(self.squared_errors['tau'])

        return {
            'avg_bias': {'v': avg_bias_v, 'a': avg_bias_a, 'tau': avg_bias_tau},
            'avg_squared_error': {'v': avg_squared_error_v, 'a': avg_squared_error_a, 'tau': avg_squared_error_tau}
        }
