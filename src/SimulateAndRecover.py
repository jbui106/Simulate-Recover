import numpy as np
from EZDiffusionModel import EZDiffusionModel

class SimulateAndRecover:
    def __init__(self, true_params, N_values=[10, 40, 4000], num_trials=1000):
        self.true_v, self.true_a, self.true_tau = true_params
        self.N_values = N_values 
        self.num_trials = num_trials
        
        # Initialize dictionaries to store results by N value
        self.biases = {N: {'v': [], 'a': [], 'tau': []} for N in self.N_values}
        self.squared_errors = {N: {'v': [], 'a': [], 'tau': []} for N in self.N_values}

    def simulate_and_recover(self):
        """
        Simulate the data, recover the parameters, and compute bias and squared error for each trial and N.
        :return: A tuple containing average bias and squared error for each parameter and N value.
        """
        for N in self.N_values:
            print(f"Running simulations for N = {N}...")
            for trial in range(self.num_trials):
                # Create model and get predicted summary statistics
                model = EZDiffusionModel(self.true_a, self.true_v, self.true_tau)
                R_pred, M_pred, V_pred = model.forward_equations()

                # Simulate noisy observed summary statistics
                R_obs, M_obs, V_obs = model.simulate_noisy_data(N, R_pred, M_pred, V_pred)

                # Recover parameters using inverse equations
                v_est, a_est, tau_est = model.inverse_equations(R_obs, M_obs, V_obs)

                # Compute biases and squared errors
                bias_v = self.true_v - v_est
                bias_a = self.true_a - a_est
                bias_tau = self.true_tau - tau_est

                squared_error_v = bias_v ** 2
                squared_error_a = bias_a ** 2
                squared_error_tau = bias_tau ** 2

                # Store the results by N value
                self.biases[N]['v'].append(bias_v)
                self.biases[N]['a'].append(bias_a)
                self.biases[N]['tau'].append(bias_tau)

                self.squared_errors[N]['v'].append(squared_error_v)
                self.squared_errors[N]['a'].append(squared_error_a)
                self.squared_errors[N]['tau'].append(squared_error_tau)

        # Compute statistics and return results
        return self.compute_statistics()

    def compute_statistics(self):
        """
        Calculate the average bias and squared error for each parameter and N value.
        :return: A dictionary with average biases and squared errors by N.
        """
        results = {
            'avg_bias': {},
            'avg_squared_error': {}
        }
        
        print("\n=== SIMULATION RESULTS ===")
        print("\nParameter recovery performance by sample size:")
        
        for N in self.N_values:
            # Calculate averages for this N value
            avg_bias_v = np.mean(self.biases[N]['v'])
            avg_bias_a = np.mean(self.biases[N]['a'])
            avg_bias_tau = np.mean(self.biases[N]['tau'])

            avg_squared_error_v = np.mean(self.squared_errors[N]['v'])
            avg_squared_error_a = np.mean(self.squared_errors[N]['a'])
            avg_squared_error_tau = np.mean(self.squared_errors[N]['tau'])
            
            # Store in results
            results['avg_bias'][N] = {'v': avg_bias_v, 'a': avg_bias_a, 'tau': avg_bias_tau}
            results['avg_squared_error'][N] = {'v': avg_squared_error_v, 'a': avg_squared_error_a, 'tau': avg_squared_error_tau}
            
            # Print results for this N
            print(f"\nN = {N} (trials = {self.num_trials}):")
            print(f"  Average Bias:")
            print(f"    v: {avg_bias_v:.6f}, a: {avg_bias_a:.6f}, tau: {avg_bias_tau:.6f}")
            print(f"  Average Squared Error:")
            print(f"    v: {avg_squared_error_v:.6f}, a: {avg_squared_error_a:.6f}, tau: {avg_squared_error_tau:.6f}")
        
        return results