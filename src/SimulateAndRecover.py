from EZDiffusionModel import EZDiffusionModel

# Function to simulate and recover parameters over multiple iterations
class SimulateAndRecover:
    def __init__(self, num_iterations=1000):
        self.num_iterations = num_iterations

    def simulate_and_recover(N, num_iterations):
        true_params = []
        estimated_params = []
        biases = []
        squared_errors = []
    
    for _ in range(num_iterations):
        # Randomly select true parameters (ν, α, τ)
        v = np.random.uniform(0.5, 2)  # Drift rate (ν)
        a = np.random.uniform(0.5, 2)  # Boundary separation (α)
        tau = np.random.uniform(0.1, 0.5)  # Nondecision time (τ)
        
        # Generate predicted summary statistics (R_pred, M_pred, V_pred)
        R_pred, M_pred, V_pred = forward_equations(v, a, tau)
        
        # Simulate noisy observed data (R_obs, M_obs, V_obs)
        R_obs, M_obs, V_obs = simulate_noisy_data(N, R_pred, M_pred, V_pred)
        
        # Compute estimated parameters (ν_est, α_est, τ_est) using inverse equations
        v_est, a_est, tau_est = inverse_equations(R_obs, M_obs, V_obs)
        
        # Compute bias for each parameter
        bias_v = v - v_est
        bias_a = a - a_est
        bias_tau = tau - tau_est
        bias = np.array([bias_v, bias_a, bias_tau])
        
        # Compute squared error (sum of squared biases)
        squared_error = np.sum(bias ** 2)
        
        # Store true and estimated parameters for analysis
        true_params.append((v, a, tau))
        estimated_params.append((v_est, a_est, tau_est))
        biases.append(bias)
        squared_errors.append(squared_error)
    
    return np.array(true_params), np.array(estimated_params), np.array(biases), np.array(squared_errors)

# Function to evaluate performance across different sample sizes
    def evaluate_performance(N_values=[10, 40, 4000], num_iterations=1000):
        for N in N_values:
            true_params, estimated_params, biases, squared_errors = simulate_and_recover(N, num_iterations)
        
        # Compute average bias and MSE
            average_bias = np.mean(biases, axis=0)
            average_squared_error = np.mean(squared_errors)
        
            print(f"N={N} Average Bias: {average_bias}, Average Squared Error (MSE): {average_squared_error}")

# Running the performance evaluation with different sample sizes
    if __name__ == "__main__":
        evaluate_performance()
