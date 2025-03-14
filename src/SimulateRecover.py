from models.ez_diffusion import EZDiffusionModel

def simulate_and_recover(N, num_iterations=1000):
    true_params = []
    estimated_params = []

    for _ in range(num_iterations):
        # Randomly select model parameters
        v = np.random.uniform(0.5, 2)
        a = np.random.uniform(0.5, 2)
        tau = np.random.uniform(0.1, 0.5)

        # Initialize the model
        model = EZDiffusionModel(a, v, tau)

        # Get forward predictions
        R_pred, M_pred, V_pred = model.forward_equations()

        # Simulate observations
        R_obs = np.random.binomial(N, R_pred) / N  # Accuracy
        M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))  # Mean RT
        V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))  # Variance of RT

        # Estimate parameters from observed data
        v_est, a_est, tau_est = model.inverse_equations(R_obs, M_obs, V_obs)

        # Store true and estimated parameters
        true_params.append((v, a, tau))
        estimated_params.append((v_est, a_est, tau_est))

    return np.array(true_params), np.array(estimated_params)

def evaluate_performance(N_values=[10, 40, 4000], num_iterations=1000):
    for N in N_values:
        true_params, estimated_params = simulate_and_recover(N, num_iterations)
        bias = np.mean(estimated_params - true_params, axis=0)
        mse = np.mean((estimated_params - true_params) ** 2, axis=0)
        print(f"N={N} Bias: {bias}, MSE: {mse}")
