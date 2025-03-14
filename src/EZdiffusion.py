import numpy as np
import scipy.stats as stats

class EZDiffusionModel:
    def __init__(self, boundary, drift_rate, non_decision_time):
        """
        Initialize the EZ Diffusion Model with given parameters.
        :param boundary: Boundary separation (alpha)
        :param drift_rate: Drift rate (v)
        :param non_decision_time: Non-decision time (tau)
        """
        self.boundary = boundary
        self.drift_rate = drift_rate
        self.non_decision_time = non_decision_time

    def forward_equations(self):
        """
        Compute the 'predicted' summary statistics (accuracy, mean, variance) from model parameters.
        :return: R_pred (accuracy), M_pred (mean RT), V_pred (variance of RT)
        """
        v = self.drift_rate
        a = self.boundary
        tau = self.non_decision_time

        # Forward equations
        y = np.exp(-a * v)
        R_pred = 1 / (y + 1)  # Accuracy
        M_pred = tau + (a / (2 * v)) * ((1 - y) / (1 + y))  # Mean RT
        V_pred = (a / (2 * v**3)) * ((1 - 2 * a * v * y - y**2) / (y + 1)**2)  # Variance of RT
        return R_pred, M_pred, V_pred

    def simulate_noisy_data(N, R_pred, M_pred, V_pred):
        """
        Compute the 'observed' summary statistics (accuracy, mean, variance) from model parameters.
        :return: R_obs (accuracy), M_obs (mean RT), V_obs (variance of RT)
        """
        # Simulate observed accuracy rate (R_obs) using Binomial distribution
        T_obs = np.random.binomial(N, R_pred)
        R_obs = T_obs / N  # Observed accuracy rate
    
        # Simulate observed mean RT (M_obs) using Normal distribution
        M_obs = np.random.normal(M_pred, np.sqrt(V_pred / N))
    
        # Simulate observed variance RT (V_obs) using Gamma distribution
        V_obs = stats.gamma.rvs((N - 1) / 2, scale=(2 * V_pred / (N - 1)))
    
        return R_obs, M_obs, V_obs


    def inverse_equations(self, R_obs, M_obs, V_obs):
        """
        Recover the parameters (v, a, tau) from the 'observed' summary statistics.
        :return: Estimated parameters (v, a, tau)
        """
        # Inverse equations
        L = np.log(R_obs / (1 - R_obs))
        v_est = np.sign(R_obs - 0.5) * ((R_obs**2 * L - R_obs * L + R_obs - 0.5) / V_obs) ** 0.25
        a_est = L / v_est
        tau_est = M_obs - (a_est / (2 * v_est)) * ((1 - np.exp(-v_est * a_est)) / (1 + np.exp(-v_est * a_est)))
        return v_est, a_est, tau_est
