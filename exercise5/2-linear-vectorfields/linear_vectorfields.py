"""
Module implementing the LinearVectorfieldApproximation Class, which allows to easily approximate Linear Vector Fields with different methods.

author: Simon Blöchinger
"""

import numpy as np
from sklearn.metrics import mean_squared_error as skl_mse
from scipy.integrate import solve_ivp as scipy_solve_ivp


class LinearVectorfieldApproximation:
    def __init__(self, x0, x1, delta_t_setup) -> None:
        """Initializes the Class and computes v_hat."""
        self.x0 = x0
        self.x1 = x1
        self.delta_t_setup = delta_t_setup
        self.v_hat = self.estimate_v_hat()

        self.A_hat = None
        self.A_hat_MSE = None
        self.x_hat = None
        self.x_hat_MSE = None

    def estimate_v_hat(self):
        """Computes v_hat for x0, x1 and delta_t."""
        return (self.x1 - self.x0) / self.delta_t_setup
    
    def approximate_A_hat(self, use_numpy=True):
        """Approximates A_hat either using the numpy lstsq solver or the closed form solution."""
        if use_numpy:
            self.A_hat = np.linalg.lstsq(self.x0, self.v_hat, rcond=None)[0]  # rcond does not seem to impact the result
        else:
            self.A_hat = (np.linalg.inv(self.x0.T @ self.x0)) @ self.x0.T @ self.v_hat
    
    def compute_MSE_A_hat(self):
        """Computes the MSE for our A_hat calculation."""
        self.A_hat_MSE = skl_mse(self.v_hat, self.x0 @ self.A_hat.T)

    def compute_MSE_x_hat(self):
        """Computes the MSE for our x_hat calculation."""
        self.x_hat_MSE = skl_mse(self.x1, self.x_hat)
    
    def get_new_point_in_time(self, delta_t, use_A_hat=False):
        """Returns a new point in time for a start point x0, a vector v and a delta_t."""
        if use_A_hat:
            return self.x0 + (self.x0 @ self.A_hat.T) * delta_t
        else:
            return self.x0 + self.v_hat * delta_t
    
    def solve_ivp_range(self, fun, t0=0, t_end=0.1, t_eval=None, args=None, y0=None):
        """Solves an ODE using skikit ivp solver for a range of start values."""
        
        if y0 is None:
            y0s = self.x0
            N = y0s.shape[0]
        else:
            if len(y0) == 2:
                N = 1
            else:
                y0s = y0
                N = y0s.shape[0]
        
        
        if t_eval is None:
            self.x_hat = np.zeros((N, 2))
        else:
            self.x_hat = np.zeros((N, 2, t_eval.shape[0]))
        
        for k in range(N):
            if N > 1:
                y0 = y0s[k]
            
            ivp_return = scipy_solve_ivp(fun, t_span=[t0, t_end], y0=y0, t_eval=t_eval, args=args)
            if ivp_return.success:
                if t_eval is None:
                    self.x_hat[k] = ivp_return.y[:,-1]
                else:
                    self.x_hat[k] = ivp_return.y
            else:
                raise ValueError("The IVP was not successful!", ivp_return)
