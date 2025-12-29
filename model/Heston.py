import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import pickle


class HestonModel:
    def __init__(self, S0, params):

        self.S0 = S0
        self.rd = params["usd_ir"]
        self.rf = params["eur_ir"]
        self.params = params
        self.set_parameters(params)

        self.random_seed = 42

    def set_parameters(self, params):
        """Set model parameters"""
        self.v0 = params["v0"]  # Initial variance
        self.theta = params["theta"]  # Long-term variance
        self.kappa = params["kappa"]  # Mean reversion
        self.sigma = params["sigma"]  # Vol of vol
        self.rho = params["rho"]  # Correlation
        self.mu = params["mu"]  # Drift
        self.params = params

        self.vol0 = np.sqrt(self.v0)
        self.long_term_vol = np.sqrt(self.theta)

    def simulate(self, T):
        """Euler Simulation: Returns time array, spot paths and volatility"""

        np.random.seed(self.random_seed)

        n_paths = 10000
        dt = 1 / 252
        n_steps = int(T / dt)

        t = np.linspace(0, T, n_steps + 1)
        S = np.zeros((n_steps + 1, n_paths))
        v = np.zeros((n_steps + 1, n_paths))

        S[0:] = self.S0
        v[0:] = self.v0

        # Generate Brownian Motion Correlation
        dW1 = np.random.normal(0, np.sqrt(dt), (n_steps, n_paths))
        dW2 = self.rho * dW1 + np.sqrt(1 - self.rho**2) * np.random.normal(
            0, np.sqrt(dt), (n_steps, n_paths)
        )

        # Euler Discretisation
        for index in range(1, n_steps + 1):
            #prevents negative variance
            sqrt_v = np.sqrt(np.maximum(v[index - 1, :], 1e-10))

            # Variance process
            v[index, :] = (
                v[index - 1, :]
                + self.kappa * (self.theta - v[index - 1, :]) * dt
                + self.sigma * sqrt_v * dW2[index - 1, :]
            )
            v[index, :] = np.maximum(v[index, :], 1e-10)

            # Spot process
            S[index, :] = S[index - 1, :] * np.exp(
                (self.mu - 0.5 * v[index - 1, :]) * dt + sqrt_v * dW1[index - 1, :]
            )

        return t, S, np.sqrt(v)

    def calculate_option_price(self, K, T, option_type="call", n_paths=10000):
        """Calculate option price via Monte Carlo"""
        _, S, _ = self.simulate(T)

        # Terminal spot prices
        S_T = S[-1, :]

        # Calculate payoff
        if option_type == "call":
            payoff = np.maximum(S_T - K, 0)
        else:
            payoff = np.maximum(K - S_T, 0)

        # Discount back
        price = np.exp(-self.rd * T) * payoff.mean()

        return price

    def calibrate(self, market_data, n_paths=10000):

        def objective_fn(obj_params):

            self.params["v0"] = obj_params[0]
            self.params["theta"] = obj_params[1]
            self.params["kappa"] = obj_params[2]
            self.params["sigma"] = obj_params[3]
            self.params["rho"] = obj_params[4]

            self.set_parameters(self.params)

            model_prices = {
                "price_atm_1y_mkt": self.calculate_option_price(
                    market_data["F_1y"], 1.0, "call", n_paths
                ),
                "price_call_1y_mkt": self.calculate_option_price(
                    market_data["K_call_1y"], 1.0, "call", n_paths
                ),
                "price_put_1y_mkt": self.calculate_option_price(
                    market_data["K_put_1y"], 1.0, "put", n_paths
                ),
                "price_atm_5y_mkt": self.calculate_option_price(
                    market_data["F_5y"], 5.0, "call", n_paths
                ),
                "price_call_5y_mkt": self.calculate_option_price(
                    market_data["K_call_5y"], 5.0, "call", n_paths
                ),
                "price_put_5y_mkt": self.calculate_option_price(
                    market_data["K_put_5y"], 5.0, "put", n_paths
                ),
            }

            errors = []
            for key in model_prices.keys():
                error = (model_prices[key] - market_data[key]) ** 2
                errors.append(error)

            total_error = np.sum(errors)

            return total_error

        bounds = [
            (1e-4, 0.25),  # v0
            (1e-4, 0.25),  # theta
            (0.1, 5.0),  # kappa
            (0.1, 1.0),  # sigma
            (-0.9, 0.0),  # rho
        ]

        obj_params = [self.v0, self.theta, self.kappa, self.sigma, self.rho]

        result = minimize(
            objective_fn,
            obj_params,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 50, "disp": True, "ftol": 1e-6},
        )

        
        if result.success:
            print("Calibrated Successfully")
            calibrated_params = result.x

            feller = 2 * calibrated_params[2] * calibrated_params[1]
            sigma_sq = calibrated_params[3] ** 2

            if feller < sigma_sq:
                print("Feller Condition Not Satisfied")
                return

            self.params["v0"] = calibrated_params[0]
            self.params["theta"] = calibrated_params[1]
            self.params["kappa"] = calibrated_params[2]
            self.params["sigma"] = calibrated_params[3]
            self.params["rho"] = calibrated_params[4]
            self.set_parameters(self.params)

            with open("model/calibrated_heston_params.pkl", "wb") as f:
                pickle.dump(self.params, f)

        else:
            print("Calibration Failed")

