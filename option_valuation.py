import numpy as np
import plotly.graph_objects as go
from scipy.integrate import quad as QUAD

from halton_sequences import (apply_box_muller_transform,
                              generate_halton_vectors)


class MonteCarloOption:
    def __init__(self, params):
        """
        Monte Carlo simulator for a path-independent option.

        Parameters (all stored in a dict):
        - initial_stock_price
        - interest_rate
        - expiration_time
        - num_paths
        - payoff_function
        - mean_function
        - deviation_function
        - random_method: 'normal', 'antithetic', 'moment_matching', 'halton'
        - halton_primes: tuple of two primes for Halton sequences
        """
        self.params = params
        self.rng = np.random.default_rng(seed=0)

    def generate_random_variables(self):
        N = self.params["num_paths"]
        method = self.params.get("random_method", "normal")

        if method == "normal":
            return self.rng.normal(0, 1, size=N)
        elif method == "antithetic":
            half = self.rng.normal(0, 1, size=N // 2)
            return np.concatenate([half, -half])
        elif method == "moment_matching":
            half = self.rng.normal(0, 1, size=N // 2)
            normed = half / np.sqrt(np.var(half))
            return np.concatenate([normed, -normed])
        elif method == "halton":
            base_a, base_b = self.params.get("halton_primes", (2, 3))
            seq = generate_halton_vectors(base_a, base_b, N)
            return apply_box_muller_transform(seq)
        else:
            raise ValueError(f"Unknown random method: {method}")

    def simulate_terminal_prices(self):
        S0 = self.params["initial_stock_price"]
        T = self.params["expiration_time"]
        f = self.params["mean_function"]
        v = self.params["deviation_function"]

        Z = self.generate_random_variables()
        return f(S0, T) + v(S0, T) * np.sqrt(T) * Z

    def compute_option_value(self):
        S_T = self.simulate_terminal_prices()
        payoff = self.params["payoff_function"](S_T)
        discounted_value = (
            np.exp(-self.params["interest_rate"] * self.params["expiration_time"])
            * payoff
        )
        return np.mean(discounted_value), np.std(discounted_value) / np.sqrt(
            self.params["num_paths"]
        )

    def plot_distribution(self, num_bins=50):
        """
        Plot the histogram of Monte Carlo simulated option values using Plotly.
        """
        S_T = self.simulate_terminal_prices()
        payoff = self.params["payoff_function"](S_T)
        discounted_value = (
            np.exp(-self.params["interest_rate"] * self.params["expiration_time"])
            * payoff
        )

        hist = np.histogram(discounted_value, bins=num_bins, density=True)
        x = 0.5 * (hist[1][1:] + hist[1][:-1])
        y = hist[0]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=x, y=y, marker_color="skyblue", name="Monte Carlo Distribution")
        )

        # Add mean line
        mean_val = np.mean(discounted_value)
        fig.add_trace(
            go.Scatter(
                x=[mean_val, mean_val],
                y=[0, max(y) * 1.1],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name=f"Mean = {mean_val:.2f}",
            )
        )

        fig.update_layout(
            title="Monte Carlo Simulated Option Values",
            xaxis_title="Option Value",
            yaxis_title="Density",
            template="plotly_white",
            bargap=0.05,
        )
        fig.show()


# Analytical integration for reference
def calculate_analytical_value(
    initial_stock_price,
    expiration_time,
    mean_function,
    deviation_function,
    payoff_function,
    interest_rate,
):
    S0 = initial_stock_price
    T = expiration_time
    f = mean_function
    v = deviation_function
    g = payoff_function
    r = interest_rate

    integrand = lambda S: g(S) * np.exp(
        -0.5 * ((S - f(S0, T)) / (v(S0, T) * np.sqrt(T))) ** 2
    )
    I1 = QUAD(integrand, -1e7, 70000)
    I2 = QUAD(integrand, 70000, 80000)
    I3 = QUAD(integrand, 80000, 1e7)

    value = (
        np.exp(-r * T) * (I1[0] + I2[0] + I3[0]) / (v(S0, T) * np.sqrt(2 * np.pi * T))
    )
    error = I1[1] + I2[1] + I3[1]
    return value, error


def main():
    # Example parameters
    INITIAL_STOCK_PRICE = 70154
    INTEREST_RATE = 0.01
    T = 2.0
    NUM_PATHS = 1000000
    THETA = 70100
    ALPHA = 0.01
    BETA = 0.01
    GAMMA = 1.05
    SIGMA = 0.26
    X1, X2 = 70000, 80000

    def payoff_function(S):
        return np.where(S < X1, X2 - S, np.where(S < X2, S - X2, X1 - S))

    f = lambda S0, T: S0 * (np.cosh(2 * BETA * T - ALPHA * T) - 1) + THETA * (
        3 - np.exp(ALPHA * T) - np.exp(BETA * T)
    )
    v = lambda S0, T: SIGMA * (1 + ALPHA * T) * 0.5 * (S0 + THETA) ** GAMMA

    params = {
        "initial_stock_price": INITIAL_STOCK_PRICE,
        "interest_rate": INTEREST_RATE,
        "expiration_time": T,
        "num_paths": NUM_PATHS,
        "payoff_function": payoff_function,
        "mean_function": f,
        "deviation_function": v,
        "random_method": "moment_matching",
        "halton_primes": (2, 3),
    }

    mc = MonteCarloOption(params)

    value, error = mc.compute_option_value()
    print(f"Monte Carlo Value: {value:.2f} ± {error:.2f}")

    mc.plot_distribution()


if __name__ == "__main__":
    main()
