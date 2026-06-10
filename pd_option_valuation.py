import numpy as np
import plotly.graph_objects as go

class PathDependentOptionMC:
    def __init__(self, params):
        """
        Monte Carlo simulation for a path-dependent option.

        Parameters (stored in params dict):
        - num_paths: number of simulated paths
        - num_steps: number of time steps
        - initial_stock_price: initial S0
        - theta, alpha, beta, gamma, sigma: model parameters
        - interest_rate: risk-free rate
        - expiration_time: time to maturity
        """
        self.params = params
        self.rng = np.random.default_rng(seed=0)

    def simulate_paths(self):
        N = self.params["num_paths"]
        M = self.params["num_steps"]
        S0 = self.params["initial_stock_price"]
        T = self.params["expiration_time"]
        dt = T / M

        theta = self.params["theta"]
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        sigma = self.params["sigma"]

        paths = np.zeros((N, M))
        paths[:, 0] = S0

        Z = self.rng.normal(0, 1, size=(N, M))

        for step in range(M - 1):
            S_prev = paths[:, step]
            drift = (alpha * theta - beta * S_prev) * dt
            diffusion = sigma * np.abs(S_prev) ** gamma * np.sqrt(dt) * Z[:, step]
            paths[:, step + 1] = S_prev + drift + diffusion

        return paths

    def compute_payoff(self, paths):
        """
        Example payoff: terminal price minus minimum over path (Asian/Lookback style).
        """
        min_stock_prices = np.min(paths[:, 1:], axis=1)
        payoffs = np.maximum(0, paths[:, -1] - min_stock_prices)
        return payoffs

    def compute_option_value(self):
        paths = self.simulate_paths()
        payoffs = self.compute_payoff(paths)
        discounted = np.exp(-self.params["interest_rate"] * self.params["expiration_time"]) * payoffs
        value = np.mean(discounted)
        error = np.std(discounted) / np.sqrt(self.params["num_paths"])
        return value, error, discounted

    def plot_distribution(self, discounted_values, num_bins=50):
        """
        Plot the distribution of discounted option payoffs.
        """
        hist = np.histogram(discounted_values, bins=num_bins, density=True)
        x = 0.5 * (hist[1][1:] + hist[1][:-1])
        y = hist[0]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=y, marker_color='lightblue', name='Payoff Distribution'))

        mean_val = np.mean(discounted_values)
        fig.add_trace(go.Scatter(
            x=[mean_val, mean_val],
            y=[0, max(y)*1.1],
            mode="lines",
            line=dict(color='red', dash='dash'),
            name=f"Mean = {mean_val:.2f}"
        ))

        fig.update_layout(
            title="Monte Carlo Path-Dependent Option Payoff Distribution",
            xaxis_title="Discounted Payoff Value",
            yaxis_title="Density",
            template="plotly_white",
            bargap=0.05
        )
        fig.show()


def main():
    params = {
        "num_paths": 100000,
        "num_steps": 1000,
        "initial_stock_price": 70154,
        "theta": 70100,
        "alpha": 0.01,
        "beta": 0.01,
        "gamma": 1.05,
        "sigma": 0.26,
        "interest_rate": 0.01,
        "expiration_time": 1.0
    }

    mc = PathDependentOptionMC(params)
    value, error, discounted = mc.compute_option_value()
    print(f"Path-Dependent Option Value: {value:.2f} ± {error:.2f}")
    mc.plot_distribution(discounted)

if __name__ == "__main__":
    main()
