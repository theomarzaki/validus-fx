from model.Heston import HestonModel
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class TestHestonModel:
    def __init__(self, model, market_data, initial_params):
        self.model = model
        self.market_data = market_data
        self.year = 1
        self.params = initial_params

        self.simulatePaths()
        self.testOptionPricing()

    def simulatePaths(self):
        n_test_paths = 5
        self.t, self.S_paths, self.vol_paths = self.model.simulate(self.year)

    def visualiseSpotPaths(self):
        for index in range(min(5, self.S_paths.shape[1])):
            fig = sns.lineplot(
                x=self.t,
                y=self.S_paths[:, index],
                alpha=0.6,
                # label=f"Sim Spot Path {index}",
            )
        fig.set_title("Simulated Spot Paths")
        fig.set_xlabel("Time")
        fig.set_ylabel("Spot Rate")
        plt.grid(True, alpha=0.3)
        plt.savefig("Figures/TrialSpotPaths.svg")
        plt.show()

    def visualiseVolPaths(self):
        for index in range(min(5, self.vol_paths.shape[1])):
            fig = sns.lineplot(
                x=np.linspace(0, self.year, self.vol_paths.shape[0]),
                y=self.vol_paths[:, index],
                alpha=0.6,
                # label=f"Sim Vol Path {index}",
            )
        fig.set_title("Simulated Volatility Paths")
        fig.set_xlabel("Time")
        fig.set_ylabel("Volatility")
        # fig.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("Figures/TrialVolPaths.svg")
        plt.show()

    def visualiseTerminalDistribution(self):
        _, S, _ = self.model.simulate(self.year)
        terminal_spot = S[-1, :]
        fig = sns.histplot(
            terminal_spot, bins=50, alpha=0.7, edgecolor="black", kde=True
        )
        fig.set_title("Terminal Spot Distribution")
        fig.set_xlabel("Spot Rate")
        fig.set_ylabel("Density")
        plt.savefig("Figures/TrialTerminalDist.svg")
        # fig.legend()
        fig.grid(True, alpha=0.3)
        plt.show()

