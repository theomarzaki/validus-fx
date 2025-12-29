import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plotIRRDistributions(results):

    for index, result in results.iterrows():
        irrs = result["IRR"]
        if len(result["IRR"]) > 2000:
            irrs = np.random.choice(result["IRR"], 2000, replace=False)
        fig = sns.histplot(irrs, label=result["Strategy Name"], bins=50, kde=True)
    fig.set_xlabel("IRR")
    fig.set_ylabel("Density")
    fig.set_title("IRR Distributions")
    fig.legend()
    plt.savefig("Figures/StrategyIRRDist.svg")
    fig.grid(True, alpha=0.3)
    plt.show()


def plotMeanRiskScatter(results):
    for index, result in results.iterrows():
        fig = sns.scatterplot(
            x=[result["IRR"].std()],
            y=[result["IRR"].mean()],
            s=800,
            # marker='x',
            label=result["Strategy Name"],
        )

    fig.set_xlabel("IRR Standard Deviation (Risk)")
    fig.set_ylabel("Mean IRR (Return)")
    fig.set_title("Risk-Return Tradeoff")
    plt.savefig("Figures/StrategyIRRScatter.svg")
    fig.grid(True, alpha=0.3)
    plt.show()


def plotMultipleDistributions(results):
    for index, result in results.iterrows():
        fig = sns.histplot(
            result["Multiples"], label=result["Strategy Name"], kde=False
        )
    fig.set_xlabel("Multiple on Invested Capital")
    fig.set_ylabel("Density")
    fig.set_xlim((0, 5))
    fig.set_title("Multiple on Invested Capital Distributions")
    fig.legend()
    plt.savefig("Figures/StrategyMultipleDist.svg")
    fig.grid(True, alpha=0.3)
    plt.show()


def plotVaR(results):

    fig = sns.barplot(y=results["VaR"],x=results['Strategy Name'])

    fig.set_xlabel("Strategy Name")
    fig.set_ylabel("VaR")
    fig.set_title("VaR (95%) for Strategy")
    plt.savefig("Figures/StrategyVaR.svg")
    fig.grid(True, alpha=0.3)
    plt.show()

def plotCVaR(results):

    fig = sns.barplot(y=results["CVaR"],x=results['Strategy Name'])

    fig.set_xlabel("Strategy Name")
    fig.set_ylabel("CVaR")
    fig.set_title("CVaR (95%) for Strategy")
    fig.tick_params(axis='x', rotation=45)
    plt.savefig("Figures/StrategyCVaR.svg")
    fig.grid(True, alpha=0.3)
    plt.show()


def plotBestWorstIRRScenario(results):

    fig, axes = plt.subplots(2, 1, sharex=True)

    for index, result in results.iterrows():
        irrs = result["worst_IRR"]
        fig = sns.histplot(irrs, label=result["Strategy Name"], bins=10, kde=False,ax=axes[0])
    fig.set_title("Worst Case IRR Distribution")
    fig.set_ylabel("Density")
    fig.legend()
    fig.grid(True, alpha=0.3)

    for index, result in results.iterrows():
        irrs = result["best_IRR"]
        if len(irrs) > 3000:
            irrs = np.random.choice(result["best_IRR"], 3000, replace=False)
        fig = sns.histplot(irrs, label=result["Strategy Name"], bins=10, kde=False,ax=axes[1])
    
    fig.set_xlabel("IRR")
    fig.set_title("Best Case IRR Distribution")
    fig.set_ylabel("Density")
    fig.legend()
    plt.savefig("Figures/StrategyIRRDistExtreme.svg")
    fig.grid(True, alpha=0.3)
    plt.show()


def plotBestWorstCaseMultiples(results):

    fig, axes = plt.subplots(2, 1, sharex=True)

    for index, result in results.iterrows():
        multiple = result["worst_Multiple"]
        fig = sns.histplot(multiple, label=result["Strategy Name"], bins=50, kde=True,ax=axes[0])
    fig.set_title("Worst Case Multiple Distribution")
    fig.set_ylabel("Density")
    fig.legend()
    fig.grid(True, alpha=0.3)

    for index, result in results.iterrows():
        multiple = result["best_Multiple"]
        fig = sns.histplot(multiple, label=result["Strategy Name"], bins=50, kde=True,ax=axes[1])
    
    fig.set_xlabel("Multiple")
    fig.set_title("Best Case Multiple Distribution")
    fig.set_ylabel("Density")
    fig.legend()
    plt.savefig("Figures/StrategyMultipleDistExtreme.svg")
    fig.grid(True, alpha=0.3)
    plt.show()


def plotComparisons(results):
    plotIRRDistributions(results)
    plotMeanRiskScatter(results)
    plotMultipleDistributions(results)
    plotVaR(results)
    plotCVaR(results)

def plotExtremes(results):
    plotBestWorstIRRScenario(results)
    plotBestWorstCaseMultiples(results)

