#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 12:56
#  Updated: 06/04/2025 12:56

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_efficient_frontier(self, num_points=20):
    """
    Plot the efficient frontier by solving for different target returns

    Parameters:
    - num_points: Number of points to plot
    """
    min_return = self.mean_returns.min()
    max_return = self.mean_returns.max()
    target_returns = np.linspace(min_return, max_return, num_points)

    volatilities = []
    returns = []

    print("Calculating efficient frontier...")
    for ret in target_returns:
        try:
            weights = self.optimize(target_return=ret)
            _, vol, _ = self.calculate_portfolio_metrics(weights)
            volatilities.append(vol)
            returns.append(ret)
        except:
            continue

    plt.figure(figsize=(10, 6))
    plt.plot(volatilities, returns, 'b-', label='Efficient Frontier')
    plt.scatter(np.sqrt(np.diag(self.cov_matrix)), self.mean_returns,
                c='red', label='Individual Assets')

    # Annotate asset tickers
    for i, txt in enumerate(self.tickers):
        plt.annotate(txt, (np.sqrt(self.cov_matrix.iloc[i, i]), self.mean_returns[i]))

    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_cvar_results(returns, weights, var, cvar, alpha):
    """
    Plot the distribution of portfolio returns with VaR and CVaR markers

    Parameters:
    - returns: numpy array of return scenarios (n_scenarios, n_assets)
    - weights: optimal portfolio weights
    - var: optimal VaR value
    - cvar: optimal CVaR value
    - alpha: confidence level
    """
    # Calculate portfolio returns for each scenario
    portfolio_returns = returns @ weights

    plt.figure(figsize=(12, 6))

    # Plot 1: Histogram of portfolio returns with VaR/CVaR
    plt.subplot(1, 2, 1)
    sns.histplot(portfolio_returns, kde=True, bins=30)
    plt.axvline(x=var, color='r', linestyle='--', label=f'VaR ({alpha * 100:.0f}%) = {var:.4f}')
    plt.axvline(x=cvar, color='g', linestyle='--', label=f'CVaR ({alpha * 100:.0f}%) = {cvar:.4f}')
    plt.title('Portfolio Return Distribution')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot 2: Cumulative distribution with VaR/CVaR
    plt.subplot(1, 2, 2)
    sorted_returns = np.sort(portfolio_returns)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    plt.plot(sorted_returns, cdf)
    plt.axvline(x=var, color='r', linestyle='--', label=f'VaR ({alpha * 100:.0f}%)')
    plt.axvline(x=cvar, color='g', linestyle='--', label=f'CVaR ({alpha * 100:.0f}%)')
    plt.axhline(y=1 - alpha, color='b', linestyle=':', label=f'{alpha * 100:.0f}% confidence level')
    plt.title('Cumulative Distribution Function')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Cumulative Probability')
    plt.legend()

    plt.tight_layout()
    plt.show()
