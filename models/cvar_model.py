#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:32
#  Updated: 06/04/2025 13:32

import numpy as np
import pandas as pd
from docplex.mp.model import Model
import yfinance as yf
from datetime import datetime, timedelta

from plotting.finance import plot_cvar_results

pd.options.display.float_format = "{:.4f}".format


class CVaRPortfolio:
    """
    Implementation of CVar Portfolio Optimization using DOCPLEX.

    Parameters:
    - tickers: List of asset tickers
    - start_date: Start date for historical data
    - end_date: End date for historical data
    - risk_free_rate: Risk-free rate (default 0)
    """

    def __init__(self, tickers, start_date, end_date, risk_free_rate=0):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.returns = None
        self.optimal_weights = None
        self.optimal_var = None
        self.optimal_cvar = None
        self.risk_free_rate = risk_free_rate
        self._download_data()
        self._calculate_metrics()

    def _download_data(self):
        """Download historical price data from Yahoo Finance"""
        print("\n=== DOWNLOAD HISTORICAL ASSET DATA ===")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=False)['Adj Close']

        data.bfill(inplace=True)
        data.ffill(inplace=True)

        self.returns = data.pct_change().dropna()

    def _calculate_metrics(self):
        """Calculate expected returns and historical returns"""

        self.mean_returns = np.array(self.returns.mean())
        self.historical_returns = np.array(self.returns)
        self.cov_matrix = self.returns.cov()
        self.benchmark = self.returns.iloc[:, -1]


    def optimize(self, target_return=None, risk_aversion=1.0, log_output=True, verbose=False, output_log=True, alpha=0.95, budget=1.0):
        """
        CVaR portfolio optimization using Docplex

        Parameters:
        - returns: numpy array of shape (n_scenarios, n_assets) containing return scenarios
        - alpha: confidence level for CVaR (between 0 and 1)
        - budget: total budget to allocate across assets

        Returns:
        - Optimal portfolio weights
        - Optimal CVaR value
        - Optimal VaR value (the threshold value-at-risk)
        """
        n = len(self.tickers)
        n_scenarios = self.historical_returns.shape[0]

        print("\n=== INICIANDO RESOLUÇÃO DO CVar ===")
        model = Model(name='CVar')
        model.context.cplex_parameters.threads = 1  # Para melhor acompanhamento

        # Variáveis inteiras com limites mais interessantes
        w = model.continuous_var_list(n, lb=0, ub=1, name='w')
        var = model.continuous_var(name='VaR')  # Value-at-Risk
        z = model.continuous_var_list(n_scenarios, lb=0, name='z')  # Auxiliary variables

        # Constraints
        model.add_constraint(model.sum(w) == budget)  # Budget constraint

        for i in range(n_scenarios):
            portfolio_return = model.sum(self.historical_returns[i, j] * w[j] for j in range(n))
            model.add_constraint(z[i] >= var - portfolio_return)

        # Função objetivo mais desafiadora
        cvar = var - (1 / (n_scenarios * (1 - alpha))) * model.sum(z)
        model.maximize(cvar)


        if verbose:
            print("\nRestrições iniciais do modelo:")
            for ct in model.iter_constraints():
                print(f"Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}")

        print("\nStarting optimization...")
        sol = model.solve(log_output=True)

        if model.solve_details.status_code == 3:  # infeasible model
            print("Infeasible Model")

        if sol:
            print("\n=== FINAL RESULT ===")
            print(f"Optimal Solution:")
            optimized_weights = [sol[weight] for weight in w]

            self.optimal_weights = {self.tickers[i]: w[i].solution_value for i in range(n)}
            self.optimal_var = var.solution_value
            self.optimal_cvar = cvar.solution_value

            print(np.array(optimized_weights))
            print(f"\nObjective Function: {sol.objective_value}")
            print(f"\nVaR at {alpha * 100:.0f}% confidence level: {self.optimal_var:.4f}")
            print(f"CVaR at {alpha * 100:.0f}% confidence level: {self.optimal_cvar:.4f}")

            model.export_as_lp("models/cvar.lp")
        else:
            print("No Solutions Found!")

        plot_cvar_results(self.historical_returns, np.array(optimized_weights), cvar=self.optimal_cvar, var=self.optimal_var, alpha=alpha)

        return self.optimal_weights


    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio return and volatility for given weights

        Parameters:
        - weights: Dictionary of {ticker: weight}

        Returns:
        - Tuple of (expected_return, volatility, sharpe_ratio)
        """

        metrics = {}

        if not isinstance(weights, dict):
            raise ValueError("Weights must be a dictionary")

        print("\n=== COMPUTE PORTFOLIO METRICS ===")

        w = np.array([weights[t] for t in self.tickers])

        metrics["expected_return"] = np.dot(w, self.mean_returns)
        metrics["volatility"]      = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))
        metrics["sharpe_ratio"]    = (metrics["expected_return"] - self.risk_free_rate) / metrics["volatility"]

        return metrics
