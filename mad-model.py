#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 01:27
#  Updated: 06/04/2025 01:27


import numpy as np
import pandas as pd
from docplex.mp.model import Model
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


pd.options.display.float_format = "{:.4f}".format

class MADPortfolio:
    """
    Implementation of MAD Mean-Variance Portfolio Optimization using DOCPLEX.

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


    def optimize(self, target_return=None, risk_aversion=1.0, log_output=True, verbose=False, output_log=True):
        """
        Solve the MAD optimization problem

        Parameters:
        - target_return: Optional target return constraint

        Returns:
        - Dictionary of optimal weights
        """
        n = len(self.tickers)
        n_scenarios = self.historical_returns.shape[0]

        print("\n=== INICIANDO RESOLUÇÃO DO MAD ===")
        model = Model(name='MAD')
        model.context.cplex_parameters.threads = 1  # Para melhor acompanhamento

        w = model.continuous_var_list(n, name="w", lb=0, ub=1)
        d = model.continuous_var_list(n_scenarios, name="d", lb=0, ub=1)

        mu = model.sum(w[i] * self.mean_returns[i] for i in range(n))

        model.add_constraint(model.sum(w) == 1, ctname="budget")

        for t in range(n_scenarios):
            scenario_return = model.sum(w[i] * self.historical_returns[t, i] for i in range(n))

            model.add_constraint(d[t] >= scenario_return - mu)
            model.add_constraint(d[t] >= mu - scenario_return)

        model.add_constraints(w[i] >= 0 for i in range(n))

        mad = model.sum(d[t] / n_scenarios for t in range(n_scenarios))

        if verbose:
            print("\nRestrições iniciais do modelo:")
            for ct in model.iter_constraints():
                print(f"Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}")

        model.minimize(mad)

        print(model.print_information())

        print("\nStarting Optimization...")
        sol = model.solve(log_output=output_log)

        if model.solve_details.status_code == 3:  # infeasible model
            print("Infeasible Model")

        if sol:
            print("\n=== RESULTADO FINAL ===")
            print(f"\nOptimal Solution:")
            optimized_weights = [sol[weight] for weight in w]
            self.optimal_weights = {self.tickers[i]: w[i].solution_value for i in range(n)}

            weights = np.array(optimized_weights)
            print(weights)
            print(f"\nObjective Function: {sol.objective_value}")

            model.export_as_lp("models/mad.lp")
        else:
            raise ValueError("No solution found. Try adjusting parameters.")

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



# Example Usage
if __name__ == "__main__":
    tickers = [
        "AXP", "AAPL", "AMGN", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS",
        "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
        "PG", "TRV", "UNH", "V", "WBA", "WMT", "XOM"]

    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=200)).strftime('%Y-%m-%d')

    portfolio = MadPortfolio(tickers, start_date, end_date)
    optimal_weights = portfolio.optimize(log_output=False)

    print("\nOptimal Weights Allocation(Risk Neutral):")
    print(pd.DataFrame([optimal_weights]))

    metrics = portfolio.calculate_portfolio_metrics(optimal_weights)
    print(pd.DataFrame([metrics]))

    print()