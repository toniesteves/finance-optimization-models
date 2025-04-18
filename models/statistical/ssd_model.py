#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 18/04/2025 00:40
#  Updated: 08/04/2025 00:00


import numpy as np
import pandas as pd
from docplex.mp.model import Model
import yfinance as yf
from datetime import datetime, timedelta

from models.callbacks import ssd_lazy_callback
from models.callbacks.ssd_lazy_callback import SSDLazyCallback

pd.options.display.float_format = "{:.4f}".format

class SSD:
    """
    Implementation of SSD Portfolio Optimization using DOCPLEX.

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
        self.n_scenarios = None
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
        self.benchmark = self.returns.iloc[:, -1]


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

        print("\n=== STARTING SSD MODEL ===")
        model = Model(name='SSD')
        model.context.cplex_parameters.threads = 1

        # Variáveis
        w  = model.continuous_var_list(self.n_scenarios, lb=0, ub=1, name='wl')
        Vs = model.continuous_var_list(self.n_scenarios, lb=0, ub=1, name='V')
        V  = model.continuous_var(name="V")
        cb = model.binary_var(name='cb_temp')  # Alterado para binary_var

        # Restrições básicas
        model.add_constraint(model.sum(w) == 1, ctname="budget")

        for t in range(self.n_scenarios):
            model.add_constraint(Vs[t] >= V, ctname=f"maxmin_{t}")

        model.maximize(V + cb)

        if callback:
            # Registra o callback e passa os parâmetros necessários
            lazy_cb = model.register_callback(SSDLazyCallback)
            lazy_cb.n_assets = self.n_assets
            lazy_cb.n_scenar = n_scenar
            lazy_cb.scenarios = scenarios
            lazy_cb.benchmark = benchmark
            lazy_cb.w_vars = w

        print("\nIniciando processo de otimização...")
        sol = model.solve(log_output=True)

        if sol:
            w_solutions = [sol[wi] for wi in w]
            print("Solução encontrada:")
            print(f"V = {sol[V]}")
            for i, val in enumerate(w_solutions):
                print(f"w[{i}] = {val}")


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

    portfolio = SSDPortfolio(tickers, start_date, end_date)
    optimal_weights = portfolio.optimize(log_output=False)

    print("\nOptimal Weights Allocation(Risk Neutral):")
    print(pd.DataFrame([optimal_weights]))

    metrics = portfolio.calculate_portfolio_metrics(optimal_weights)
    print(pd.DataFrame([metrics]))

    print()