#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 14:11
#  Updated: 06/04/2025 13:30


import numpy as np
import pandas as pd
from docplex.mp.model import Model
import yfinance as yf
from datetime import datetime, timedelta

from models.callbacks import ssd_lazy_callback
from models.callbacks.mip_lazy_callback import MIPLazyCallback

pd.options.display.float_format = "{:.4f}".format

class MIPLazyDemo:
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
        print("\n=== INICIANDO RESOLUÇÃO DO MIP ===")
        model = Model("\tMIP_Lazy_Demo")
        model.context.cplex_parameters.threads = 1  # Para melhor acompanhamento

        # Variáveis inteiras com limites mais interessantes
        x = model.integer_var(lb=0, ub=10, name='x')
        y = model.integer_var(lb=0, ub=10, name='y')

        # Função objetivo mais desafiadora
        model.maximize(2 * x + 3 * y)

        # Restrições iniciais mínimas para criar soluções candidatas interessantes
        model.add_constraint(x <= 8, "ct_ub_x")
        model.add_constraint(y <= 6, "ct_ub_y")

        print("\nRestrições iniciais do modelo:")
        for ct in model.iter_constraints():
            print(f"\tName: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}")


        # Configuração do callback
        lazy_cb = model.register_callback(MIPLazyCallback)
        lazy_cb.x = x
        lazy_cb.y = y
        lazy_cb.verbose = False

        print("\nIniciando processo de otimização...")
        sol = model.solve(log_output=True)

        if model.solve_details.status_code == 3:  # infeasible model
            print("\tInfeasible Model")

        if sol:
            print("\n=== RESULTADO FINAL ===")
            print(f"\tSolução ótima:")
            print(f"\tx = {sol[x]}, y = {sol[y]}")
            print(f"\tValor objetivo: {sol.objective_value}")

            print("\nVerificação pós-otimização:")
            print(f"\tx + 3y = {sol[x] + 3 * sol[y]} (deve ser ≤ 10)")
            print(f"\tx + y = {sol[x] + sol[y]} (deve ser ≤ 8)")
        else:
            raise ValueError("\tNo solution found. Try adjusting parameters.")

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