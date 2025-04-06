#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:23
#  Updated: 06/04/2025 13:23

import  pandas as pd

from models.cvar_model import CVaRPortfolio
from util.core import read_input, inspect_input

if __name__ == "__main__":

    try:
        input = read_input('input.json')
        inspect_input(input)
    except Exception as e:
        print(f"Error: {e}")

    print(input.tickers)

    portfolio = CVaRPortfolio(input.tickers, input.start_date, input.end_date)
    optimal_weights = portfolio.optimize(log_output=False)

    print("\nOptimal Weights Allocation(Risk Neutral):")
    print(pd.DataFrame([optimal_weights]))

    metrics = portfolio.calculate_portfolio_metrics(optimal_weights)
    print(pd.DataFrame([metrics]))

    print()