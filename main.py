#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:23
#  Updated: 06/04/2025 13:23

from models.core import *
from util.core import read_input, inspect_input

if __name__ == "__main__":

    try:
        input = read_input('input.json')
        inspect_input(input)
    except Exception as e:
        print(f"Error: {e}")

    try:
        portfolio_class = portfolio_classes.get(input.model)
    except Exception as e:
        raise ValueError(f"Unknown portfolio type: {input.model}")

    portfolio = portfolio_class(input.tickers, input.start_date, input.end_date)

    optimal_weights = portfolio.optimize(log_output=False)

    print()