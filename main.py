#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:23
#  Updated: 06/04/2025 13:23

import argparse
import sys
from models.core import *
from util.core import read_input, inspect_input

def _main(args):

    try:
        input = read_input('input.json')
        inspect_input(input)
    except Exception as e:
        print(f"Error: {e}")

    try:
        portfolio_class = PORTFOLIO_CLASSES.get(input.model)
    except Exception as e:
        raise ValueError(f"Unknown portfolio type: {input.model}")

    portfolio = portfolio_class(input.tickers, input.start_date, input.end_date)

    optimal_weights = portfolio.optimize(log_output=False)

    print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main script for processing JSON data.")

    parser.add_argument('-i', "--input_file", type=str, help="Path to the JSON file")
    parser.add_argument('-f', "--file",       type=str, help="Assets returns data")
    parser.add_argument('-d', '--display',    type=int, default=10, help='Number of columns to display (default is 10).')
    parser.add_argument('-v', '--verbose',    action='store_true', help='Enable verbose output.')
    parser.add_argument('-o', '--output',     type=str, help='Specify the output file.')

    args = parser.parse_args()

    _main(args)