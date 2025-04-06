import pandas as pd
import numpy  as np
import yfinance as yf
from docplex.mp.model import Model
import datetime

def get_stocks():
    tickers = [
        "AXP", "AAPL", "AMGN", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS",
        "HD", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE",
        "PG", "TRV", "UNH", "V", "WBA", "WMT", "XOM"]

    # number of years
    n_years = 3.0

    # historical period
    end_date = datetime.datetime.today().date()
    start_date = end_date - datetime.timedelta(round(n_years * 365))

    print("\n=== DOWNLOAD HISTORICAL ASSET DATA ===")
    assets = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)["Adj Close"]

    assets.bfill(inplace=True)
    assets.ffill(inplace=True)

    return assets


def get_returns_data():

    daily_returns = get_stocks()
    daily_returns = daily_returns.diff()[1:] / daily_returns.shift(1)[1:]

    return  daily_returns

def report_stats():
    pass


def run_model(verbose=False, output_log=True):

    returns = get_returns_data()

    expected_returns = np.array(returns.mean())
    historical_returns = np.array(returns)

    n_assets = len(expected_returns)
    n_scenarios = historical_returns.shape[0]

    print("\n=== INICIANDO RESOLUÇÃO DO MAD ===")
    model = Model(name='Markowitz')
    model.context.cplex_parameters.threads = 1  # Para melhor acompanhamento

    w = model.continuous_var_list(n_assets, name="w", lb=0, ub=1)
    d = model.continuous_var_list(n_scenarios, name="d", lb=0, ub=1)

    mu = model.sum(w[i] * expected_returns[i] for i in range(n_assets))

    model.add_constraint(model.sum(w) == 1, ctname="budget")

    for t in range(n_scenarios):
        scenario_return = model.sum(w[i] * historical_returns[t, i] for i in range(n_assets))

        model.add_constraint(d[t] >= scenario_return - mu)
        model.add_constraint(d[t] >= mu - scenario_return)

    model.add_constraints(w[i] >= 0 for i in range(n_assets))

    mad = model.sum(d[t] / n_scenarios for t in range(n_scenarios))

    if verbose:
        print("\nRestrições iniciais do modelo:")
        for ct in model.iter_constraints():
            print(f"Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}")

    model.minimize(mad)

    print(model.print_information())

    print("\nIniciando processo de otimização...")
    sol = model.solve(log_output=output_log)

    if model.solve_details.status_code == 3:  # infeasible model
        print("Infeasible Model")

    if sol:
        print("\n=== RESULTADO FINAL ===")
        print(f"\nSolução ótima:")
        optimized_weights = [sol[weight] for weight in w]
        weights = np.array(optimized_weights)
        print(weights)
        print(f"\nValor objetivo: {sol.objective_value}")
        print()

        model.export_as_lp("models/mad.lp")
    else:
        print("Nenhuma solução encontrada!")



if __name__ == "__main__":
    run_model()

