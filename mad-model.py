import pandas as pd
import numpy  as np
import yfinance as yf
import cplex
import docplex


def get_stocks():
    pass


def get_returns_data():
    pass

def run_model():
    print("\n=== INICIANDO RESOLUÇÃO DO SSD ===")
    model = Model(name='SSD')
    model.context.cplex_parameters.threads = 1  # Para melhor acompanhamento


    w = model.continuous_var_list(n_assets, name="w")
    d = model.continuous_var_list(n_scenarios, name="d")

    mu = model.sum(w[i] * expected_returns[i] for i in range(n_assets))

    model.add_constraint(model.sum(w) == 1, ctname="budget")

    for t in range(n_scenarios):
        scenario_return = model.sum(w[i] * historical_returns[t, i] for i in range(n_assets))

        model.add_constraint(d[t] >= scenario_return - mu)
        model.add_constraint(d[t] >= mu - scenario_return)

    model.add_constraints(w[i] >= 0 for i in range(n_assets))

    mad = model.sum(d[t] / n_scenarios for t in range(n_scenarios))

    print("\nRestrições iniciais do modelo:")
    for ct in model.iter_constraints():
        print(f"Name: {ct.name}, Expression: {ct.left_expr} {ct.sense} {ct.right_expr}")

    model.minimize(mad)

    print(model.print_information())

    print("\nIniciando processo de otimização...")
    sol = model.solve(log_output=True)

    if model.solve_details.status_code == 3:  # infeasible model
        print("Infeasible Model")

    if sol:
        print("\n=== RESULTADO FINAL ===")
        print(f"Solução ótima:")
        print(f"x = {sol[x]}, y = {sol[y]}")
        print(f"Valor objetivo: {sol.objective_value}")

        optimized_weights = [sol[weight] for weight in w]
        weights = np.array(optimized_weights)
        print(f"{sol.get_objective_value()}")
        model.export_as_lp("toni_ssd_model.lp")
    else:
        print("Nenhuma solução encontrada!")



if __name__ == "__main__":
    run_model()




# print(model.print_information())

sol = model.solve(log_output=False, clean_before_solve = True)

# print(f"* Solve status is: '{model.solve_details.status}'")
if model.solve_details.status_code == 103: # infeasible model
    print("Infeasible Model")

optimized_weights = [sol[weight] for weight in w]
weights = np.array(optimized_weights)

model.solution.export(SOL_PATH + "solution.json")

return sol.get_objective_value(), weights