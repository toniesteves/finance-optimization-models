#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 08/04/2025 00:24
#  Updated: 08/04/2025 00:24

from typing import List, Dict
from docplex.mp.model import DOcplexException
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
from cplex.callbacks import LazyConstraintCallback, SolveCallback
class ReturnPercentagesCallback(ConstraintCallbackMixin, LazyConstraintCallback, SolveCallback):

    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)

    def set_callback_variables(
            self,
            min_return_percentages: List[float],
            lazy_constraints: Dict[float, DOcplexException],
            current_percentage: float,
            w_solutions: Dict,
            x_solutions: Dict,
            num_strategies: int,
            verbose: bool
        ):
        self.min_return_percentages = min_return_percentages
        self.lazy_constraints = lazy_constraints
        self.current_percentage = current_percentage
        self.w_solutions = w_solutions
        self.x_solutions = x_solutions
        self.num_strategies = num_strategies
        self.verbose = verbose

    def __call__(self):
        if self.get_cplex_status() == self.status.optimal:
            current_w = []
            current_x = []

            for s in range(self.num_strategies):
                current_w.append(self.get_values("W_" + str(s)))
                current_x.append(self.get_values("X_" + str(s)))

            if self.current_percentage != "none":
                self.w_solutions[self.current_percentage] = current_w
                self.x_solutions[self.current_percentage] = current_x

            if self.verbose:
                print(f'[INFO]: Model solve status: {self.get_cplex_status()}')
                print(f'[INFO]: Model objective value: {self.get_objective_value()}')
                print(f'[INFO]: Number of selected strategies: {np.sum(current_x)}')

            if len(self.min_return_percentages) > 0:
                self.current_percentage = self.min_return_percentages.pop(0)

                if self.verbose:
                    print(f"* Adding a new Lazy Constraint, represented by {self.current_percentage * 100:.2f}%")

                cst = self.lazy_constraints[self.current_percentage]
                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(cst)
                self.add(cpx_lhs, sense, cpx_rhs)
            else:
                # terminate the model
                if self.verbose:
                    print()
                    print("[INFO]: Finalizing Optimization")
                    print("-------------------------------------------------------")
                    print()