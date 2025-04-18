#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 15/04/2025 16:07
#  Updated: 15/04/2025 16:07


class Markowitz(Algorithm):
    def __init__(
            self,
            target_return: float,
            min_return_percentages: List[float],
            total_contracts: int,
            max_num_strategies: int,
            max_contracts: int,
            expected_returns: List[float],
            strats_limited_ct: Union[Dict, None] = None,
            optimality_target: int = 0,
            time_limit: int = 600,
            threads: int = None
    ):
        """
        Class responsible for Markowitz Portfolio Optmization model using Cplex.

        :param target_return: (float) Specifies the historical (greedy) expected return of the portfolio.
        :param min_return_percentages: (list[float]) List containing the multiples of the historical return that the portfolio must achieve, in ascending order, ranging from 0 to 1.
        :param total_contracts: (int) Specifies the total number of contracts that will be allocated in the strategies.
        :param max_num_strategies: (int) Specifies the maximum number of strategies that can be selected.
        :param max_contracts: (int) Specifies the maximum amount of contracts that will be allocated in a single strategy.
        :param expected_returns: (list[float]) List containing the historical returns of each strategy.
        :param w_init: (list[int]) List containing an initial solution for the W decision variable.
        :param x_init: (list[int]) List containing an initial solution for the X decision variable.
        :param optimality_target: (int) Specifies type of optimality that CPLEX targets (For more information search in the cplex documentation).
        :param time_limit: (int) Specifies the timeout that cplex will search for a solution to the problem (In seconds).
        :param strats_limited_ct: (dict) Limit strategies dictionary.
        """
        self.model = Model('markowitz_portfolio_optimization')

        self.target_return = target_return
        self.min_return_percentages = min_return_percentages.copy()
        self.use_callback = len(min_return_percentages) != 0
        self.total_contracts = total_contracts
        self.max_num_strategies = max_num_strategies
        self.max_contracts = max_contracts
        self.expected_returns = expected_returns
        self.strats_limited_ct = strats_limited_ct
        self.initial_percentage = self.min_return_percentages.pop(0)
        self.num_strategies = len(expected_returns)
        self.BIG_M = 9999
        self.BIG_O = self.num_strategies
        self.omega = 50
        self.optimality_target = optimality_target
        self.time_limit = time_limit
        self.threads = threads
        self.w_solutions = dict()
        self.x_solutions = dict()
        if strats_limited_ct is None:
            self.strats_limited_ct = dict()
        self._set_model_configuration()
        self.cov_matrix = None
        self.strategies = None
        self.portfolios = None

    def _set_decision_variables(self):
        """
        Set the decisions variables used in the markowitz model.
        """
        self.w = self.model.continuous_var_list(self.num_strategies, lb=0, ub=1, name='W')
        self.x = self.model.integer_var_list(self.num_strategies, lb=0, ub=1, name='X')

    def _set_constraints(self):
        """
        Set the constraints used in the markowitz model.
        """

        # TEMPLATE MODEL

        # Constraint that guarantees the maximum allocation of contracts
        self.model.add_constraints(
            [(self.w[i] <= ((1 / self.total_contracts) * self.max_contracts * self.x[i])) for i in
             range(self.num_strategies)]
        )
        # Constraint that guarantees the minimum allocation of contracts
        self.model.add_constraints(
            [(self.w[i] >= ((1 / self.total_contracts) * self.x[i])) for i in range(self.num_strategies)]
        )

        # Binding restrictions between W and X
        self.model.add_constraints(
            [(self.x[i] <= (self.BIG_M * self.w[i])) for i in range(self.num_strategies)]
        )
        self.model.add_constraints(
            [(self.x[i] >= (self.w[i])) for i in range(self.num_strategies)]
        )

        # Constraint that controls the maximum number of strategies that can be selected
        self.model.add_constraint(
            self.model.sum(self.x) <= self.max_num_strategies
        )

        # Constraint that controls the total amount of contracts distributed across all strategies
        self.model.add_constraint(
            self.model.sum(self.w) == 1
        )

        # Constraint that controls the minimum return of the portfolio
        self.model.add_constraint(
            self.model.sum([(self.expected_returns[i] * self.w[i]) for i in range(self.num_strategies)]) >= (
                        self.target_return * self.initial_percentage)
        )

        # Constraint that controls the maximum number of contracts that can be allocated in a single strategy
        for i in self.strats_limited_ct.keys():
            self.model.add_constraint(
                self.w[i] <= ((1 / self.total_contracts) * self.strats_limited_ct[i] * self.x[i])
            )

    def _set_lazy_constraints_callback(self, verbose: bool = False):
        """
        Set the lazy constraints callback used in the markowitz model for minimum return percentages relative to the historical return.
        """
        dict_lazy_constraints = dict()
        for p in range(len(self.min_return_percentages)):
            cst = self.model.sum([(self.expected_returns[i] * self.w[i]) for i in range(self.num_strategies)]) >= (
                        self.target_return * self.min_return_percentages[p])
            dict_lazy_constraints[self.min_return_percentages[p]] = cst

        callback: ReturnPercentagesCallback = self.model.register_callback(ReturnPercentagesCallback)
        callback.set_callback_variables(
            min_return_percentages=self.min_return_percentages.copy(),
            lazy_constraints=dict_lazy_constraints,
            current_percentage=self.initial_percentage,
            w_solutions=self.w_solutions,
            x_solutions=self.x_solutions,
            num_strategies=self.num_strategies,
            verbose=verbose
        )
        self.model.lazy_callback = callback

    def _set_objective_function(self):
        """
        Set the objective function used in the markowitz model.
        """
        risk_expr = self.model.sum(
            self.w[i] * self.cov_matrix[i][j] * self.w[j] for i in range(self.num_strategies) for j in
            range(i, self.num_strategies))
        self.model.minimize((risk_expr))

    def _set_model_configuration(self):
        """
        Set the model configuration.
        """
        self.model.set_time_limit(self.time_limit)
        self.model.parameters.optimalitytarget = self.optimality_target
        self.model.parameters.preprocessing.presolve = 1
        if self.threads is not None:
            self.model.parameters.threads = self.threads

    def _compute_covariance_matrix(self, df_returns: pd.DataFrame):
        """
        Compute the covariance matrix used in the markowitz model.

        :param df_returns: (pd.DataFrame) DataFrame containing the returns of the strategies.

        """
        self.cov_matrix = get_numba_matrix(df_returns.fillna(0).cov().values)

    def run(self, data: StrategiesDataset, model_verbose: bool = False,
            callback_verbose: bool = False) -> ModelPortfolios:
        """
        Run the model with the defined constants.

        :param data: (StrategiesDataset) data collection with at least the strategies returns dataframe.
        :param portfolios: (ModelPortfolios) Previous model Portfolios (if exists).
        :param model_verbose: (bool) Verbose of the model.
        :param callback_verbose: (bool) Verbose of the callback.

        :return: (ModelPortfolios) Portfolio with the strategies and the number of contracts assigned to each one.
        """
        df_returns = data.returns
        self.strategies = df_returns.columns.tolist()
        self._compute_covariance_matrix(df_returns)
        self._set_decision_variables()
        self._set_constraints()
        self._set_objective_function()
        if self.use_callback:
            self._set_lazy_constraints_callback(callback_verbose)
        self._set_model_configuration()
        self.model.solve(log_output=model_verbose)
        self._set_model_porfolios()

        return self.portfolios

    def _set_model_porfolios(self):
        self.portfolios = ModelPortfolios()

        for pct in [self.initial_percentage] + self.min_return_percentages:

            # Grab the solution for current percentage
            x_solution = self.x_solutions[pct]
            w_solution = self.w_solutions[pct]

            # Get the indexes of the selected strategies
            cols_indexes = [index for index, element in enumerate(x_solution) if round(element) == 1]

            # Get the selected strategies
            selected_strategies = [self.strategies[i] for i in cols_indexes]
            if np.sum(w_solution) >= 2:
                contracts = [np.round(w_solution[i]) for i in cols_indexes]
            else:
                contracts = [w_solution[i] * self.total_contracts for i in cols_indexes]

            # Create the portfolio
            portfolio = pd.DataFrame({"strategies": selected_strategies, "contracts": contracts})

            # Transform the contracts in integer
            portfolio = get_integer_contracts(portfolio, self.total_contracts, self.max_contracts)
            portfolio = Portfolio(strategies=portfolio.strategies.tolist(), weights=portfolio.contracts.tolist())
            self.portfolios[pct] = portfolio

    def get_model_solution(self) -> pd.Series:
        """
        Returns the found values for both model decision variables.

        :return: (dict) Both model decision variables in the format of dictionary.

        """
        if self.use_callback == False:
            self.w_solutions[self.initial_percentage] = [self.w[i].solution_value for i in range(self.num_strategies)]
            self.x_solutions[self.initial_percentage] = [self.x[i].solution_value for i in range(self.num_strategies)]
        return self.w_solutions, self.x_solutions