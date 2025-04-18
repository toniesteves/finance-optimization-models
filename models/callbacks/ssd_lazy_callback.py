#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 01:45
#  Updated: 06/04/2025 01:45

from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin


class SSDLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback, SolveCallback):
    def __init__(self, env):
        LazyConstraintCallback.__init__(self, env)
        ConstraintCallbackMixin.__init__(self)
        self.n_calls = 0
        self.n_assets = None  # Será definido depois
        self.n_scenar = None  # Será definido depois
        self.scenarios = None  # Será definido depois
        self.benchmark = None  # Será definido depois
        self.w_vars = None

    def __call__(self):
        self.n_calls += 1
        print(f"\n--> CUSTOM CALLBACK CALLED #{self.n_calls} ---")

        # Obtém a solução corrente
        current_solution = self.make_complete_solution()
        print(f"CURRENT_SOLUTION: {current_solution}")

        current_w_values = {var.name: self.get_values(var.index) for var, val in zip(self.w_vars, self.w_vars)}
        print(f"CURRENT_W: {current_w_values}")

        # Obtém o valor de V
        valor = self.model.get_var_by_name('V')
        print(f"CURRENT_V: {valor}")

        V_value = current_solution[self.model.get_var_by_name('V')]
        print(f"CURRENT_V: {V_value}")

        print(f"Valores atuais de w: {current_w_values}")
        print(f"Valor atual de V: {V_value}")

        # Verifica as restrições SSD para cada cenário
        for t in range(self.n_scenar):
            # Calcula o retorno do portfolio no cenário t
            portfolio_return = sum(self.scenarios[t, i] * self.w_values[i] for i in range(self.n_assets))

            # Verifica se a restrição SSD é violada
            if V_value > (portfolio_return - self.benchmark[t]):
                print(f"⚠️ Restrição SSD violada para cenário {t}")

                # Adiciona a restrição lazy
                w_vars = [self.get_model().get_var_by_name(f'wl_{i}')
                          for i in range(self.n_assets)]
                scenario_return_expr = sum(self.scenarios[t, i] * w_vars[i]
                                           for i in range(self.n_assets))

                # Cria e adiciona a restrição lazy
                # ct = (self.get_model().get_var_by_name('V') <=
                #      scenario_return_expr - self.benchmark[t])
                # self.add(ct, 'L', 0)

                print(f"✅ Restrição lazy adicionada para cenário {t}")