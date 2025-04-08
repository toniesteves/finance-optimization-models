#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 07/04/2025 15:31
#  Updated: 07/04/2025 15:31

import cplex
from docplex.mp.model import Model
from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
class MIPLazyCallback(LazyConstraintCallback, ConstraintCallbackMixin):
    def __init__(self, env):
        super().__init__(env)
        self.x = None
        self.y = None
        self.n_calls = 0
        self.verbose = None

    def __call__(self):
        self.n_calls += 1
        try:
            # Obtém os valores atuais da solução candidata
            x_val = self.get_values(self.x.index)
            y_val = self.get_values(self.y.index)

            if self.verbose:
                print(f"\n\t--- Chamada de Callback #{self.n_calls} ---")
                print(f"\tSolução candidata: x = {x_val}, y = {y_val}")
                print(f"\tAvaliando se x + 3y > 10: {x_val} + 3*{y_val} = {x_val + 3 * y_val}")

            # Condição para adicionar restrição
            if x_val + 3 * y_val > 10:
                print("\t⚠️ Solução viola a condição x + 3y ≤ 10")
                print("\tAdicionando restrição lazy: x + y ≤ 8")

                # Adiciona a restrição lazy
                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(self.x + self.y <= 8)
                self.add(cpx_lhs, sense, cpx_rhs)

                print("\t✅ Restrição lazy adicionada com sucesso!")
            else:
                print("\t✅ Solução válida, nenhuma restrição adicionada")

        except Exception as e:
            print(f"\tErro no callback: {str(e)}")
            raise