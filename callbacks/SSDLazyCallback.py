#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 01:45
#  Updated: 06/04/2025 01:45

from cplex.callbacks import LazyConstraintCallback
from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
class SSDLazyCallback(LazyConstraintCallback, ConstraintCallbackMixin):
    def __init__(self, env):
        super().__init__(env)
        self.x = None
        self.y = None
        self.n_calls = 0

    def __call__(self):
        self.n_calls += 1
        try:
            # Obtém os valores atuais da solução candidata
            x_val = self.get_values(self.x.index)
            y_val = self.get_values(self.y.index)

            print(f"\n--- Chamada de Callback #{self.n_calls} ---")
            print(f"Solução candidata: x = {x_val}, y = {y_val}")
            print(f"Avaliando se x + 3y > 10: {x_val} + 3*{y_val} = {x_val + 3 * y_val}")

            # Condição para adicionar restrição
            if x_val + 3 * y_val > 10:
                print("⚠️ Solução viola a condição x + 3y ≤ 10")
                print("Adicionando restrição lazy: x + y ≤ 8")

                # Adiciona a restrição lazy
                cpx_lhs, sense, cpx_rhs = self.linear_ct_to_cplex(self.x + self.y <= 8)
                self.add(cpx_lhs, sense, cpx_rhs)

                print("✅ Restrição lazy adicionada com sucesso!")
            else:
                print("✅ Solução válida, nenhuma restrição adicionada")

        except Exception as e:
            print(f"Erro no callback: {str(e)}")
            raise
