#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 07/04/2025 23:01
#  Updated: 07/04/2025 23:01

from models.estimators.cvar_model import CVaR
from models.estimators.mad_model import MAD
from models.mip_lazy_model import MIPLazyDemo
from models.estimators.mvo_model import MVO
from models.stochastic.ssd_model import SSD

# Mapping of portfolio types to classes
PORTFOLIO_CLASSES = {
    'mip':  MIPLazyDemo,
    'mvo':  MVO,
    'mad':  MAD,
    'ssd':  SSD,
    'cvar': CVaR
}