#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 07/04/2025 23:01
#  Updated: 07/04/2025 23:01

from models.mad_model import MADPortfolio
from models.mip_lazy_model import MIPLazyDemo
from models.mvo_model import MarkowitzPortfolio
from models.ssd_model import SSDPortfolio

# Mapping of portfolio types to classes
PORTFOLIO_CLASSES = {
    'mip': MIPLazyDemo,
    'mvo': MarkowitzPortfolio,
    'mad': MADPortfolio,
    'ssd': SSDPortfolio
}