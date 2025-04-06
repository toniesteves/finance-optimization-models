#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:02
#  Updated: 06/04/2025 13:02

import  numpy as np
# General stats(mean, volatility, correlations)

def annualized_volatility(returns: np.ndarray) -> float:
    """Convert daily volatility to annualized."""
    return np.std(returns) * np.sqrt(252)

def max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown (peak-to-trough decline)."""
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    return (cumulative / peak - 1).min()