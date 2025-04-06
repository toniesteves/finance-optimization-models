#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:02
#  Updated: 06/04/2025 13:02

import numpy as np
# Finance-specific metrics (VaR, CVaR, Sharpe ratio)

def compute_var(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Compute Value-at-Risk (VaR) for a return series."""
    return np.percentile(returns, 100 * (1 - alpha))

def compute_cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Compute Conditional VaR (average loss beyond VaR)."""
    var = compute_var(returns, alpha)
    return returns[returns <= var].mean()

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)