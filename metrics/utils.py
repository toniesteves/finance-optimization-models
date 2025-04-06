#  Toni Esteves Copyright (c) 2025.
#  Project Name: finance-optimization-models
#  Author: Toni Esteves <toni.esteves@gmail.com.br>
#  Created: 06/04/2025 13:02
#  Updated: 06/04/2025 13:02

import  numpy as np
# Helper functions (e.g., data validation)

def validate_returns(returns: np.ndarray) -> None:
    """Check for NaN/inf values."""
    if np.any(np.isnan(returns)):
        raise ValueError("Returns contain NaN values!")
    if np.any(np.isinf(returns)):
        raise ValueError("Returns contain infinite values!")