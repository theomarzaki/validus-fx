import numpy as np


def calculate_var(returns, confidence=0.95):
    return -np.percentile(returns, (1 - confidence) * 100)


def calculate_cvar(returns):
    var = calculate_var(returns)
    cvar = -returns[returns <= -var].mean()
    return cvar
