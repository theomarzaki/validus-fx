import numpy as np
from scipy.stats import norm

def calculate_25delta_strikes(S, T, sigma_atm, sigma_25d_call, sigma_25d_put, r, q):
    """Calculate strikes for 25-delta options"""
    
    # For 25-delta call
    d1_call = norm.ppf(0.25 * np.exp(q * T))
    K_call = S * np.exp((r - q - 0.5 * sigma_25d_call**2) * T - d1_call * sigma_25d_call * np.sqrt(T))
    
    # For 25-delta put
    d1_put = -norm.ppf(0.25 * np.exp(q * T))
    K_put = S * np.exp((r - q - 0.5 * sigma_25d_put**2) * T + d1_put * sigma_25d_put * np.sqrt(T))
    
    return K_call, K_put
