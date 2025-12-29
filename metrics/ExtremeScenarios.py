import numpy as np
from metrics.MultipleCapital import calculate_multiple_on_capital

def calculate_extreme_scenarios(result, n_extreme=100):
    
    irrs = result['IRR']
    usd_cfs = result['USD_CF']

    worst_idx = np.argsort(irrs)[:n_extreme]
    worst_irrs = irrs[worst_idx]
    worst_cfs = usd_cfs[worst_idx, :]
    
    best_idx = np.argsort(irrs)[-n_extreme:]
    best_irrs = irrs[best_idx]
    best_cfs = usd_cfs[best_idx, :]
    
    return {
        'Strategy Name': result['Strategy Name'],
        'worst_IRR': worst_irrs,
        'best_IRR': best_irrs,
        'worst_Multiple': calculate_multiple_on_capital(worst_cfs),
        'best_Multiple': calculate_multiple_on_capital(best_cfs)
    }