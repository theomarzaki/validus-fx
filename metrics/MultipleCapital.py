import numpy as np

def calculate_multiple_on_capital(cash_flows):
    inflows = np.sum(cash_flows[:, 1:], axis=1)
    outflows = np.abs(cash_flows[:, 0]) 
    multiples = inflows / outflows
    return multiples
