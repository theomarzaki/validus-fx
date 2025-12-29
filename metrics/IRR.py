import numpy as np

def calculate_irr(cash_flows, times):
    n_paths = cash_flows.shape[0]
    irrs = np.zeros(n_paths)

    for j in range(n_paths):

        def f(r):
            return np.sum(cash_flows[j, :] / (1 + r) ** np.array(times))
        
        # Search for root between -0.5 and 1.0
        a, b = -0.5, 1.0
        for _ in range(50):
            m = (a + b) / 2
            if f(a) * f(m) <= 0:
                b = m
            else:
                a = m
        irrs[j] = (a + b) / 2

    return irrs[(irrs > -0.5) & (irrs < 2.0)]
