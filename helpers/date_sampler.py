import numpy as np

def sampleAtKeyDates(t, S, v, target_dates):
    """ 
        Get Spot and Vol paths that are closest to the key dates
    """
    sampled_S = np.zeros((len(target_dates), S.shape[1]))
    sampled_v = np.zeros((len(target_dates), v.shape[1]))
        
    for index, target_t in enumerate(target_dates):
        idx = np.argmin(np.abs(t - target_t))
        sampled_S[index, :] = S[idx, :]
        sampled_v[index, :] = v[idx, :]
    
    return sampled_S, sampled_v

