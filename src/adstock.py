import numpy as np

def adstock_geometric(series, rate=0.5):
    out = np.zeros(len(series))
    for i in range(len(series)):
        x = series.iloc[i] if hasattr(series, 'iloc') else series[i]
        out[i] = x if i==0 else x + rate*out[i-1]
    return out
