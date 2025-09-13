import numpy as np
import pandas as pd
from .adstock import adstock_geometric

def identify_skewed_columns(df, threshold=1.0, exclude=None):
    if exclude is None: exclude = []
    num = df.select_dtypes(include=[np.number])
    skews = num.skew().abs().sort_values(ascending=False)
    return [c for c in skews.index if c not in exclude and skews[c] > threshold]

def log_transform(df, cols):
    for c in cols:
        df[c+'_log'] = np.log1p(df[c].fillna(0))
    return df

def make_adstock_columns(df, channels_with_rates):
    for col,rate in channels_with_rates.items():
        if col in df.columns:
            src = col+'_log' if (col+'_log') in df.columns else col
            df[col+'_ad'] = adstock_geometric(df[src].fillna(0), rate)
    return df

def prepare_model_df(df, adstock_rates, log_threshold=1.0):
    exclude = ['kpi_val_sales_mn','vol_sales_mn','wtd_dist_max','num_dist_max']
    skewed = identify_skewed_columns(df, log_threshold, exclude)
    df = log_transform(df, skewed)
    df = make_adstock_columns(df, adstock_rates)
    df['log_kpi'] = np.log(df['kpi_val_sales_mn']+1e-6)
    return df
