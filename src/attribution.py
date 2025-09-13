import numpy as np, pandas as pd

def compute_contributions(model,X,media_predictors):
    pred_full = np.exp(model.predict(X))
    contrib={}
    for ch in media_predictors:
        X0 = X.copy(); X0[ch]=0
        contrib[ch] = float(np.sum(pred_full - np.exp(model.predict(X0))))
    return pd.DataFrame([{'channel':k,'contribution_kpi':v} for k,v in contrib.items()])

def compute_roi(contrib_df,spend_map):
    contrib_df['total_spend_mn'] = contrib_df['channel'].map(spend_map).fillna(0)
    contrib_df['roi_kpi_per_mn'] = contrib_df.apply(lambda r: r['contribution_kpi']/r['total_spend_mn'] if r['total_spend_mn']>0 else np.nan,axis=1)
    return contrib_df
