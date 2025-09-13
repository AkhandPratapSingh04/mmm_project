# src/modeling.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

# ---------- OLS ----------
def fit_ols_log_target(df, predictors, target='log_kpi'):
    """
    Fit OLS on df[target] ~ predictors.
    Returns: model (statsmodels OLSResults), X (DataFrame with const), y (Series)
    """
    X = df[predictors].astype(float).copy()
    X = sm.add_constant(X)
    y = df[target].astype(float)
    model = sm.OLS(y, X).fit(cov_type='HC3')
    return model, X, y

# ---------- VIF ----------
def compute_vif(X):
    """
    X: DataFrame including constant if present.
    Returns DataFrame with VIF per column.
    """
    vif_vals = []
    for i in range(X.shape[1]):
        try:
            v = variance_inflation_factor(X.values, i)
        except Exception:
            v = np.nan
        vif_vals.append(v)
    return pd.DataFrame({'feature': X.columns, 'VIF': vif_vals})

# ---------- Ridge (for collinearity) ----------
def fit_ridge_log_target(df, predictors, target='log_kpi', alphas=None, scale=True):
    """
    Fit Ridge with CV on df[target] ~ predictors.
    Returns:
      - ridge (fitted sklearn estimator),
      - coef_df (DataFrame: feature, coef),
      - scaler (StandardScaler or None), 
      - intercept (float)
    """
    if alphas is None:
        alphas = np.logspace(-3, 3, 13)

    X = df[predictors].astype(float).copy().values
    y = df[target].astype(float).values

    scaler = None
    if scale:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
    else:
        Xs = X

    ridge = RidgeCV(alphas=alphas, store_cv_values=False)
    ridge.fit(Xs, y)

    # Convert scaled coefficients back to original feature scale if scaled
    if scale:
        coef_scaled = ridge.coef_
        stds = scaler.scale_
        with np.errstate(divide='ignore', invalid='ignore'):
            coef_orig = coef_scaled / stds
        intercept = ridge.intercept_ - np.sum((scaler.mean_ / stds) * coef_scaled)
    else:
        coef_orig = ridge.coef_
        intercept = ridge.intercept_

    coef_df = pd.DataFrame({
        'feature': predictors,
        'coef': coef_orig
    })
    return ridge, coef_df, scaler, intercept

# ---------- Model chooser ----------
def fit_with_collinearity_check(df, predictors, target='log_kpi', vif_threshold=5.0, ridge_alphas=None):
    """
    Check VIF on predictors. If any predictor's VIF > vif_threshold, fit RidgeCV instead of OLS.
    Returns a dict with:
      - 'model_type': 'ols' or 'ridge'
      - 'model': fitted model object (statsmodels OLSResults or sklearn RidgeCV)
      - 'X': design matrix (DataFrame) used for OLS (if OLS; None for Ridge)
      - 'y': target Series (if OLS; None for Ridge)
      - 'coef_df': DataFrame of coefficients with feature names (for both models)
      - 'vif': DataFrame of VIFs computed on design matrix (with const for OLS)
      - 'intercept': model intercept (float)
    """
    # prepare X without constant for VIF check
    X_raw = df[predictors].astype(float).copy()
    X_for_vif = sm.add_constant(X_raw)
    vif_df = compute_vif(X_for_vif)

    high_vif = vif_df.loc[vif_df['feature'] != 'const', 'VIF'].max()
    if np.isfinite(high_vif) and high_vif > vif_threshold:
        # Fit Ridge
        ridge, coef_df, scaler, intercept = fit_ridge_log_target(
            df, predictors, target=target, alphas=ridge_alphas, scale=True
        )
        result = {
            'model_type': 'ridge',
            'model': ridge,
            'X': None,
            'y': None,
            'coef_df': coef_df,
            'vif': vif_df,
            'intercept': float(intercept)
        }
    else:
        # Fit OLS
        model, X, y = fit_ols_log_target(df, predictors, target=target)
        coef_df = pd.DataFrame({'feature': model.params.index, 'coef': model.params.values})
        result = {
            'model_type': 'ols',
            'model': model,
            'X': X,
            'y': y,
            'coef_df': coef_df,
            'vif': vif_df,
            'intercept': float(model.params.get('const', 0.0))
        }
    return result
