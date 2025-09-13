
"""
Robustness pipeline for MMM using Ridge (tuning + CV metrics + contributions + ROI + reallocation simulation).
Saves outputs to: outputs/robustness
 - ridge_alpha.json
 - ridge_cv_mse.csv
 - ridge_cv_full_results.csv
 - ridge_coefs_tuned.csv
 - contrib_full.csv
 - roi_clean.csv
 - reallocation_simulation.json
 - optional: holdout_metrics.json
    @author: Akhand Pratap Singh
    @date: 2025-09-12
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# --- project imports: assume repo root contains src/ ---
import sys
proj = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(proj))

from src.utils import read_excel, ensure_dir
from src.preprocessing import prepare_model_df
from src.modeling import fit_with_collinearity_check
from src.attribution import compute_contributions

# ---------------- Configuration ----------------
DATA_PATH = proj / 'data' / 'Case_Study_Input_File.xlsx'
OUT_DIR = proj / 'outputs' / 'robustness'
ensure_dir(OUT_DIR)

ALPHAS = np.logspace(-4, 4, 50)
N_SPLITS = 5
USE_TIMESERIES_CV = True
N_JOBS = -1
USE_TUNED_RIDGE = True

# Optional holdout evaluation (time-based 80/20). Set False to skip.
DO_HOLDOUT = True
HOLDOUT_RATIO = 0.2

# ---------------- Helpers ----------------
def safe_to_csv(df, path, **kwargs):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=kwargs.pop('index', False), encoding=kwargs.pop('encoding', 'utf-8-sig'), **kwargs)
        print("Saved:", path)
    except Exception as e:
        print("Warning: could not save", path, ":", e)

def read_data(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data not found: {p}")
    if p.suffix.lower() in ('.xls', '.xlsx'):
        return read_excel(str(p))
    return pd.read_csv(str(p))

# ---------------- Load data ----------------
print("Loading data:", DATA_PATH)
df = read_data(DATA_PATH)
print("Data loaded. Rows:", len(df), "Cols:", len(df.columns))

# ---------------- Dist index ----------------
dist_cols = [c for c in ['wtd_dist_max', 'num_dist_max'] if c in df.columns]
if dist_cols:
    normed = []
    for c in dist_cols:
        col = df[c].astype(float)
        if col.max() == col.min():
            norm = col - col.min()
        else:
            norm = (col - col.min()) / (col.max() - col.min())
        normed.append(norm)
    df['dist_index'] = pd.concat(normed, axis=1).mean(axis=1)
else:
    df['dist_index'] = 0.0
print("Created dist_index from:", dist_cols)

# ---------------- Prepare model df ----------------
adstock_rates = {
    'tv_total_spends_mn': 0.6, 'newspaper_spends_mn': 0.25, 'google_display_spends_mn': 0.45,
    'youtube_spends_mn': 0.4, 'fb_ig_spends_mn': 0.4, 'whatsapp_spends_mn': 0.3,
    'jio_spends_mn': 0.25, 'hotstar_spends_mn': 0.35, 'ventes_avenue_spends_mn': 0.2,
    'trade_promo_spends_mn': 0.3, 'consumer_promo_spends_mn': 0.2
}
print("Preparing model df with adstock rates...")
model_df = prepare_model_df(df, adstock_rates)
print("Model df prepared. Rows:", len(model_df))

# ---------------- Predictors ----------------
predictors = [c for c in model_df.columns if c.endswith('_ad')]
predictors = ['dist_index'] + predictors
print("Predictors:", predictors)

# ---------------- Collinearity check (VIF) ----------------
df_m = model_df.dropna(subset=['log_kpi']).copy()
print("Dropped NA from log_kpi. Rows remaining:", len(df_m))

fit_res = fit_with_collinearity_check(df_m, predictors, target='log_kpi', vif_threshold=5.0)
vif_df = fit_res.get('vif')
if vif_df is not None:
    safe_to_csv(vif_df, OUT_DIR / 'vif.csv')
    print("Saved VIF to:", OUT_DIR / 'vif.csv')
else:
    print("No VIF returned by fit_with_collinearity_check.")

# ---------------- Prepare X,y for tuning ----------------
X_df = df_m[predictors].astype(float).copy()
y = df_m['log_kpi'].astype(float).values
X = X_df.values
print("Prepared X matrix shape:", X.shape)

# ---------------- Ridge tuning ----------------
if USE_TUNED_RIDGE:
    print("Tuning Ridge alpha using GridSearchCV...")
    cv = TimeSeriesSplit(n_splits=N_SPLITS) if USE_TIMESERIES_CV else None

    ridge = Ridge()
    param_grid = {'alpha': ALPHAS}

    grid = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        return_train_score=True,
        n_jobs=N_JOBS,
        verbose=1
    )
    grid.fit(X, y)

    best_alpha = float(grid.best_params_['alpha'])
    print("Best alpha found:", best_alpha)

    # --- CV metrics for best alpha ---
    cv_res = pd.DataFrame(grid.cv_results_)
    # get mean_test_score for best alpha (neg MSE)
    try:
        best_mask = cv_res['param_alpha'].astype(float) == best_alpha
        best_neg_mse = float(cv_res.loc[best_mask, 'mean_test_score'].iloc[0])
        best_mse = -best_neg_mse
        best_rmse = float(np.sqrt(best_mse))
        print(f"Best CV MSE (mean across folds): {best_mse:.6f} -> RMSE: {best_rmse:.6f}")
    except Exception as e:
        print("Warning: could not extract best CV MSE:", e)
        best_mse = None
        best_rmse = None

    # CV R^2 summary (cross_val_score on r2)
    try:
        ridge_best = Ridge(alpha=best_alpha)
        r2_scores = cross_val_score(ridge_best, X, y, cv=cv, scoring='r2', n_jobs=N_JOBS)
        print(f"CV R² (mean ± std) for best alpha: {r2_scores.mean():.4f} ± {r2_scores.std():.4f}")
    except Exception as e:
        print("Warning: could not compute CV R² via cross_val_score:", e)
        r2_scores = None

    # fit final ridge on full data
    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(X, y)

    # in-sample (optimistic) metrics on full data
    y_in_pred = ridge_final.predict(X)
    in_r2 = float(r2_score(y, y_in_pred))
    in_rmse = float(np.sqrt(mean_squared_error(y, y_in_pred)))
    print(f"In-sample (full) R²: {in_r2:.4f}, RMSE: {in_rmse:.6f}")

    # Save tuned results and cv outputs
    with open(OUT_DIR / "ridge_alpha.json", "w") as f:
        json.dump({"best_alpha": best_alpha, "cv_rmse": best_rmse, "cv_mse": best_mse}, f, indent=2)
    print("Saved tuned alpha to:", OUT_DIR / "ridge_alpha.json")

    try:
        cv_df = pd.DataFrame({
            'alpha': cv_res['param_alpha'].astype(float),
            'cv_mse': -cv_res['mean_test_score'].astype(float),
            'std_test_score': cv_res['std_test_score'].astype(float)
        }).sort_values('alpha')
        safe_to_csv(cv_df, OUT_DIR / "ridge_cv_mse.csv")
        safe_to_csv(cv_res, OUT_DIR / "ridge_cv_full_results.csv")
        print("Saved CV results to:", OUT_DIR / "ridge_cv_mse.csv")
    except Exception as e:
        print("Warning: could not save CV results:", e)

    coef_df = pd.DataFrame({'feature': predictors, 'coef': ridge_final.coef_.tolist()})
    safe_to_csv(coef_df, OUT_DIR / 'ridge_coefs_tuned.csv')
    print("Saved tuned ridge coefficients to:", OUT_DIR / 'ridge_coefs_tuned.csv')

    intercept = float(ridge_final.intercept_)
    final_coefs = coef_df.copy()
    model_type = 'ridge_tuned'
else:
    # fallback to fit_res (if not tuning)
    if fit_res.get('model_type') == 'ols' and not USE_TUNED_RIDGE:
        print("Using OLS model returned by fit_with_collinearity_check")
        model = fit_res['model']
        intercept = float(model.intercept_)
        final_coefs = pd.DataFrame({'feature': X_df.columns, 'coef': model.coef_})
        model_type = 'ols'
    else:
        coef_df = fit_res.get('coef_df')
        intercept = fit_res.get('intercept', 0.0)
        if coef_df is not None:
            final_coefs = coef_df.copy()
            final_coefs.columns = ['feature', 'coef'] if 'feature' in final_coefs.columns else final_coefs.columns
            model_type = fit_res.get('model_type', 'unknown')
        else:
            raise RuntimeError("No model available and USE_TUNED_RIDGE is False. Aborting.")

print("Model type used for contributions:", model_type)

# ---------------- Contributions ----------------
if model_type == 'ols' and not USE_TUNED_RIDGE:
    model = fit_res['model']
    X_for_contrib = fit_res['X']
    contrib_df = compute_contributions(model, X_for_contrib, [p for p in predictors if p.endswith('_ad')])
else:
    if 'feature' not in final_coefs.columns:
        final_coefs = final_coefs.rename(columns={final_coefs.columns[0]: 'feature', final_coefs.columns[1]: 'coef'})
    coef_map = dict(zip(final_coefs['feature'], final_coefs['coef']))

    preds = X_df.copy()
    coefs_arr = np.array([float(coef_map.get(p, 0.0)) for p in predictors])
    lin_pred = float(intercept) + (preds.values * coefs_arr).sum(axis=1)
    pred_full = np.exp(lin_pred)

    contrib_map = {}
    for i, p in enumerate(predictors):
        X0 = preds.copy()
        X0[p] = 0.0
        lin_no = float(intercept) + (X0.values * coefs_arr).sum(axis=1)
        pred_no = np.exp(lin_no)
        contrib_map[p] = float((pred_full - pred_no).sum())

    contrib_df = pd.DataFrame([{'channel': k, 'contribution_kpi': v} for k, v in contrib_map.items()])

contrib_df = contrib_df.rename(columns={'contribution_kpi': 'total_contribution_kpi_mn'})
safe_to_csv(contrib_df, OUT_DIR / 'contrib_full.csv')

# ---------------- Spend map & ROI ----------------
spend_map = {}
for c in df.columns:
    key = c.strip()
    if 'spend' in key or 'spends' in key:
        try:
            spend_map[key] = float(df[c].sum())
        except Exception:
            spend_map[key] = 0.0

def map_spend(ch):
    ch_base = ch.replace('_ad', '')
    if ch_base in spend_map:
        return spend_map[ch_base]
    for k in spend_map:
        if ch_base in k:
            return spend_map[k]
    return 0.0

contrib_df['total_spend_mn'] = contrib_df['channel'].apply(map_spend)
contrib_df['roi_kpi_per_mn_spend'] = contrib_df.apply(
    lambda r: r['total_contribution_kpi_mn'] / r['total_spend_mn'] if r['total_spend_mn'] > 0 else float('nan'),
    axis=1
)

roi_clean = contrib_df[contrib_df['total_spend_mn'] > 0].copy().reset_index(drop=True)
safe_to_csv(roi_clean, OUT_DIR / 'roi_clean.csv')
print("Saved cleaned ROI to:", OUT_DIR / 'roi_clean.csv')

# ---------------- Reallocation simulation ----------------
donors = [d for d in ['tv_total_spends_mn_ad', 'google_display_spends_mn_ad'] if d in contrib_df['channel'].values]
recipients = [r for r in ['hotstar_spends_mn_ad', 'whatsapp_spends_mn_ad'] if r in contrib_df['channel'].values]

sim_results = []
for pct in [0.05, 0.10]:
    total_realloc = sum(map_spend(d) * pct for d in donors)
    per_rec = total_realloc / max(1, len(recipients))
    uplift = 0.0
    details = []
    for r in recipients:
        row = roi_clean[roi_clean['channel'] == r]
        if not row.empty and not pd.isna(row.iloc[0]['roi_kpi_per_mn_spend']):
            roi_val = float(row.iloc[0]['roi_kpi_per_mn_spend'])
            expected = per_rec * roi_val
            details.append({'recipient': r, 'added_spend_mn': per_rec, 'roi': roi_val, 'expected_kpi_uplift': expected})
            uplift += expected
        else:
            details.append({'recipient': r, 'added_spend_mn': per_rec, 'roi': None, 'expected_kpi_uplift': None})
    conservative_uplift = uplift * 0.5
    sim_results.append({'pct_realloc': pct, 'total_realloc_mn': total_realloc, 'gross_uplift_kpi': uplift, 'conservative_uplift_kpi': conservative_uplift, 'details': details})

with open(OUT_DIR / 'reallocation_simulation.json', 'w') as f:
    json.dump(sim_results, f, indent=2)
print("Saved reallocation simulation to:", OUT_DIR / 'reallocation_simulation.json')

# ---------------- Optional holdout evaluation ----------------
if DO_HOLDOUT:
    print("Running optional time-based holdout evaluation (80/20)...")
    n = len(X_df)
    split_idx = int(n * (1 - HOLDOUT_RATIO))
    X_df_full = X_df.reset_index(drop=True)
    X_train_ts = X_df_full.iloc[:split_idx].values
    y_train_ts = y[:split_idx]
    X_test_ts = X_df_full.iloc[split_idx:].values
    y_test_ts = y[split_idx:]

    ridge_holdout = Ridge(alpha=best_alpha)
    ridge_holdout.fit(X_train_ts, y_train_ts)
    y_test_pred = ridge_holdout.predict(X_test_ts)
    holdout_r2 = float(r2_score(y_test_ts, y_test_pred))
    holdout_rmse = float(np.sqrt(mean_squared_error(y_test_ts, y_test_pred)))
    holdout_metrics = {'holdout_r2': holdout_r2, 'holdout_rmse': holdout_rmse}
    with open(OUT_DIR / 'holdout_metrics.json', 'w') as f:
        json.dump(holdout_metrics, f, indent=2)
    print(f"Holdout (time-split) R²: {holdout_r2:.4f}, RMSE: {holdout_rmse:.6f}")
else:
    print("Holdout evaluation skipped (DO_HOLDOUT=False).")

# ---------------- Final prints ----------------
print("\n--- Pipeline finished ---")
print("Outputs written to:", OUT_DIR)
print("\nTop ROI (clean):")
if not roi_clean.empty:
    print(roi_clean.sort_values('roi_kpi_per_mn_spend', ascending=False).head().to_string(index=False))
else:
    print("No ROI rows (check spend_map and contribs).")

print("\nSimulation results (sample):")
print(json.dumps(sim_results, indent=2))
