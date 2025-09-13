# src/diagnostics.py
import os
import traceback
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error

def train_test_split_time(df, test_size=0.2):
    """Split dataset into train/test keeping chronological order."""
    n = len(df)
    split_idx = int(n * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test

def compute_fit_metrics(model, X, y, label="Train"):
    """Compute R² and RMSE for given dataset using provided model and exog (X)."""
    # model.predict must accept X in the same format as it was fit on
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    return {"label": label, "R2": r2, "RMSE": rmse}

def rolling_coeff_stability(model_func, df, predictors, window=0.8, step=0.1, target='log_kpi'):
    """
    Fit model on rolling windows and track coefficients.
    model_func: function(df, predictors[, target]) -> (model, X, y)
    This function will attempt to call model_func with or without the target argument.
    """
    n = len(df)
    win_size = int(n * window)
    step_size = max(1, int(n * step))
    coefs = []

    # if dataset too small to make at least one window, return empty df
    if win_size >= n:
        return pd.DataFrame(coefs)

    for start in range(0, n - win_size + 1, step_size):
        end = start + win_size
        dwin = df.iloc[start:end]
        # try calling model_func with different signatures
        try:
            model, X, y = model_func(dwin, predictors, target)
        except TypeError:
            # maybe model_func expects (df, predictors) only
            model, X, y = model_func(dwin, predictors)

        # model.params may be a Series (statsmodels) or coef_ (sklearn) — handle both
        try:
            params = {}
            if hasattr(model, 'params'):  # statsmodels
                params = model.params.to_dict()
            elif hasattr(model, 'coef'):   # sklearn estimator
                cols = X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
                for i, c in enumerate(cols):
                    params[str(c)] = float(model.coef_[i]) if len(model.coef_) > i else None
            else:
                params = {'unknown_model_coef': None}
        except Exception:
            params = {'error_extracting_params': True}

        params['window'] = f"{start}-{end}"
        coefs.append(params)
    return pd.DataFrame(coefs)

def flag_suspicious_rois(roi_df, high_thresh=50, low_thresh=-10):
    """
    Flag suspicious ROI values that may indicate over/underfit.
    """
    if roi_df is None or roi_df.empty:
        return pd.DataFrame(columns=['channel', 'roi_kpi_per_mn_spend'])
    flagged = roi_df[
        (roi_df['roi_kpi_per_mn_spend'] > high_thresh) |
        (roi_df['roi_kpi_per_mn_spend'] < low_thresh)
    ]
    return flagged

def run_diagnostics(model, X, y, df_full, predictors, roi_df, model_func, out_dir="outputs/diagnostics"):
    """
    Run diagnostics and write outputs to out_dir.
    - model, X, y: the model fitted on TRAIN and the X,y used to train it (preferred)
    - df_full: the full model dataframe (used to create train/test splits)
    - model_func: callable to build (model, X, y) from a dataframe and predictor list
      (this is used to construct X_test and to perform rolling windows)
    Returns: results_list, coef_df, flagged_df
    """
    os.makedirs(out_dir, exist_ok=True)
    traceback_path = os.path.join(out_dir, "diagnostics_traceback.txt")

    try:
        results = []

        # 1. Train/Test Split Performance
        # Use chronological split from df_full (not refit on test)
        train_df, test_df = train_test_split_time(df_full)

        # Fit model on the TRAIN split using model_func to ensure consistent preprocessing
        try:
            m_train, X_train, y_train = model_func(train_df, predictors, 'log_kpi')
        except TypeError:
            m_train, X_train, y_train = model_func(train_df, predictors)

        # Build X_test and y_test from test_df using model_func, BUT DO NOT FIT on test
        try:
            _m_test_dummy, X_test, y_test = model_func(test_df, predictors, 'log_kpi')
        except TypeError:
            _m_test_dummy, X_test, y_test = model_func(test_df, predictors)

        # Now evaluate: train metrics from m_train (or provided model) and test metrics using m_train predictions on X_test
        # Prefer the provided model if it was fitted to TRAIN and provided by the caller (avoid double-fitting)
        # We'll use the provided 'model' argument for Train metrics if possible; otherwise use m_train.
        model_for_eval = model if model is not None else m_train

        fit_train = compute_fit_metrics(model_for_eval, X_train, y_train, label="Train")
        # Evaluate TEST by predicting with the TRAIN-fitted model on X_test
        # Ensure predict uses model_for_eval
        y_pred_test = model_for_eval.predict(X_test)
        fit_test = {"label": "Test", "R2": float(r2_score(y_test, y_pred_test)), "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred_test)))}

        results.extend([fit_train, fit_test])

        # 2. Rolling window coefficient stability (still useful)
        coef_df = rolling_coeff_stability(model_func, df_full, predictors, target='log_kpi')
        try:
            coef_df.to_csv(os.path.join(out_dir, "rolling_coeffs.csv"), index=False, encoding='utf-8-sig')
        except Exception as e:
            with open(traceback_path, 'a', encoding='utf-8') as tf:
                tf.write(f"Failed to save rolling_coeffs.csv: {e}\n")

        # 3. ROI sanity check
        flagged = flag_suspicious_rois(roi_df)
        try:
            flagged.to_csv(os.path.join(out_dir, "suspicious_rois.csv"), index=False, encoding='utf-8-sig')
        except Exception as e:
            with open(traceback_path, 'a', encoding='utf-8') as tf:
                tf.write(f"Failed to save suspicious_rois.csv: {e}\n")

        # 4. Save authoritative summary text (use UTF-8)
        summary_lines = []
        summary_lines.append("=== Model Diagnostics Summary ===")
        for r in results:
            summary_lines.append(f"{r['label']}: R²={r['R2']:.3f}, RMSE={r['RMSE']:.3f}")

        if flagged is not None and not flagged.empty:
            summary_lines.append("\n⚠ Suspicious ROI values detected:")
            for _, row in flagged.iterrows():
                summary_lines.append(f"  {row['channel']}: ROI={row['roi_kpi_per_mn_spend']:.2f}")
        else:
            summary_lines.append("\nNo suspicious ROI values flagged.")

        summary_text = "\n".join(summary_lines)
        summary_path = os.path.join(out_dir, "diagnostics_summary.txt")
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write(summary_text)

        # also print to console
        print(summary_text)

        return results, coef_df, flagged

    except Exception as exc:
        tb = traceback.format_exc()
        # write a traceback file to help debug
        try:
            with open(traceback_path, 'w', encoding='utf-8') as tf:
                tf.write(tb)
            print("Diagnostics failed; traceback written to", traceback_path)
        except Exception as e:
            # last resort: print to console
            print("Diagnostics failed and traceback could not be written:", e)
            print(tb)
        # return minimal so pipeline continues
        return None, None, None
