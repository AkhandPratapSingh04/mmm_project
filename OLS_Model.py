# run.py
import os,sys,io,json
from pathlib import Path
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
import pandas as pd
import numpy as np
from src.utils import read_excel, ensure_dir
from src.eda import run_eda
from src.preprocessing import prepare_model_df
from src.modeling import fit_with_collinearity_check, fit_ols_log_target
from src.attribution import compute_contributions
from src.gen_ai import build_narrative, pretty_print_narrative
from src.diagnostics import run_diagnostics



# wrap stdout/stderr to UTF-8 to avoid Windows encoding errors during prints
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    # not critical; continue if we cannot wrap (some environments)
    pass



def main():
    # --------- configuration (edit if needed) ----------
    task = "pipeline"   # change to "eda" to default run EDA

    # Dynamically set project root and paths
    proj = Path(__file__).resolve().parents[0]
    DATA_PATH = proj / "data" / "Case_Study_Input_File.xlsx"
    OUT_DIR = proj / "outputs"

    VIF_THRESHOLD = 5.0   # threshold to switch to Ridge
    USE_DIST_INDEX = True  # prefer a single combined distribution variable
    # ----------------------------------------------------

    # Ensure out dir
    ensure_dir(str(OUT_DIR))

    print("Loading data from:", DATA_PATH)
    df = read_excel(DATA_PATH)
    print("Rows:", len(df), "Cols:", len(df.columns))

    if task == 'eda':
        print("Running EDA...")
        run_eda(df, str(OUT_DIR))
        print('EDA done, saved to', OUT_DIR)
        return

    if task != 'pipeline':
        raise ValueError("task must be 'pipeline' or 'eda'")

    # -------------------------
    # 1) PREP: adstock + log target
    # -------------------------
    adstock_rates = {
        'tv_total_spends_mn':0.6,'newspaper_spends_mn':0.25,'google_display_spends_mn':0.45,
        'youtube_spends_mn':0.4,'fb_ig_spends_mn':0.4,'whatsapp_spends_mn':0.3,
        'jio_spends_mn':0.25,'hotstar_spends_mn':0.35,'ventes_avenue_spends_mn':0.2,
        'trade_promo_spends_mn':0.3,'consumer_promo_spends_mn':0.2
    }
    model_df = prepare_model_df(df, adstock_rates)
    print("Prepared model_df. Rows:", len(model_df))

    # -------------------------
    # 2) predictors: adstocked channels + distribution controls
    # -------------------------
    ad_predictors = [c for c in model_df.columns if c.endswith('_ad')]
    if USE_DIST_INDEX:
        # create dist_index if not present (safe fallback to normalized combination)
        if 'dist_index' not in model_df.columns:
            dist_cols = [c for c in ['wtd_dist_max','num_dist_max'] if c in model_df.columns]
            if dist_cols:
                normed = []
                for c in dist_cols:
                    col = model_df[c].astype(float)
                    if col.max() == col.min():
                        norm = col - col.min()
                    else:
                        norm = (col - col.min()) / (col.max() - col.min())
                    normed.append(norm)
                model_df['dist_index'] = pd.concat(normed, axis=1).mean(axis=1)
            else:
                model_df['dist_index'] = 0.0
        predictors = ['dist_index'] + ad_predictors
    else:
        predictors = [c for c in ['wtd_dist_max','num_dist_max'] if c in model_df.columns] + ad_predictors

    print("Using predictors:", predictors)

    # -------------------------
    # 3) Fit model with collinearity check (auto OLS or Ridge)
    # -------------------------
    df_model_ready = model_df.dropna(subset=['log_kpi']).copy()
    fit_res = fit_with_collinearity_check(df_model_ready, predictors, target='log_kpi', vif_threshold=VIF_THRESHOLD)

    vif_df = fit_res.get('vif')
    model_type = fit_res.get('model_type', 'unknown')
    print("Model type selected by fit_with_collinearity_check:", model_type)

    # Save VIF (utf-8-sig so Excel on Windows reads BOM)
    try:
        if vif_df is not None:
            vif_df.to_csv(OUT_DIR / 'vif.csv', index=False, encoding='utf-8-sig')
        else:
            pd.DataFrame().to_csv(OUT_DIR / 'vif.csv', index=False, encoding='utf-8-sig')
    except Exception as e:
        print("Warning: could not save vif.csv:", e)

    # -------------------------
    # 4) Attribution depending on model type
    # -------------------------
    contrib_df = None
    if model_type == 'ols':
        model = fit_res['model']
        X = fit_res['X']
        y = fit_res['y']
        contrib_df = compute_contributions(model, X, [p for p in predictors if p.endswith('_ad')])
        # save OLS coefs for inspection
        try:
            ols_params = model.params
            ols_df = pd.DataFrame({'feature': ols_params.index, 'coef': ols_params.values})
            ols_df.to_csv(OUT_DIR / 'ols_coefs.csv', index=False, encoding='utf-8-sig')
        except Exception as e:
            print("Warning: could not save ols_coefs.csv:", e)
    else:
        # Ridge path
        coef_df = fit_res.get('coef_df')
        intercept = float(fit_res.get('intercept', 0.0))

        X_pred = df_model_ready[predictors].astype(float).copy()
        coefs = []
        for p in predictors:
            try:
                coefs.append(float(coef_df.loc[p, 'coef']) if p in coef_df.index else 0.0)
            except Exception:
                coefs.append(0.0)
        coefs = np.array(coefs)
        lin_pred = intercept + (X_pred.values * coefs).sum(axis=1)
        pred_full = np.exp(lin_pred)

        contrib_map = {}
        for i, p in enumerate(predictors):
            X_zero = X_pred.copy()
            X_zero[p] = 0
            lin_no = intercept + (X_zero.values * coefs).sum(axis=1)
            pred_no = np.exp(lin_no)
            contrib_map[p] = float((pred_full - pred_no).sum())
        contrib_df = pd.DataFrame([{'channel': k, 'contribution_kpi': v} for k, v in contrib_map.items()])

        # save ridge coefs (if present)
        try:
            if coef_df is not None:
                coef_df.to_csv(OUT_DIR / 'ridge_coefs.csv', index=False, encoding='utf-8-sig')
        except Exception as e:
            print("Warning: could not save ridge_coefs.csv:", e)

        # write intercept info to model_summary
        try:
            with open(OUT_DIR / 'model_summary.txt', 'w', encoding='utf-8') as f:
                f.write(f"Model type: ridge\nIntercept: {intercept}\n")
        except Exception as e:
            print("Warning: could not save model_summary.txt for ridge:", e)

    # Normalize column name
    if contrib_df is None:
        raise RuntimeError("Contributions DataFrame was not created.")
    contrib_df = contrib_df.rename(columns={'contribution_kpi': 'total_contribution_kpi_mn',
                                            'contribution': 'total_contribution_kpi_mn'})

    # -------------------------
    # 5) Prepare spend map and save it
    # -------------------------
    spend_map = {}
    for c in df.columns:
        key = str(c).strip()
        if 'spend' in key or 'spends' in key:
            try:
                spend_map[key] = float(df[c].sum())
            except Exception:
                spend_map[key] = 0.0

    # Save spend map for traceability
    try:
        spend_map_df = pd.DataFrame([{'channel_raw': k, 'total_spend_mn': v} for k, v in spend_map.items()])
        spend_map_df.to_csv(OUT_DIR / 'spend_map.csv', index=False, encoding='utf-8-sig')
    except Exception as e:
        print("Warning: could not save spend_map.csv:", e)

    # -------------------------
    # 6) map spend to contribution channels and compute ROI
    # -------------------------
    def map_spend_for_channel(ch):
        ch_base = ch.replace('_ad', '')
        # exact matches
        if ch_base in spend_map:
            return spend_map[ch_base]
        # common variants
        for suffix in ('_spends_mn','_spend_mn','_spends','_spend'):
            key = ch_base + suffix
            if key in spend_map:
                return spend_map[key]
        # substring match fallback
        for k in spend_map:
            if ch_base in k or k in ch_base:
                return spend_map[k]
        return 0.0

    contrib_df['total_spend_mn'] = contrib_df['channel'].apply(map_spend_for_channel)
    contrib_df['roi_kpi_per_mn_spend'] = contrib_df.apply(
        lambda r: (r['total_contribution_kpi_mn'] / r['total_spend_mn']) if r['total_spend_mn'] > 0 else float('nan'),
        axis=1
    )

    # 7) Save contrib & roi (utf-8-sig)
    try:
        contrib_df.to_csv(OUT_DIR / 'contrib.csv', index=False, encoding='utf-8-sig')
        contrib_df.to_csv(OUT_DIR / 'roi.csv', index=False, encoding='utf-8-sig')
        print("Saved contrib.csv and roi.csv to", OUT_DIR)
    except Exception as e:
        print("Warning: could not save contrib/roi csvs:", e)

    # If OLS model present and model_summary not yet saved, save it
    try:
        if model_type == 'ols':
            # prefer to save the statsmodels summary text
            with open(OUT_DIR / 'model_summary.txt', 'w', encoding='utf-8') as f:
                f.write(fit_res['model'].summary().as_text())
    except Exception as e:
        print("Warning: could not save model_summary.txt:", e)

    # -------------------------
    # Diagnostics (fit OLS for diagnostics if needed)
    # -------------------------
    try:
        if model_type == 'ols':
            ols_model_for_diag = fit_res['model']
            ols_X_for_diag = fit_res['X']
            ols_y_for_diag = fit_res['y']
        else:
            ols_model_for_diag, ols_X_for_diag, ols_y_for_diag = fit_ols_log_target(df_model_ready, predictors)

        roi_df = pd.read_csv(OUT_DIR / 'roi.csv', encoding='utf-8-sig')
        diag_out_dir = OUT_DIR / 'diagnostics'
        ensure_dir(str(diag_out_dir))

        results, rolling_coef_df, flagged = run_diagnostics(
            ols_model_for_diag,
            ols_X_for_diag,
            ols_y_for_diag,
            df_model_ready,
            predictors,
            roi_df,
            fit_ols_log_target,
            out_dir=str(diag_out_dir)
        )

        # save rolling coeffs if returned
        try:
            if rolling_coef_df is not None:
                # rolling_coef_df may be a DataFrame or None
                rolling_coeffs_path = Path(diag_out_dir) / 'rolling_coeffs.csv'
                rolling_coef_df.to_csv(rolling_coeffs_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print("Warning: could not save rolling_coeffs.csv:", e)

        # if run_diagnostics returned a flagged DataFrame, save it
        try:
            if flagged is not None:
                flagged_path = Path(diag_out_dir) / 'suspicious_rois.csv'
                if hasattr(flagged, 'to_csv'):
                    flagged.to_csv(flagged_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            print("Warning: could not save suspicious_rois.csv:", e)

        print("Diagnostics finished. See", diag_out_dir)
    except Exception as e:
        print("Warning: diagnostics failed:", str(e))

    # -------------------------
    # Narrative
    # -------------------------
    try:
        narr = build_narrative(
            df_path=str(OUT_DIR / 'roi.csv'),
            vif_path=str(OUT_DIR / 'vif.csv'),
            model_summary_path=str(OUT_DIR / 'model_summary.txt'),
            stability_path=str(OUT_DIR / 'diagnostics' / 'coef_stability_summary.csv')
        )
        text = pretty_print_narrative(narr)
        print("\n=== Auto-generated summary ===\n")
        print(text)
        with open(OUT_DIR / 'auto_summary.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Saved auto_summary.txt")
    except Exception as e:
        print("Warning: could not generate narrative summary:", str(e))

    print("\nPipeline finished. Outputs saved to:", OUT_DIR)

if __name__ == '__main__':
    main()
