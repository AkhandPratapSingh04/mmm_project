# src/gen_ai.py
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List

# --- existing helpers (read_roi_csv, summarize_top_bottom, generate_recommendations) ---
# If you already have read_roi_csv, summarize_top_bottom etc., keep them unchanged.
# Below we re-use read_roi_csv if present; otherwise redefine minimal fallback:

def read_roi_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Basic mappings if names differ:
    if 'total_contribution_kpi_mn' not in df.columns:
        # try common alternative column names
        for alt in ['contribution_kpi','contribution']:
            if alt in df.columns:
                df = df.rename(columns={alt:'total_contribution_kpi_mn'})
                break
    if 'total_spend_mn' not in df.columns:
        for alt in ['total_spend','spend','spend_mn']:
            if alt in df.columns:
                df = df.rename(columns={alt:'total_spend_mn'})
                break
    if 'roi_kpi_per_mn_spend' not in df.columns:
        for alt in ['roi_kpi_per_mn','roi','roi_kpi_per_mn_spend']:
            if alt in df.columns:
                df = df.rename(columns={alt:'roi_kpi_per_mn_spend'})
                break
    return df

def read_coef_stability(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(p)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    # Ensure required columns
    for col in ['feature','mean','std','cv','sign_changes','min','max']:
        if col not in df.columns:
            raise ValueError(f"stability file missing '{col}'")
    return df

def classify_stability(stab_df: pd.DataFrame, cv_threshold_high=1.0, cv_threshold_medium=0.5) -> Dict[str, List[str]]:
    """
    Classify features into 'high', 'medium', 'low' trust groups.
    - High trust: cv < medium threshold and sign_changes == 0
    - Medium trust: cv between medium and high threshold (and sign_changes==0)
    - Low trust: cv >= high threshold or sign_changes > 0
    Returns dict with lists of features for each bucket and a dict reasons.
    """
    high, med, low = [], [], []
    reasons = {}
    for _, r in stab_df.iterrows():
        feat = str(r['feature'])
        cv = float(r['cv']) if not pd.isna(r['cv']) else np.inf
        sc = int(r['sign_changes']) if not pd.isna(r['sign_changes']) else 0
        reason_list = []
        if sc > 0:
            reason_list.append(f"{sc} sign change(s)")
        if cv >= cv_threshold_high:
            reason_list.append(f"high CV={cv:.2f}")
        elif cv >= cv_threshold_medium:
            reason_list.append(f"moderate CV={cv:.2f}")
        else:
            reason_list.append(f"low CV={cv:.2f}")
        # classification
        if sc > 0 or cv >= cv_threshold_high:
            low.append(feat)
        elif cv >= cv_threshold_medium:
            med.append(feat)
        else:
            high.append(feat)
        reasons[feat] = "; ".join(reason_list)
    return {'high': high, 'medium': med, 'low': low, 'reasons': reasons}

# --- Updated build_narrative that includes stability ---
def build_narrative(df_path: str, vif_path: str = None, model_summary_path: str = None, stability_path: str = None) -> Dict[str, Any]:
    """
    Build narrative with ROI + optional VIF + optional model summary + optional coefficient stability.
    stability_path should point to outputs/diagnostics/coef_stability_summary.csv
    """
    df = read_roi_csv(df_path)
    # summarize top/bottom by ROI
    dfc = df.copy().replace([np.inf, -np.inf], np.nan)
    df_sorted = dfc.sort_values('roi_kpi_per_mn_spend', ascending=False)
    top = df_sorted.head(3).to_dict(orient='records')
    bottom = df_sorted.tail(3).to_dict(orient='records')
    summary_stats = {
        'num_channels': int(len(df)),
        'mean_roi': float(np.nanmean(dfc['roi_kpi_per_mn_spend'])),
        'median_roi': float(np.nanmedian(dfc['roi_kpi_per_mn_spend']))
    }

    narrative = {
        'headline': f"MMM Attribution Summary — {len(df)} channels analyzed",
        'top_channels': top,
        'bottom_channels': bottom,
        'metrics': summary_stats
        # 'caveats': [
        #     "This is an observational MMM summary (associations, not causal proof).",
        #     "Check multicollinearity (VIF) and model stability before major budget moves.",
        #     "Tune adstock/saturation and validate via holdouts or experiments."
        # ]
    }

    # VIF note
    if vif_path:
        try:
            vif_df = pd.read_csv(vif_path)
            high = vif_df[vif_df['VIF'] > 5]
            if not high.empty:
                narrative['caveats'].append("High multicollinearity detected for: " + ", ".join(high['feature'].astype(str).tolist()))
        except Exception:
            narrative['caveats'].append("VIF file could not be read or parsed.")

    # Model summary snippet
    if model_summary_path:
        try:
            txt = Path(model_summary_path).read_text(encoding='utf-8')[:3000]
            narrative['model_summary_snippet'] = txt
        except Exception:
            narrative['model_summary_snippet'] = None

    # Stability integration
    if stability_path:
        try:
            stab_df = read_coef_stability(stability_path)
            classified = classify_stability(stab_df)
            # produce human-friendly lists
            narrative['stability'] = {
                'high_trust': classified['high'],
                'medium_trust': classified['medium'],
                'low_trust': classified['low'],
                'reasons': classified['reasons']
            }
        except Exception as e:
            narrative['stability_error'] = f"Could not read stability file: {e}"

    return narrative

# --- Updated pretty printer ---
def pretty_print_narrative(narr: Dict[str, Any]) -> str:
    lines = [narr.get('headline','MMM Summary')]
    lines.append("\nTop channels:")
    for t in narr.get('top_channels', []):
        lines.append(f" - {t.get('channel','?')}: contribution={t.get('total_contribution_kpi_mn', t.get('contribution_kpi','?'))}, spend={t.get('total_spend_mn','?')}, ROI={t.get('roi_kpi_per_mn_spend', 'n/a')}")
    lines.append("\nBottom channels:")
    for b in narr.get('bottom_channels', []):
        lines.append(f" - {b.get('channel','?')}: contribution={b.get('total_contribution_kpi_mn', b.get('contribution_kpi','?'))}, spend={b.get('total_spend_mn','?')}, ROI={b.get('roi_kpi_per_mn_spend', 'n/a')}")

    # Stability section (if present)
    stab = narr.get('stability')
    if stab:
        lines.append("\nChannel stability (trust levels):")
        if stab.get('high_trust'):
            lines.append(" High trust (stable coeffs): " + ", ".join(stab['high_trust']))
        if stab.get('medium_trust'):
            lines.append(" Medium trust (moderate variability): " + ", ".join(stab['medium_trust']))
        if stab.get('low_trust'):
            lines.append(" Low trust (unstable / sign flips): " + ", ".join(stab['low_trust']))
        # list reasons for the most important low-trust channels
        if stab.get('reasons'):
            lines.append("\nStability notes:")
            for feat in (stab['low_trust'] + stab['medium_trust'])[:10]:
                reason = stab['reasons'].get(feat, '')
                lines.append(f" - {feat}: {reason}")

    lines.append("\nRecommendations:")
    for r in narr.get('recommendations', []):
        lines.append(" - " + r)
    lines.append("\nCaveats:")
    for c in narr.get('caveats', []):
        lines.append(" - " + c)
    # optionally include model snippet
    if narr.get('model_summary_snippet'):
        lines.append("\nModel summary snippet:")
        lines.append(narr['model_summary_snippet'][:1000])
    return "\n".join(lines)
