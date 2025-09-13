import pandas as pd, os
from .plotting import save_line, scatter_and_save

def run_eda(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    info = pd.DataFrame({'column': df.columns, 'dtype': df.dtypes.astype(str), 'missing': df.isna().sum().values})
    info.to_csv(os.path.join(out_dir,'columns_info.csv'), index=False)
    num_summary = df.select_dtypes(include=['number']).describe().T
    num_summary.to_csv(os.path.join(out_dir,'numeric_summary.csv'))
    if 'date' in df.columns:
        save_line(df['date'], df['kpi_val_sales_mn'],'KPI trend','kpi_trend.png', out_dir)
    core = ['wtd_dist_max','num_dist_max','trade_promo_spends_mn','consumer_promo_spends_mn','tv_total_spends_mn']
    for c in core:
        if c in df.columns:
            scatter_and_save(df[c], df['kpi_val_sales_mn'], f'{c} vs KPI', f'scatter_{c}_kpi.png', out_dir)
    corr = df.corr(numeric_only=True)
    corr.to_csv(os.path.join(out_dir,'correlation_matrix.csv'))
    return out_dir
