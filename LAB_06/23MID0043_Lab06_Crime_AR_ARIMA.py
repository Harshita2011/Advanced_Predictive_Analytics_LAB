"""
================================================================================
LAB 06: Time-Series Forecasting with AR and ARIMA
MDI3003 - Advanced Predictive Analytics
Faculty: Dr. Durgesh Kumar, Assistant Professor (Senior), SCOPE, VIT Vellore
Semester: Fall Semester 2026-2027

Student Name: Harshita
Registration Number: 23MID0043
Core Experiment: Weekly NYPD Reported Crime Incidents in Brooklyn
================================================================================
"""

import os
import sys
import json
import time
import shutil
import zipfile
import platform
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

# ============================================================
# STUDENT / SUBMISSION IDENTIFICATION
# ============================================================
STUDENT_NAME = "Harshita"
REGISTRATION_NUMBER = "23MID0043"
COURSE_CODE = "MDI3003"
LAB_TITLE = "Lab06_Crime_AR_ARIMA"
FACULTY = "Dr. Durgesh Kumar, Assistant Professor (Senior), SCOPE, VIT Vellore"
SEMESTER = "Fall Semester 2026-2027"

SEED = 42
np.random.seed(SEED)

# Output directory structure
OUT = Path("outputs") / "nypd"
FIG = OUT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def get_default_config():
    return {
        'dataset': 'NYPD Complaint Data - Current (Year To Date)',
        'dataset_url': 'https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243',
        'access_date': '2026-09-01',
        'date_col': 'CMPLNT_FR_DT',
        'location_col': 'BORO_NM',
        'location_value': 'BROOKLYN',          # core-experiment location
        'second_location_value': 'MANHATTAN',   # replication location
        'category_col': 'OFNS_DESC',
        'category_value': None,                 # None = all offenses
        'frequency': 'W-MON',                   # weekly series starting Monday
        'test_periods': 12,                     # locked future holdout
        'valid_year_min': 2026,
        'seed': SEED,
    }


def find_dataset(filename="NYPD_Complaint_Data_Current__Year_To_Date__20260901.csv"):
    candidate_paths = [
        Path(filename),
        Path("datasets") / filename,
        Path("..") / filename,
        Path("..") / "datasets" / filename
    ]
    for p in candidate_paths:
        if p.exists():
            return str(p)
    return filename


def score(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }


def build_series(data, location_value, config):
    loc = data[data[config['location_col']].astype(str) == str(location_value)].copy()
    if config.get('category_value'):
        loc = loc[loc[config['category_col']].astype(str).str.upper() == config['category_value'].upper()]
    assert len(loc) > 0, f'Selected location has no observations: {location_value}'

    y = (loc.set_index(config['date_col'])
           .resample(config['frequency'])
           .size()
           .rename('incidents')
           .asfreq(config['frequency'], fill_value=0))

    assert y.index.is_monotonic_increasing
    assert y.index.is_unique
    assert y.isna().sum() == 0
    return y, loc


def calendar_exog(index):
    woy = index.isocalendar().week.astype(float)
    return pd.DataFrame({
        'sin52': np.sin(2 * np.pi * woy / 52),
        'cos52': np.cos(2 * np.pi * woy / 52),
    }, index=index)


def rolling_origins(series, initial, horizon, step):
    origins = []
    end = initial
    while end + horizon <= len(series):
        origins.append((end, end + horizon))
        end += step
    return origins


def run_pipeline(data_path=None, show_plots=False):
    print("=" * 70)
    print(f"Executing Time-Series Crime Forecasting Pipeline: {REGISTRATION_NUMBER}")
    print("=" * 70)

    config = get_default_config()
    actual_data_path = data_path if data_path else find_dataset()

    if not Path(actual_data_path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {actual_data_path}. Please run generate_datasets.py or specify --data-path."
        )

    print(f"\n[1] Loading Data from: {actual_data_path}")
    usecols = [config['date_col'], config['location_col'], 'CMPLNT_NUM',
               config['category_col'], 'LAW_CAT_CD', 'ADDR_PCT_CD']
    
    # Load and audit
    df_raw = pd.read_csv(actual_data_path, usecols=usecols, low_memory=False)
    print(f"Raw shape: {df_raw.shape}")

    required = {config['date_col'], config['location_col']}
    missing = required - set(df_raw.columns)
    assert not missing, f'Missing required columns: {missing}'

    # Parse dates and filter
    df = df_raw.copy()
    df[config['date_col']] = pd.to_datetime(df[config['date_col']], format='%m/%d/%Y', errors='coerce')
    n_before = len(df)
    df = df.dropna(subset=[config['date_col']]).copy()
    print(f"Dropped {n_before - len(df)} unparseable date rows.")

    bad_year_mask = (df[config['date_col']].dt.year < config['valid_year_min'] - 1) | \
                    (df[config['date_col']].dt.year > pd.Timestamp.today().year + 1)
    df = df.loc[~bad_year_mask].copy()
    df = df[~df[config['location_col']].isin(['(null)', None]) & df[config['location_col']].notna()].copy()
    print(f"Clean shape after audit: {df.shape}")

    # Build primary series
    print(f"\n[2] Building Location Series for {config['location_value']}...")
    y, loc_df = build_series(df, config['location_value'], config)
    print(f"Total weekly periods: {len(y)}")

    # Plot raw series
    fig, ax = plt.subplots(figsize=(11, 4))
    y.plot(ax=ax, title=f"Weekly Reported Crime Incidents  {config['location_value']} (NYPD)", color='navy')
    ax.set_ylabel('Incidents / week')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / '01_raw_series.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Rolling statistics
    fig, ax = plt.subplots(figsize=(11, 4))
    y.plot(ax=ax, label='Weekly counts', alpha=0.5, color='gray')
    y.rolling(4).mean().plot(ax=ax, label='4-week rolling mean', color='tab:blue', linewidth=2)
    y.rolling(8).mean().plot(ax=ax, label='8-week rolling mean', color='tab:orange', linewidth=2)
    ax.set_title(f"Rolling Central Tendency  {config['location_value']}")
    ax.set_ylabel('Incidents / week')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / '02_rolling_stats.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Split train/test
    H = config['test_periods']
    train = y.iloc[:-H]
    test = y.iloc[-H:]
    print(f"\n[3] Chronological Split: Train={len(train)} periods, Test={len(test)} periods")

    # ADF Test
    adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(train, autolag='AIC')
    print(f"ADF Statistic: {adf_stat:.4f} (p-value: {adf_p:.4f})")

    # ACF / PACF Plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(train, lags=min(20, len(train)//2 - 1), ax=ax[0], title=f"ACF - Train ({config['location_value']})")
    plot_pacf(train, lags=min(20, len(train)//2 - 1), ax=ax[1], title=f"PACF - Train ({config['location_value']})", method='ywm')
    plt.tight_layout()
    plt.savefig(FIG / '03_acf_pacf.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # 1. Naive Baseline
    naive_pred = np.repeat(train.iloc[-1], len(test))
    naive_scores = score(test, naive_pred)

    # 2. AR(p) Model
    AR_LAGS = 4
    ar_model = AutoReg(train, lags=AR_LAGS, old_names=False, trend='ct').fit()
    ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    ar_scores = score(test, ar_pred)

    # 3. ARIMA Candidate Search
    print("\n[4] ARIMA Order Search (training-only):")
    candidates = [(1, 0, 0), (2, 0, 0), (1, 1, 1), (2, 1, 1), (2, 1, 2), (3, 1, 1)]
    arima_rows = []
    for order in candidates:
        try:
            m = ARIMA(train, order=order).fit()
            arima_rows.append({'order': order, 'aic': m.aic, 'bic': m.bic, 'model': m})
        except Exception as e:
            arima_rows.append({'order': order, 'aic': np.nan, 'bic': np.nan, 'model': None})

    candidate_df = pd.DataFrame(arima_rows).sort_values('aic')
    best = candidate_df.dropna(subset=['aic']).iloc[0]
    arima_model = best['model']
    print(f"Selected ARIMA Order: {best['order']} (AIC: {best['aic']:.2f})")

    arima_pred = arima_model.forecast(len(test))
    arima_scores = score(test, arima_pred)

    # Model comparison table
    results = pd.DataFrame([
        {'Model': 'Naive', **naive_scores},
        {'Model': f"AR({AR_LAGS})", **ar_scores},
        {'Model': f"ARIMA{best['order']}", **arima_scores},
    ]).sort_values('MAE')
    print("\n[5] Model Comparison on Test Set:")
    print(results.to_string(index=False))
    results.to_csv(OUT / 'model_comparison.csv', index=False)

    # Forecast and Confidence Interval Plot
    fc_res = arima_model.get_forecast(len(test))
    ci = fc_res.conf_int(alpha=0.05)
    ci.columns = ['lower', 'upper']

    pred_df = pd.DataFrame({
        'actual': test,
        'naive': naive_pred,
        'AR': ar_pred.values if hasattr(ar_pred, 'values') else ar_pred,
        'ARIMA': arima_pred.values if hasattr(arima_pred, 'values') else arima_pred,
        'ARIMA_lower95': ci['lower'].values,
        'ARIMA_upper95': ci['upper'].values,
    }, index=test.index)
    pred_df.to_csv(OUT / 'test_predictions.csv')

    fig, ax = plt.subplots(figsize=(12, 5))
    train.iloc[-20:].plot(ax=ax, label='Train (recent)', color='black', alpha=0.6)
    test.plot(ax=ax, label='Actual (test)', color='black', linewidth=2)
    ax.plot(test.index, naive_pred, label='Naive', linestyle=':', color='tab:gray')
    ax.plot(test.index, ar_pred, label=f'AR({AR_LAGS})', linestyle='--', color='tab:orange')
    ax.plot(test.index, arima_pred, label=f'ARIMA{best["order"]}', linestyle='-', color='tab:red', linewidth=2)
    ax.fill_between(test.index, ci['lower'], ci['upper'], color='tab:red', alpha=0.15, label='ARIMA 95% CI')
    ax.set_title(f"Forecast Comparison on Locked Test Set  {config['location_value']}")
    ax.set_ylabel('Incidents / week')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / '04_forecast_comparison.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Residual Diagnostics
    print("\n[6] Residual Diagnostics...")
    resid = arima_model.resid.iloc[max(best['order']):]
    lb_lags = min(6, len(resid)//3)
    lb = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
    print(f"Ljung-Box p-value at lag {lb_lags}: {lb['lb_pvalue'].iloc[0]:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    resid.plot(ax=ax[0], title='ARIMA Residuals')
    ax[0].axhline(0, color='gray', linestyle='--')
    plot_acf(resid, lags=min(15, len(resid)//2 - 1), ax=ax[1], title='Residual ACF')
    resid.plot(kind='kde', ax=ax[2], title='Residual Density')
    plt.tight_layout()
    plt.savefig(FIG / '05_residual_diagnostics.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Rolling-Origin Backtest
    print("\n[7] Rolling-Origin Backtesting...")
    cv_horizon = 4
    cv_step = 4
    cv_initial = max(24, len(train) - 16)
    origins = rolling_origins(train, initial=cv_initial, horizon=cv_horizon, step=cv_step)

    fold_rows = []
    for k, (tr_end, te_end) in enumerate(origins):
        tr_k = train.iloc[:tr_end]
        te_k = train.iloc[tr_end:te_end]
        m_ar = AutoReg(tr_k, lags=AR_LAGS, old_names=False, trend='ct').fit()
        pred_ar = m_ar.predict(start=len(tr_k), end=len(tr_k) + len(te_k) - 1, dynamic=False)
        m_ar_score = score(te_k, pred_ar)

        m_arima = ARIMA(tr_k, order=best['order']).fit()
        pred_arima = m_arima.forecast(len(te_k))
        m_arima_score = score(te_k, pred_arima)

        fold_rows.append({
            'fold': k + 1, 'origin_date': train.index[tr_end-1],
            'AR_MAE': m_ar_score['MAE'], 'ARIMA_MAE': m_arima_score['MAE'],
            'AR_RMSE': m_ar_score['RMSE'], 'ARIMA_RMSE': m_arima_score['RMSE']
        })

    fold_df = pd.DataFrame(fold_rows)
    print(fold_df.to_string(index=False))
    fold_df.to_csv(OUT / 'rolling_origin_results.csv', index=False)

    # Replicate on Second Location
    print(f"\n[8] Replicating on Second Location: {config['second_location_value']}...")
    y2, loc2_df = build_series(df, config['second_location_value'], config)
    tr2, te2 = y2.iloc[:-H], y2.iloc[-H:]

    nv2 = np.repeat(tr2.iloc[-1], len(te2))
    ar2 = AutoReg(tr2, lags=AR_LAGS, old_names=False, trend='ct').fit()
    ar2_pred = ar2.predict(start=len(tr2), end=len(tr2) + len(te2) - 1, dynamic=False)
    m2 = ARIMA(tr2, order=best['order']).fit()
    m2_pred = m2.forecast(len(te2))

    results2 = pd.DataFrame([
        {'Location': config['second_location_value'], 'Model': 'Naive', **score(te2, nv2)},
        {'Location': config['second_location_value'], 'Model': f"AR({AR_LAGS})", **score(te2, ar2_pred)},
        {'Location': config['second_location_value'], 'Model': f"ARIMA{best['order']}", **score(te2, m2_pred)},
    ])
    results1_labeled = results.copy()
    results1_labeled.insert(0, 'Location', config['location_value'])
    two_loc_comp = pd.concat([results1_labeled, results2], ignore_index=True)
    two_loc_comp.to_csv(OUT / 'two_location_comparison.csv', index=False)

    # SARIMA
    print("\n[9] Fitting SARIMA (Monthly cycle s=4)...")
    SEASONAL_PERIOD = 4
    sarima = SARIMAX(train, order=best['order'], seasonal_order=(1, 0, 1, SEASONAL_PERIOD),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima.forecast(len(test))
    sarima_scores = score(test, sarima_pred)
    print(f"SARIMA Scores: {sarima_scores}")

    # Count-aware log1p ARIMA
    print("\n[10] Count-Aware Modeling (Log1p-transformed ARIMA)...")
    train_log = np.log1p(train)
    m_log = ARIMA(train_log, order=best['order']).fit()
    pred_log_scale = m_log.forecast(len(test))
    pred_count_scale = np.expm1(pred_log_scale)
    log_scores = score(test, pred_count_scale)

    comparison_countaware = pd.DataFrame([
        {'Model': f"Plain ARIMA{best['order']}", **arima_scores, 'Negative_Forecasts': int((arima_pred < 0).sum())},
        {'Model': f"Log1p-ARIMA{best['order']}", **log_scores, 'Negative_Forecasts': int((pred_count_scale < 0).sum())},
    ])
    print(comparison_countaware.to_string(index=False))
    comparison_countaware.to_csv(OUT / 'count_aware_comparison.csv', index=False)

    # SARIMAX with Calendar Harmonics
    print("\n[11] SARIMAX with Calendar Exogenous Features...")
    exog_train = calendar_exog(train.index)
    exog_test = calendar_exog(test.index)
    sarimax_model = SARIMAX(train, order=best['order'], exog=exog_train,
                            enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_pred = sarimax_model.forecast(len(test), exog=exog_test)
    sarimax_scores = score(test, sarimax_pred)
    print(f"SARIMAX Scores: {sarimax_scores}")

    # Multi-Borough Replication
    print("\n[12] Five-Borough Replication...")
    all_boroughs = sorted(df[config['location_col']].dropna().unique().tolist())
    multi_rows = []
    for b in all_boroughs:
        try:
            yb, _ = build_series(df, b, config)
            trb, teb = yb.iloc[:-H], yb.iloc[-H:]
            nv_b = np.repeat(trb.iloc[-1], len(teb))
            arb = AutoReg(trb, lags=AR_LAGS, old_names=False, trend='ct').fit()
            arb_p = arb.predict(start=len(trb), end=len(trb) + len(teb) - 1, dynamic=False)
            mib = ARIMA(trb, order=best['order']).fit()
            mib_p = mib.forecast(len(teb))

            multi_rows.append({
                'Borough': b, 'n_periods': len(yb), 'mean_weekly': yb.mean(), 'std_weekly': yb.std(),
                'Naive_MAE': score(teb, nv_b)['MAE'],
                'AR_MAE': score(teb, arb_p)['MAE'],
                'ARIMA_MAE': score(teb, mib_p)['MAE'],
            })
        except Exception as e:
            print(f"Skipped {b}: {e}")

    multi_df = pd.DataFrame(multi_rows).sort_values('mean_weekly', ascending=False)
    multi_df.to_csv(OUT / 'multi_borough_replication.csv', index=False)
    print(multi_df.to_string(index=False))

    # Structural Break Screen (CUSUM)
    print("\n[13] Structural Break Screen (CUSUM)...")
    z = (y - y.mean()) / y.std()
    cusum = z.cumsum()
    fig, ax = plt.subplots(figsize=(11, 4))
    cusum.plot(ax=ax, color='purple')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title(f"CUSUM of Standardized Weekly Counts  {config['location_value']}")
    ax.set_ylabel('Cumulative standardized deviation')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / '12_cusum_screen.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Category Series
    top_categories = df[df[config['location_col']] == config['location_value']][config['category_col']].value_counts().head(5)
    CATEGORY_VALUE = top_categories.index[0]

    # Save Manifest
    print("\n[14] Saving Manifest & Reproducibility Record...")
    manifest = {
        **config,
        'student_name': STUDENT_NAME,
        'registration_number': REGISTRATION_NUMBER,
        'course_code': COURSE_CODE,
        'faculty': FACULTY,
        'semester': SEMESTER,
        'n_total_periods_core_location': int(len(y)),
        'n_train': int(len(train)),
        'n_test': int(len(test)),
        'ar_lags': AR_LAGS,
        'arima_candidate_orders': [list(c) for c in candidates],
        'selected_arima_order': list(best['order']),
        'sarima_seasonal_order': [1, 0, 1, SEASONAL_PERIOD],
        'python': sys.version,
        'platform': platform.platform(),
        'generated_utc': pd.Timestamp.utcnow().isoformat(),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

    repro_record = pd.DataFrame([
        {'Item': 'Dataset name/version/access date', 'Record': f"{config['dataset']} / accessed {config['access_date']}"},
        {'Item': 'Source URL', 'Record': config['dataset_url']},
        {'Item': 'Local extract file', 'Record': actual_data_path},
        {'Item': 'Date column used', 'Record': config['date_col']},
        {'Item': 'Location column/value (core)', 'Record': f"{config['location_col']} = {config['location_value']}"},
        {'Item': 'Location column/value (2nd)', 'Record': f"{config['location_col']} = {config['second_location_value']}"},
        {'Item': 'Crime category filter', 'Record': CATEGORY_VALUE},
        {'Item': 'Aggregation frequency', 'Record': config['frequency']},
        {'Item': 'Observation window', 'Record': f"{y.index.min().date()} to {y.index.max().date()}"},
        {'Item': 'Forecast horizon', 'Record': f"{H} weeks"},
        {'Item': 'AR lags', 'Record': AR_LAGS},
        {'Item': 'ARIMA candidate orders', 'Record': str(candidates)},
        {'Item': 'Selected ARIMA order', 'Record': str(best['order'])},
        {'Item': 'SARIMA seasonal order', 'Record': f"(1,0,1,{SEASONAL_PERIOD})"},
        {'Item': 'Python / statsmodels versions', 'Record': f"{sys.version.split()[0]}"},
    ])
    repro_record.to_csv(OUT / 'reproducibility_record.csv', index=False)

    # Build ZIP package from outputs/nypd
    zip_name = f"{REGISTRATION_NUMBER}_{LAB_TITLE}_outputs.zip"
    if Path(zip_name).exists():
        Path(zip_name).unlink()

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
        for p in OUT.rglob('*'):
            zf.write(p, arcname=p.relative_to(OUT.parent))

    print(f"\nCreated submission package: {zip_name} ({Path(zip_name).stat().st_size / 1024:.1f} KB)")
    print(f"Outputs written to: {OUT.resolve()}")
    print("Execution completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Lab 06 AR/ARIMA Crime Forecasting Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input CSV file")
    parser.add_argument("--show-plots", action="store_true", help="Display matplotlib interactive plots")
    args = parser.parse_args()

    run_pipeline(data_path=args.data_path, show_plots=args.show_plots)
