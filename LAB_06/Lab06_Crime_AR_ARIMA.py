"""
================================================================================
LAB 06: Time-Series Forecasting with AR and ARIMA
MDI3003 - Advanced Predictive Analytics
Chicago Crime Incident Forecasting (District Level)
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

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

CONFIG = {
    'dataset': 'Chicago Crimes - 2001 to Present (instructor-frozen extract, accessed 2026-08-31)',
    'source_url': 'https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2',
    'date_col': 'Date',
    'location_col': 'District',
    'location_value': 12,        # primary district
    'second_location_value': 8,  # replication district
    'category_col': 'Primary Type',
    'category_value': None,
    'frequency': 'D',            # daily
    'test_periods': 14,          # 14 days locked test horizon
    'ar_lags': 7,
    'arima_candidates': [(1, 0, 0), (2, 0, 0), (1, 1, 1), (2, 1, 1), (1, 1, 0)],
    'rolling_horizon': 7,
    'rolling_step': 7,
    'seed': SEED,
}


def score(y_true, y_pred):
    return {
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred)))
    }


def build_series(data, location_value, config):
    loc = data[data[config['location_col']] == location_value].copy()
    if config['category_value'] is not None:
        loc = loc[loc[config['category_col']].astype(str).str.upper() == config['category_value'].upper()]

    assert len(loc) > 0, f'Selected location has no observations: {location_value}'
    y = (
        loc.set_index(config['date_col'])
        .resample(config['frequency'])
        .size()
        .rename('incidents')
        .asfreq(config['frequency'], fill_value=0)
    )
    assert y.index.is_monotonic_increasing
    assert y.index.is_unique
    assert y.isna().sum() == 0
    return y, loc


def calendar_exog(index):
    dow = index.dayofweek.astype(float)
    return pd.DataFrame({
        'sin_dow': np.sin(2 * np.pi * dow / 7),
        'cos_dow': np.cos(2 * np.pi * dow / 7),
    }, index=index)


def rolling_origins(series, initial, horizon, step):
    origins = []
    end = initial
    while end + horizon <= len(series):
        origins.append((end, end + horizon))
        end += step
    return origins


def run_pipeline(data_path=None, show_plots=False):
    out_dir = Path("outputs") / "chicago"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    actual_data_path = data_path
    if actual_data_path is None:
        for p in [Path("Crimes_-_2001_to_Present_20260831.csv"),
                  Path("datasets/Crimes_-_2001_to_Present_20260831.csv")]:
            if p.exists():
                actual_data_path = str(p)
                break
        if actual_data_path is None:
            actual_data_path = "Crimes_-_2001_to_Present_20260831.csv"

    print("=" * 70)
    print("Chicago Crime Forecasting Pipeline (AR, ARIMA, SARIMA, SARIMAX)")
    print(f"Data file: {actual_data_path}")
    print("=" * 70)

    df = pd.read_csv(actual_data_path)
    required = {CONFIG['date_col'], CONFIG['location_col']}
    missing = required - set(df.columns)
    assert not missing, f'Missing required columns: {missing}'

    df[CONFIG['date_col']] = pd.to_datetime(df[CONFIG['date_col']], errors='coerce', format='mixed')
    n_before = len(df)
    df = df.dropna(subset=[CONFIG['date_col']]).copy()
    if 'Case Number' in df.columns:
        df = df.drop_duplicates(subset=['Case Number'])

    print(f"Loaded {len(df)} records. Date range: {df[CONFIG['date_col']].min()} to {df[CONFIG['date_col']].max()}")

    # Primary series
    y, loc1_df = build_series(df, CONFIG['location_value'], CONFIG)
    print(f"District {CONFIG['location_value']} series: {len(y)} daily observations")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    y.plot(ax=ax, title=f"Daily Reported Incidents - District {CONFIG['location_value']}", color='steelblue')
    ax.set_ylabel("Incidents / day")
    plt.tight_layout()
    plt.savefig(fig_dir / f"series_district_{CONFIG['location_value']}.png", bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Split
    H = CONFIG['test_periods']
    train, test = y.iloc[:-H], y.iloc[-H:]

    # ADF
    adf_stat, adf_p, *_ = adfuller(train)
    print(f"ADF Statistic = {adf_stat:.3f}, p-value = {adf_p:.4f}")

    # ACF / PACF
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    plot_acf(train, lags=min(28, len(train)//2 - 1), ax=ax[0], title='ACF (Training)')
    plot_pacf(train, lags=min(28, len(train)//2 - 1), ax=ax[1], title='PACF (Training)', method='ywm')
    plt.tight_layout()
    plt.savefig(fig_dir / 'acf_pacf_training.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Models
    naive_pred = np.repeat(train.iloc[-1], len(test))
    naive_score = score(test, naive_pred)

    ar_model = AutoReg(train, lags=CONFIG['ar_lags'], trend='ct').fit()
    ar_pred = ar_model.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
    ar_score = score(test, ar_pred)

    arima_rows = []
    for order in CONFIG['arima_candidates']:
        try:
            m = ARIMA(train, order=order).fit()
            arima_rows.append({'order': order, 'aic': m.aic, 'bic': m.bic, 'model': m})
        except Exception:
            pass

    cand_df = pd.DataFrame(arima_rows).sort_values('aic')
    best = cand_df.dropna(subset=['aic']).iloc[0]
    arima_model = best['model']
    arima_pred = arima_model.forecast(len(test))
    arima_score = score(test, arima_pred)

    cand_df[['order', 'aic', 'bic']].to_csv(out_dir / 'arima_candidate_table.csv', index=False)

    results = pd.DataFrame([
        {'Model': 'Naive', **naive_score},
        {'Model': f"AR({CONFIG['ar_lags']})", **ar_score},
        {'Model': f"ARIMA{best['order']}", **arima_score},
    ]).sort_values('MAE')
    results.to_csv(out_dir / 'model_comparison.csv', index=False)
    print("\nModel Comparison:")
    print(results.to_string(index=False))

    # Forecast and intervals
    fc = arima_model.get_forecast(steps=len(test))
    mean_fc = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    ci.columns = ['lower', 'upper']

    pred_table = pd.DataFrame({
        'actual': test,
        'naive': naive_pred,
        'AR': ar_pred.values if hasattr(ar_pred, 'values') else ar_pred,
        'ARIMA': arima_pred.values if hasattr(arima_pred, 'values') else arima_pred,
        'ARIMA_lower95': ci['lower'].values,
        'ARIMA_upper95': ci['upper'].values,
    }, index=test.index)
    pred_table.to_csv(out_dir / 'test_predictions.csv')

    fig, ax = plt.subplots(figsize=(11, 4))
    train.iloc[-30:].plot(ax=ax, label='Train (recent)', color='black', alpha=0.5)
    test.plot(ax=ax, label='Actual Test', color='black', linewidth=2)
    ax.plot(test.index, naive_pred, label='Naive', linestyle=':', color='gray')
    ax.plot(test.index, ar_pred, label=f"AR({CONFIG['ar_lags']})", linestyle='--', color='orange')
    ax.plot(test.index, ar_pred, label=f"ARIMA{best['order']}", linestyle='-', color='crimson', linewidth=2)
    ax.fill_between(test.index, ci['lower'], ci['upper'], color='crimson', alpha=0.15, label='ARIMA 95% CI')
    ax.set_title(f"Forecast Comparison on Test Set - District {CONFIG['location_value']}")
    ax.set_ylabel('Incidents / day')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / 'forecast_comparison.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Residual diagnostics
    resid = arima_model.resid.iloc[max(best['order']):]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    resid.plot(ax=ax[0], title='ARIMA Residuals')
    ax[0].axhline(0, color='gray', linestyle='--')
    plot_acf(resid, lags=min(20, len(resid)//2 - 1), ax=ax[1], title='Residual ACF')
    plt.tight_layout()
    plt.savefig(fig_dir / 'residual_diagnostics.png', bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close()

    # Rolling origin
    origins = rolling_origins(train, initial=len(train) - 3 * CONFIG['rolling_step'],
                              horizon=CONFIG['rolling_horizon'], step=CONFIG['rolling_step'])
    cv_rows = []
    for k, (tr_end, te_end) in enumerate(origins):
        tr_k, te_k = train.iloc[:tr_end], train.iloc[tr_end:te_end]
        m_ar_k = AutoReg(tr_k, lags=CONFIG['ar_lags'], trend='ct').fit()
        pred_ar_k = m_ar_k.predict(start=len(tr_k), end=len(tr_k) + len(te_k) - 1, dynamic=False)
        m_arima_k = ARIMA(tr_k, order=best['order']).fit()
        pred_arima_k = m_arima_k.forecast(len(te_k))

        cv_rows.append({
            'fold': k + 1, 'origin_date': str(train.index[tr_end-1].date()),
            'AR_MAE': score(te_k, pred_ar_k)['MAE'],
            'ARIMA_MAE': score(te_k, pred_arima_k)['MAE'],
            'AR_RMSE': score(te_k, pred_ar_k)['RMSE'],
            'ARIMA_RMSE': score(te_k, pred_arima_k)['RMSE'],
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / 'rolling_origin_results.csv', index=False)

    # Second location replication
    y2, _ = build_series(df, CONFIG['second_location_value'], CONFIG)
    tr2, te2 = y2.iloc[:-H], y2.iloc[-H:]
    nv2 = np.repeat(tr2.iloc[-1], len(te2))
    ar2 = AutoReg(tr2, lags=CONFIG['ar_lags'], trend='ct').fit()
    ar2_p = ar2.predict(start=len(tr2), end=len(tr2) + len(te2) - 1, dynamic=False)
    m2 = ARIMA(tr2, order=best['order']).fit()
    m2_p = m2.forecast(len(te2))

    res2 = pd.DataFrame([
        {'Location': CONFIG['second_location_value'], 'Model': 'Naive', **score(te2, nv2)},
        {'Location': CONFIG['second_location_value'], 'Model': f"AR({CONFIG['ar_lags']})", **score(te2, ar2_p)},
        {'Location': CONFIG['second_location_value'], 'Model': f"ARIMA{best['order']}", **score(te2, m2_p)},
    ])
    res1_labeled = results.copy()
    res1_labeled.insert(0, 'Location', CONFIG['location_value'])
    two_loc = pd.concat([res1_labeled, res2], ignore_index=True)
    two_loc.to_csv(out_dir / 'two_location_comparison.csv', index=False)

    # SARIMA
    sarima = SARIMAX(train, order=best['order'], seasonal_order=(1, 0, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarima_pred = sarima.forecast(len(test))
    sarima_score = score(test, sarima_pred)
    sarima_comp = pd.DataFrame([
        {'Model': f"ARIMA{best['order']}", **arima_score, 'AIC': arima_model.aic},
        {'Model': f"SARIMA{best['order']}x(1,0,1,7)", **sarima_score, 'AIC': sarima.aic},
    ])
    sarima_comp.to_csv(out_dir / 'sarima_comparison.csv', index=False)

    # SARIMAX
    ex_tr = calendar_exog(train.index)
    ex_te = calendar_exog(test.index)
    sarimax = SARIMAX(train, order=best['order'], exog=ex_tr,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_p = sarimax.forecast(len(test), exog=ex_te)
    sarimax_comp = pd.DataFrame([
        {'Model': f"Plain ARIMA{best['order']}", **arima_score},
        {'Model': f"SARIMAX{best['order']} + harmonic exog", **score(test, sarimax_p)},
    ])
    sarimax_comp.to_csv(out_dir / 'sarimax_comparison.csv', index=False)

    # Log1p
    tr_log = np.log1p(train)
    m_log = ARIMA(tr_log, order=best['order']).fit()
    p_log = np.expm1(m_log.forecast(len(test)))
    log_comp = pd.DataFrame([
        {'Model': f"Plain ARIMA{best['order']}", **arima_score, 'Negative_Forecasts': int((arima_pred < 0).sum())},
        {'Model': f"Log1p-ARIMA{best['order']}", **score(test, p_log), 'Negative_Forecasts': int((p_log < 0).sum())},
    ])
    log_comp.to_csv(out_dir / 'count_aware_comparison.csv', index=False)

    # Save manifest
    manifest = {
        **{k: v for k, v in CONFIG.items() if k != 'arima_candidates'},
        'arima_candidates': [list(c) for c in CONFIG['arima_candidates']],
        'n_total_periods_loc1': int(len(y)),
        'n_train_loc1': int(len(train)),
        'n_test_loc1': int(len(test)),
        'selected_arima_order_loc1': list(best['order']),
        'python': sys.version,
        'platform': platform.platform(),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nChicago pipeline completed. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chicago Crimes AR/ARIMA Forecasting Pipeline")
    parser.add_argument("--data-path", type=str, default=None, help="Path to input CSV")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    args = parser.parse_args()

    run_pipeline(data_path=args.data_path, show_plots=args.show_plots)
