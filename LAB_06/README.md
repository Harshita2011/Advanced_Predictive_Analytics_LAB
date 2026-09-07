# Lab 06: Time-Series Forecasting with AR and ARIMA Models

**Course**: MDI3003 — Advanced Predictive Analytics  
**Faculty**: Dr. Durgesh Kumar, Assistant Professor (Senior), SCOPE, VIT Vellore  
**Semester**: Fall Semester 2026–2027  
**Student Name**: Harshita  
**Registration Number**: `23MID0043`  

---

## 📌 1. Project Overview

This repository implements an end-to-end time-series predictive modeling pipeline to forecast area-level reported crime incident counts. The study evaluates classical, autoregressive, moving-average, and seasonal models on real-world administrative public safety records.

### Key Objectives:
1. **Core Forecasting**: Formulate, train, and validate AutoRegressive $\text{AR}(p)$ and $\text{ARIMA}(p,d,q)$ models on regularized weekly crime incident series.
2. **Baseline Benchmark**: Establish locked chronological holdouts against a persistence/naive baseline.
3. **Statistical Diagnostics**: Evaluate stationarity (Augmented Dickey-Fuller test), lag selection (PACF/ACF), and residual autocorrelation (Ljung-Box white noise test).
4. **Stability & Replication**: Execute rolling-origin backtesting (cross-validation without lookahead bias) and spatial replication across multiple municipal boroughs / districts.
5. **Advanced Extensions**:
   - **SARIMA**: Seasonal ARIMA modeling periodic recurring cycles.
   - **SARIMAX**: Exogenous calendar harmonics ($\sin/\cos$ Fourier series of week-of-year).
   - **Count-Aware Log1p ARIMA**: Log-transformed variance-stabilizing forecasting to guarantee non-negative predictions.
   - **CUSUM Structural Break Screening**: Cumulative standardized residual screening for policy or data regime shifts.

---

## 📁 2. Repository Structure

```text
LAB_06/
├── 23MID0043_Lab06_Crime_AR_ARIMA.ipynb   # Primary NYPD crime forecasting notebook (23MID0043)
├── 23MID0043_Lab06_Crime_AR_ARIMA.py      # Primary NYPD crime forecasting Python script
├── Lab06_Crime_AR_ARIMA.ipynb             # Chicago district crime notebook
├── Lab06_Crime_AR_ARIMA.py                # Chicago district crime Python script
├── 23MID0043_Lab_06_Report.pdf            # Comprehensive academic lab report
├── 23MID0043_Lab06_Crime_AR_ARIMA_outputs.zip # Submission package archive
├── requirements.txt                       # Project dependencies
├── README.md                              # Main documentation
│
├── datasets/                              # Datasets and metadata
│   ├── NYPD_Complaint_Data_Current__Year_To_Date__20260901.csv  # NYPD extract (~638k records)
│   ├── Crimes_-_2001_to_Present_20260831.csv                    # Chicago extract (~154k records)
│   └── README.md                                                # Dataset schemas and metadata
│
└── outputs/                               # Unified experiment outputs
    ├── nypd/                              # NYPD Borough Forecast Outputs (23MID0043)
    │   ├── model_comparison.csv
    │   ├── test_predictions.csv
    │   ├── rolling_origin_results.csv
    │   ├── two_location_comparison.csv
    │   ├── multi_borough_replication.csv
    │   ├── count_aware_comparison.csv
    │   ├── reproducibility_record.csv
    │   ├── manifest.json
    │   └── figures/
    │       ├── 01_raw_series.png
    │       ├── 02_rolling_stats.png
    │       ├── 03_acf_pacf.png
    │       ├── 04_forecast_comparison.png
    │       ├── 05_residual_diagnostics.png
    │       └── 12_cusum_screen.png
    │
    └── chicago/                           # Chicago District Forecast Outputs
        ├── model_comparison.csv
        ├── test_predictions.csv
        ├── arima_candidate_table.csv
        ├── rolling_origin_results.csv
        ├── two_location_comparison.csv
        ├── sarima_comparison.csv
        ├── sarimax_comparison.csv
        ├── count_aware_comparison.csv
        ├── manifest.json
        └── figures/
            ├── series_district_12.png
            ├── acf_pacf_training.png
            ├── forecast_comparison.png
            └── residual_diagnostics.png
```

---

## ⚙️ 3. Installation & Environment Setup

### 3.1 Prerequisites
- Python `3.9`, `3.10`, `3.11`, `3.12`, or `3.13`
- `pip` package manager

### 3.2 Setup Virtual Environment (Recommended)

```bash
# Clone or navigate to the directory
cd "d:/Adv Predictive Lab/LAB_06"

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
# On Linux / macOS:
source venv/bin/activate
```

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 4. Datasets

The repository includes both raw datasets:

1. **NYPD Complaint Data - Current (Year-To-Date)**:
   - **Path**: `datasets/NYPD_Complaint_Data_Current__Year_To_Date__20260901.csv`
   - **Temporal Scope**: Weekly aggregated incident counts (`W-MON`)
   - **Primary Spatial Focus**: `BORO_NM = BROOKLYN` (Replicated across Manhattan, Bronx, Queens, Staten Island).
   - **Source**: [NYC Open Data Portal](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243)

2. **Chicago Crime Data (Crimes 2001 to Present)**:
   - **Path**: `datasets/Crimes_-_2001_to_Present_20260831.csv`
   - **Temporal Scope**: Daily aggregated incident counts (`D`)
   - **Primary Spatial Focus**: `District = 12` (Replicated on District `8`).
   - **Source**: [City of Chicago Data Portal](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)

---

## 🚀 5. How to Run

### 5.1 Running Python Scripts (`.py`)

#### Primary Student Pipeline (23MID0043):
```bash
python 23MID0043_Lab06_Crime_AR_ARIMA.py
```
*Options:*
- `--data-path <path>`: Custom path to NYPD CSV extract.
- `--show-plots`: Interactive matplotlib window display.

#### Chicago Crime Pipeline:
```bash
python Lab06_Crime_AR_ARIMA.py
```

---

### 5.2 Running Jupyter Notebooks (`.ipynb`)

Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook
```
Open any of the following:
- `23MID0043_Lab06_Crime_AR_ARIMA.ipynb`
- `Lab06_Crime_AR_ARIMA.ipynb`

Execute cells sequentially (**Cell > Run All**).

---

## 🔬 6. Methodology & Model Summary

### 6.1 Data Processing & Aggregation
- Occurrence timestamps (`CMPLNT_FR_DT`) parsed and validated.
- Weekly regular resampling (`W-MON`) with zero-filling ensures strictly monotonic, non-sparse time indices.
- Chronological, un-shuffled split: **75 training weeks** and **12 locked testing weeks** ($H=12$).

### 6.2 Model Specifications
1. **Naive (Last-Value Baseline)**: $\hat{y}_{t+h} = y_T$
2. **AutoRegressive Model $\text{AR}(p)$**:
   $$y_t = c + \beta t + \sum_{i=1}^p \phi_i y_{t-i} + \epsilon_t$$
   Lag order $p=4$ justified via Training Partial Autocorrelation Function (PACF).
3. **$\text{ARIMA}(p,d,q)$**:
   Candidate grid search over $[(1,0,0), (2,0,0), (1,1,1), (2,1,1), (2,1,2), (3,1,1)]$ evaluated on Training Akaike Information Criterion (AIC). Selected order: $\text{ARIMA}(2,1,2)$.
4. **$\text{SARIMA}(p,d,q) \times (P,D,Q)_s$**: Incorporates cyclical intra-quarter seasonality ($s=4$).
5. **$\text{SARIMAX}$**: Incorporates Fourier calendar exogenous harmonics:
   $$x_{1,t} = \sin\left(\frac{2\pi \cdot \text{WOY}_t}{52}\right), \quad x_{2,t} = \cos\left(\frac{2\pi \cdot \text{WOY}_t}{52}\right)$$
6. **Log1p Count-Aware $\text{ARIMA}$**: Models $z_t = \ln(1 + y_t)$ and inverts forecasts via $\hat{y} = \exp(\hat{z}) - 1$ to guarantee positivity.

---

## 📈 7. Results & Key Findings

### 7.1 Locked Test-Set Performance (Brooklyn Core Location)

| Model | MAE (Incidents/Week) | RMSE (Incidents/Week) |
| :--- | :---: | :---: |
| **$\text{AR}(4)$ (Trend + Intercept)** | **45.71** | **60.55** |
| **$\text{SARIMAX}(2,1,2) + \text{Harmonics}$** | **50.70** | **58.45** |
| **$\text{SARIMA}(2,1,2) \times (1,0,1)_4$** | 103.97 | 115.74 |
| **Naive Baseline** | 117.50 | 125.95 |
| **$\text{ARIMA}(2,1,2)$** | 128.36 | 141.96 |
| **$\text{Log1p-ARIMA}(2,1,2)$** | 128.48 | 142.44 |

### 7.2 Residual Diagnostics
- **ADF Stationarity Test**: Statistic = $-3.7530$ ($p = 0.0034 < 0.05$), confirming stationarity.
- **Ljung-Box Autocorrelation Test**: $p = 0.9795$ at lag 6 (residuals are indistinguishable from white noise).
- **Negative Forecast Check**: 0 negative predictions generated across all evaluated models.

---

## ⚖️ 8. Responsible Use & Governance Statement

1. **Administrative Measure**: Reported incident counts represent officially logged complaints, not a comprehensive census of all criminal activity. Reporting rates vary across neighborhoods and offense types.
2. **Area-Level Aggregate Only**: All forecasts are computed strictly at the aggregate borough/district level. No individual, personal, or address-level inferences are made.
3. **Decision-Support Scope**: These time-series forecasts are intended solely for academic study, resource planning, and analytical decision support—not for automated predictive policing or biased patrol routing.

---

## 📜 9. License & Academic Integrity

This project is prepared as part of the academic coursework for **MDI3003 (Advanced Predictive Analytics)** at **Vellore Institute of Technology (VIT Vellore)**. All code and documentation adhere to institutional academic integrity guidelines.
