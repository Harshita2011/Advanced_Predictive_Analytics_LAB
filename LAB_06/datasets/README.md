# Lab 06 Datasets Guide: Time-Series Crime Forecasting

This directory contains the datasets and generation utilities for **Lab 06: Time-Series Forecasting with AR and ARIMA Models** (Course: `MDI3003 - Advanced Predictive Analytics`, SCOPE, VIT Vellore).

---

## 1. Datasets Overview

| Dataset | Filename | Frequency | Primary Unit | Records | Source / Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NYPD Complaint Data** | `NYPD_Complaint_Data_Current__Year_To_Date__20260901.csv` | Weekly (`W-MON`) | Borough (`BORO_NM = BROOKLYN`) | ~638,988 | NYC Open Data (YTD extract covering Jan 2025 to Aug 2026) |
| **Chicago Crime Data** | `Crimes_-_2001_to_Present_20260831.csv` | Daily (`D`) | Police District (`District = 12`) | ~154,753 | City of Chicago Data Portal (Extract covering Dec 2025 to Aug 2026) |

---

## 2. NYPD Complaint Data Schema

Used by:
- `23MID0043_Lab06_Crime_AR_ARIMA.py` / `23MID0043_Lab06_Crime_AR_ARIMA.ipynb`

### Key Fields:
- `CMPLNT_NUM`: Unique incident identifier (used strictly for auditing & deduplication).
- `CMPLNT_FR_DT`: Exact occurrence date (`%m/%d/%Y`).
- `BORO_NM`: Borough name (`BROOKLYN`, `MANHATTAN`, `BRONX`, `QUEENS`, `STATEN ISLAND`).
- `OFNS_DESC`: Primary offense description (e.g. `PETIT LARCENY`, `GRAND LARCENY`, `ASSAULT 3 & RELATED OFFENSES`, etc.).
- `LAW_CAT_CD`: Level of offense (`MISDEMEANOR`, `FELONY`, `VIOLATION`).
- `ADDR_PCT_CD`: Police precinct identifier code.

### Official Source URL:
[NYC Open Data - NYPD Complaint Data Current (Year To Date)](https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243)

---

## 3. Chicago Crime Data Schema

Used by:
- `Lab06_Crime_AR_ARIMA.py` / `Lab06_Crime_AR_ARIMA.ipynb`

### Key Fields:
- `ID` / `Case Number`: Unique incident case identifier.
- `Date`: Timestamp of crime occurrence.
- `Primary Type`: Offense classification (`THEFT`, `BATTERY`, `CRIMINAL DAMAGE`, etc.).
- `District`: Police District integer identifier (e.g., District `12` primary, District `8` secondary replication).
- `Arrest`: Boolean flag indicating whether an arrest was made.
- `Domestic`: Boolean flag indicating domestic-related incidents.

### Official Source URL:
[City of Chicago Data Portal - Crimes 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2)
