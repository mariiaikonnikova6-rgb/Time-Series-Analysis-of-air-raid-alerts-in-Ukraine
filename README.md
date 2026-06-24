# Historical Time Series Analysis and Forecasting of Air Raid Alert Duration in Ukraine

## Overview

This project performs historical time series analysis and one-step-ahead forecasting of daily air raid alert duration in Ukraine.

The workflow transforms event-level air raid alert records into continuous daily regional time series, investigates temporal structure, and compares forecasting models for **Kyiv City**.

The project is educational and analytical. It is **not** a real-time warning system and must not be used for safety, operational, military, or personal decision-making.

---

## Research Goal

The main research question is:

> Can the daily total duration of air raid alerts in Kyiv City be forecast more accurately using historical time-series models than with a simple weekly baseline?

The forecasting target is:

```text
Kyiv City daily total alert duration in minutes
```

The task is formulated as **one-step-ahead forecasting**:

```text
Use information available before day t
→ forecast total alert duration for day t
```

---

## Time Series Definition

A time series is a sequence of numerical observations ordered in time. It may contain dependence on previous values, trend, seasonality, irregular variation, and uncertainty.

The main time series in this project is:

```text
date → total_duration_min
```

for `Kyiv City`.

Each observation represents the total duration of oblast-level air raid alerts occurring during one local calendar day.

---

## Data Processing Pipeline

```text
Raw event-level air raid alert data
                ↓
Filtering of oblast-level records
                ↓
Timezone conversion to Europe/Kyiv
                ↓
Duplicate removal and duration validation
                ↓
Splitting alerts that cross midnight
                ↓
Daily aggregation by region
                ↓
Continuous daily regional time series
                ↓
Exploratory time-series analysis
                ↓
STL decomposition
                ↓
Forecast feature engineering
                ↓
Seasonal Naive, Ridge, ARIMA and SARIMA forecasting
                ↓
Chronological final test evaluation
                ↓
Rolling-origin backtesting
```

---

## Dataset and Preprocessing

The raw dataset is stored in:

```text
data/raw/official_data_en.csv
```

Only records with:

```text
level == "oblast"
```

are used in the analysis.

The main preprocessing steps are:

* conversion of timestamps to the `Europe/Kyiv` timezone;
* filtering to the project analysis period;
* removal of exact duplicate events;
* validation of alert duration;
* removal of invalid non-positive durations;
* splitting alerts that continue across midnight;
* aggregation into daily regional metrics;
* validation that each regional daily series is continuous.

The main processed dataset contains one row per:

```text
date × region
```

with variables such as:

```text
alert_start_count
active_alert_count
total_duration_min
mean_active_duration_min
```

---

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── official_data_en.csv
│   └── processed/
│       ├── alerts_cleaned.csv
│       ├── daily_region_metrics.csv
│       ├── kyiv_city_forecast_features.csv
│       ├── kyiv_city_forecast_train.csv
│       ├── kyiv_city_forecast_test.csv
│       ├── kyiv_city_acf_pacf_lags.csv
│       ├── kyiv_city_weekday_pattern.csv
│       └── kyiv_city_stl_components.csv
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_forecasting_preparation.ipynb
│
├── reports/
│   ├── forecast_metrics.csv
│   ├── seasonal_naive_test_predictions.csv
│   ├── ridge_test_predictions.csv
│   ├── ridge_feature_coefficients.csv
│   ├── rolling_origin_backtest_metrics.csv
│   ├── rolling_origin_backtest_fold_summary.csv
│   ├── arima_sarima_validation_metrics.csv
│   ├── arima_sarima_test_predictions.csv
│   └── figures/
│       ├── 07_kyiv_city_ridge_vs_baseline.png
│       ├── 08_kyiv_city_daily_duration_with_rolling_means.png
│       ├── 09_kyiv_city_acf_pacf_lags.png
│       ├── 10_kyiv_city_weekday_pattern.png
│       ├── 11_rolling_origin_backtest_mae.png
│       ├── 12_kyiv_city_stl_decomposition.png
│       └── 13_kyiv_city_seasonal_naive_arima_sarima.png
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── aggregation.py
│   ├── visualization.py
│   └── forecasting.py
│
├── tests/
│   ├── conftest.py
│   ├── test_preprocessing.py
│   ├── test_aggregation.py
│   ├── test_visualization.py
│   └── test_forecasting.py
│
├── requirements.txt
└── README.md
```

---

## Notebooks

### `01_data_audit.ipynb`

This notebook prepares the event-level data for time-series analysis.

Main tasks:

* loading the raw CSV dataset;
* checking dataset structure;
* filtering oblast-level events;
* converting timestamps to `Europe/Kyiv`;
* removing exact duplicates;
* validating event duration;
* splitting cross-midnight alerts;
* aggregating events into daily regional metrics.

Main output:

```text
data/processed/daily_region_metrics.csv
```

---

### `02_exploratory_analysis.ipynb`

This notebook investigates historical temporal patterns.

The analysis includes:

* monthly alert-duration comparison across selected regions;
* regional ranking by total alert duration;
* individual alert-duration distributions;
* month-by-hour analysis of alert starts;
* correlation of daily durations between selected regions;
* Kyiv City daily duration with 7-day and 28-day rolling means;
* train-only ACF/PACF diagnostics;
* train-only weekday-pattern analysis;
* STL decomposition of daily total alert duration in Kyiv City.

---

### `03_forecasting_preparation.ipynb`

This notebook builds and evaluates forecasting models.

Main tasks:

* creation of lag, rolling, and calendar features;
* chronological train/test split;
* Seasonal Naive baseline evaluation;
* Ridge Regression with `TimeSeriesSplit`;
* rolling-origin backtesting;
* train-only ARIMA order selection;
* train-only SARIMA order selection;
* one-step-ahead walk-forward forecasting on the final test period;
* comparison of Seasonal Naive, Ridge, ARIMA, and SARIMA.

---

## Exploratory Time Series Analysis

### Daily Time Series and Rolling Means

The Kyiv City daily time series is visualized with:

* daily total alert duration;
* 7-day trailing rolling mean;
* 28-day trailing rolling mean;
* chronological boundary between training and final test periods.

This helps identify short-term variation, longer local changes in the series level, and the final evaluation boundary.

Output:

```text
reports/figures/08_kyiv_city_daily_duration_with_rolling_means.png
```

---

### ACF and PACF Diagnostics

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) were calculated only on the training period.

The final 180-day test period was excluded from lag diagnostics to avoid using future observations when justifying lag features or seasonal structure.

The diagnostics show:

* meaningful short-term dependence at lag 1;
* moderate weekly dependence at lag 7;
* weaker direct dependence at lag 14 after shorter lags are considered.

Output:

```text
data/processed/kyiv_city_acf_pacf_lags.csv
reports/figures/09_kyiv_city_acf_pacf_lags.png
```

---

### Weekday Pattern

The weekday analysis was calculated only on the training period.

The results indicate a moderate, but non-deterministic, weekly pattern. Calendar variables can therefore be useful supporting features, but they cannot explain abrupt changes in alert duration.

Output:

```text
data/processed/kyiv_city_weekday_pattern.csv
reports/figures/10_kyiv_city_weekday_pattern.png
```

---

### STL Decomposition

STL decomposition separates the daily Kyiv City series into:

[
y_t = T_t + S_t + R_t,
]

where:

* (y_t) is the observed daily total alert duration;
* (T_t) is the long-term trend;
* (S_t) is the repeating weekly seasonal component;
* (R_t) is the residual variation not explained by trend or weekly seasonality.

The decomposition uses:

```text
period = 7
seasonal = 13
robust = True
```

The `robust=True` option does not remove long real-world alerts. It reduces the influence of isolated extreme values while estimating trend and seasonal components.

Outputs:

```text
data/processed/kyiv_city_stl_components.csv
reports/figures/12_kyiv_city_stl_decomposition.png
```

---

## Forecasting Setup

### Target Variable

```text
total_duration_min
```

for `Kyiv City`.

### Ridge Regression Features

The Ridge model uses 13 forecasting features.

Lag features:

```text
lag_1
lag_7
lag_14
```

Rolling mean features:

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
```

Calendar features:

```text
day_of_week
is_weekend
month
day_of_week_sin
day_of_week_cos
month_sin
month_cos
```

All rolling means are shifted by one day. Therefore, they use only information available before the forecasted date.

---

## Chronological Train/Test Split

The final evaluation uses a chronological split.

| Dataset part               | Date range               | Purpose                                          |
| -------------------------- | ------------------------ | ------------------------------------------------ |
| Ridge training data        | 2022-04-12 to 2025-06-03 | Feature-based model fitting and cross-validation |
| ARIMA/SARIMA training data | 2022-03-15 to 2025-06-03 | Classical univariate model fitting               |
| Final test data            | 2025-06-04 to 2025-11-30 | Untouched out-of-sample evaluation               |
| Final test size            | 180 days                 | Final model comparison                           |

ARIMA and SARIMA use 28 additional early training days because they model the original daily series directly and do not require a 28-day rolling feature window.

All models are evaluated on the same final 180-day test period.

---

## Forecasting Models

### Seasonal Naive Baseline

The Seasonal Naive model predicts the current day using the value from the same weekday one week earlier:

[
\hat{y}*{t} = y*{t-7}.
]

It is a meaningful baseline because exploratory analysis identified a moderate weekly pattern.

---

### Ridge Regression

Ridge Regression uses lag, rolling, and calendar features.

The regularization parameter was selected using chronological `TimeSeriesSplit`.

Best parameter:

```text
alpha = 1000.0
```

---

### ARIMA

ARIMA models the temporal dependence of the univariate daily duration series.

A small set of candidate orders was evaluated using three expanding validation folds inside the training period only.

The selected model was:

[
\text{ARIMA}(0,1,1).
]

---

### SARIMA

SARIMA extends ARIMA with explicit weekly seasonality:

[
(p,d,q) \times (P,D,Q)_7.
]

The selected seasonal model was:

[
\text{SARIMA}(1,1,1) \times (0,1,1)_7.
]

The seasonal period was:

```text
s = 7 days
```

Both ARIMA and SARIMA were fitted using:

[
z_t = \log(1+y_t),
]

where (y_t) is daily alert duration in minutes.

Forecasts were returned to the original scale using:

[
\hat{y}_t = \exp(\hat{z}_t)-1.
]

---

## Model Selection and Evaluation Protocol

ARIMA and SARIMA orders were selected only inside the training period.

The final 180-day test period was not used:

* for ACF/PACF diagnostics;
* for feature selection;
* for ARIMA order selection;
* for SARIMA order selection;
* for hyperparameter tuning.

Final predictions were created with one-step-ahead walk-forward forecasting:

[
\hat{y}_{t \mid t-1}
====================

f(y_1, y_2, \ldots, y_{t-1}).
]

After each test day, only the newly observed actual value was added to the available historical information before forecasting the next day.

---

## Final Test Results

The untouched final test period was:

```text
2025-06-04 to 2025-11-30
```

with 180 daily observations.

| Model                             |    MAE, min |   RMSE, min |    sMAPE |
| --------------------------------- | ----------: | ----------: | -------: |
| ARIMA(0, 1, 1)                    | **100.338** |     168.680 | 106.441% |
| SARIMA(1, 1, 1) × (0, 1, 1, 7)    |     101.104 |     168.314 | 108.176% |
| Ridge Regression (`alpha=1000.0`) |     111.104 | **150.887** | 106.666% |
| Seasonal Naive (`lag_7`)          |     134.345 |     198.484 | 124.905% |

### Interpretation

* **ARIMA(0,1,1)** achieved the lowest MAE and was the best model according to the main metric of typical daily absolute error.
* SARIMA achieved a very similar result, but weekly seasonal differencing did not improve MAE relative to the simpler ARIMA model.
* Ridge Regression had a higher MAE but the lowest RMSE, suggesting relatively better performance on some large forecast errors.
* Seasonal Naive was clearly weaker than ARIMA, SARIMA, and Ridge Regression.
* MAE is treated as the main metric because percentage metrics become unstable for zero-duration and very low-duration days.

ARIMA reduced MAE relative to Seasonal Naive by approximately:

[
\frac{134.345 - 100.338}{134.345}
\times 100%
\approx 25.3%.
]

---

## Final Forecast Comparison

The final comparison figure includes:

* the last 180 days of the training period;
* actual daily duration values during the final test period;
* Seasonal Naive forecasts;
* ARIMA forecasts;
* SARIMA forecasts;
* a 95% SARIMA prediction interval;
* the boundary between training and final test data.

The wide SARIMA prediction interval is an important result: abrupt changes in air raid alert duration are not reliably predictable using historical duration values and weekly seasonality alone.

Output:

```text
reports/figures/13_kyiv_city_seasonal_naive_arima_sarima.png
```

---

## Rolling-Origin Backtesting

Rolling-origin backtesting evaluates Ridge Regression and Seasonal Naive across multiple historical periods rather than only one final test interval.

Configuration:

```text
Number of historical folds: 3
Test horizon per fold: 120 days
Training window: expanding
Final 180-day test period: excluded
```

Ridge Regression outperformed Seasonal Naive in all three historical folds.

Outputs:

```text
reports/rolling_origin_backtest_metrics.csv
reports/rolling_origin_backtest_fold_summary.csv
reports/figures/11_rolling_origin_backtest_mae.png
```

---

## Key Output Files

```text
data/processed/daily_region_metrics.csv
data/processed/kyiv_city_stl_components.csv

reports/forecast_metrics.csv
reports/arima_sarima_validation_metrics.csv
reports/arima_sarima_test_predictions.csv

reports/figures/12_kyiv_city_stl_decomposition.png
reports/figures/13_kyiv_city_seasonal_naive_arima_sarima.png
```

---

## Installation

Create and activate a virtual environment.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Running the Project

Run the notebooks in this order:

```text
1. notebooks/01_data_audit.ipynb
2. notebooks/02_exploratory_analysis.ipynb
3. notebooks/03_forecasting_preparation.ipynb
```

This order is important because each notebook creates files used by later stages.

---

## Running Tests

From the project root:

```bash
python -m pytest -q
```

The test suite checks:

* preprocessing and duplicate removal;
* cross-midnight alert splitting;
* daily aggregation;
* time-series continuity;
* visualization outputs;
* ACF/PACF diagnostics;
* weekday-pattern calculations;
* Seasonal Naive forecasting;
* Ridge Regression;
* rolling-origin backtesting.

---

## Limitations

This project has important limitations:

* it uses historical alert-duration data only;
* it does not include causes of alerts, military activity, weather, intelligence information, geopolitical events, or operational context;
* the time series is non-stationary and can change abruptly;
* long-duration alert days are real observations and were not automatically removed as outliers;
* relationships observed in the past may not remain stable in the future;
* all forecasts are retrospective statistical estimates;
* the project must not be used for safety, emergency, military, or operational decisions.

---

## Technologies

```text
Python
pandas
NumPy
Matplotlib
scikit-learn
statsmodels
pytest
Jupyter Notebook / JupyterLab
```

---

## Key Methodological Principles

* event-level alerts are converted into a continuous daily time series;
* all final evaluation periods are chronological;
* random train/test splitting is not used;
* future data are excluded from model selection and evaluation;
* ACF/PACF diagnostics use training data only;
* ARIMA and SARIMA orders are selected using train-only expanding validation;
* models are compared with a meaningful Seasonal Naive baseline;
* final forecasts use a one-step-ahead walk-forward protocol;
* conclusions are limited to retrospective statistical performance.
