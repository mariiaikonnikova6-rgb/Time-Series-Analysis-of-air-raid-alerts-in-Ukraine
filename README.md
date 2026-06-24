# Historical Time Series Analysis and Forecasting of Air Raid Alert Duration in Ukraine

## Overview

This project performs historical time series analysis and one-step-ahead forecasting of daily air raid alert duration in Ukraine.

The workflow transforms event-level air raid alert records into continuous daily regional time series, investigates temporal patterns, and evaluates forecasting models for **Kyiv City**.

The project is educational and analytical. It is designed to demonstrate reproducible time series preprocessing, exploratory data analysis, lag diagnostics, feature engineering, forecasting, and backtesting.

> **Important:** This project is not a real-time warning system and must not be used for safety, operational, military, or personal decision-making.

---

## Research Goal

The main goal is to answer the following question:

> Can the daily total duration of air raid alerts in Kyiv City be forecast more accurately using historical time-series features than with a simple weekly baseline?

The forecast target is:

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

A time series is a sequence of numerical observations ordered in time, where trends, seasonality, dependence on previous values, and future forecasting are important.

In this project, the main time series is:

```text
date → total_duration_min
```

for `Kyiv City`.

Each observation represents the total duration of oblast-level air raid alerts during one calendar day.

---

## Data Processing Pipeline

```text
Raw event-level alert data
        ↓
Filtering of oblast-level records
        ↓
Timezone conversion to Europe/Kyiv
        ↓
Duplicate removal and duration validation
        ↓
Splitting events that cross midnight
        ↓
Daily aggregation by region
        ↓
Continuous daily regional time series
        ↓
Exploratory time-series analysis
        ↓
Forecast feature engineering
        ↓
Baseline and Ridge Regression forecasting
        ↓
Final test evaluation and rolling-origin backtesting
```

---

## Dataset and Preprocessing

The raw dataset is stored in:

```text
data/raw/official_data_en.csv
```

The analysis uses only records with:

```text
level == "oblast"
```

The main preprocessing steps are:

* conversion of timestamps to the `Europe/Kyiv` timezone;
* filtering to the project analysis period;
* removal of exact duplicate events;
* removal of invalid non-positive durations;
* splitting of alerts that continue across midnight;
* aggregation into daily regional metrics;
* validation that the daily series is continuous.

The processed daily dataset contains one row per:

```text
date × region
```

with variables such as:

* `alert_start_count`;
* `active_alert_count`;
* `total_duration_min`;
* `mean_active_duration_min`.

---

## Project Structure

```text
PythonProject2/
│
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
│       └── kyiv_city_weekday_pattern.csv
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   └── 03_forecasting_preparation.ipynb
│
├── reports/
│   ├── forecast_metrics.csv
│   ├── ridge_test_predictions.csv
│   ├── ridge_feature_coefficients.csv
│   ├── rolling_origin_backtest_metrics.csv
│   ├── rolling_origin_backtest_fold_summary.csv
│   └── figures/
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

This notebook prepares the data for analysis.

Main tasks:

* loading the raw CSV file;
* checking the dataset structure;
* filtering oblast-level events;
* converting timestamps to the local analysis timezone;
* removing duplicates;
* validating durations;
* creating cleaned event-level data;
* splitting cross-midnight alerts;
* aggregating events into daily regional metrics.

Main output:

```text
data/processed/daily_region_metrics.csv
```

---

### `02_exploratory_analysis.ipynb`

This notebook investigates the historical structure of the time series.

The analysis includes:

* monthly alert-duration comparison across selected regions;
* regional ranking by total alert duration;
* individual alert-duration distributions;
* month-by-hour analysis of alert starts;
* correlation of daily durations between selected regions;
* Kyiv City daily time series with 7-day and 28-day trailing rolling means;
* train-only ACF/PACF lag diagnostics;
* train-only weekday pattern analysis.

This notebook explains why the project is a time series analysis rather than a standard regression task.

---

### `03_forecasting_preparation.ipynb`

This notebook builds and evaluates forecasting models.

Main tasks:

* creation of lag, rolling, and calendar features;
* chronological train/test split;
* Seasonal Naive baseline evaluation;
* Ridge Regression training;
* hyperparameter selection with `TimeSeriesSplit`;
* final test evaluation;
* rolling-origin backtesting across historical periods.

---

## Exploratory Time Series Analysis

### Daily Time Series and Rolling Means

The Kyiv City series is visualized as:

* daily total alert duration;
* 7-day trailing rolling mean;
* 28-day trailing rolling mean;
* chronological boundary between training and final test periods.

This helps reveal short-term fluctuations, longer changes in the local level of the series, and the location of the final evaluation period.

Output figure:

```text
reports/figures/08_kyiv_city_daily_duration_with_rolling_means.png
```

---

### ACF and PACF Diagnostics

Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) were calculated using only the training period.

The final 180-day test period was excluded from lag diagnostics to avoid using future observations during feature justification.

Important training-period results:

| Lag |   ACF |  PACF | Interpretation                                                       |
| --: | ----: | ----: | -------------------------------------------------------------------- |
|   1 | 0.327 | 0.327 | Strong direct short-term dependence                                  |
|   7 | 0.266 | 0.116 | Moderate weekly pattern                                              |
|  14 | 0.152 | 0.002 | Indirect relationship; little direct contribution after shorter lags |

Interpretation:

* `lag_1` is justified because the preceding day contains useful information;
* `lag_7` is justified because the same weekday one week earlier contains a moderate signal;
* `lag_14` is retained as an additional feature, while Ridge regularization can reduce its influence when it contributes little.

Output files:

```text
data/processed/kyiv_city_acf_pacf_lags.csv
reports/figures/09_kyiv_city_acf_pacf_lags.png
```

---

### Weekday Pattern

The weekday analysis was also calculated only on the training period.

The results indicate a moderate, but not deterministic, weekly pattern:

* Thursday had the highest mean and median daily duration;
* Sunday had the lowest median duration and the largest share of zero-duration days;
* Friday had the smallest number of zero-duration days;
* standard deviations were large for every weekday.

Therefore, calendar features are useful as supporting signals, but they are weaker than recent historical values such as `lag_1` and `lag_7`.

Output files:

```text
data/processed/kyiv_city_weekday_pattern.csv
reports/figures/10_kyiv_city_weekday_pattern.png
```

---

## Forecasting Setup

### Target

```text
total_duration_min
```

for `Kyiv City`.

### Forecast Features

The Ridge Regression model uses 13 features.

#### Lag features

```text
lag_1
lag_7
lag_14
```

#### Rolling mean features

```text
rolling_mean_7
rolling_mean_14
rolling_mean_28
```

Forecasting rolling means are shifted by one day, so they use only information known before the forecasted date.

#### Calendar features

```text
day_of_week
is_weekend
month
day_of_week_sin
day_of_week_cos
month_sin
month_cos
```

---

## Chronological Train/Test Split

The final evaluation uses a chronological split.

| Dataset part    | Date range               | Purpose                                        |
| --------------- | ------------------------ | ---------------------------------------------- |
| Training data   | 2022-04-12 to 2025-06-03 | Model fitting and time-series cross-validation |
| Final test data | 2025-06-04 to 2025-11-30 | Untouched final evaluation                     |
| Final test size | 180 days                 | Out-of-sample performance measurement          |

The split is chronological rather than random because forecasting must not use future observations when predicting earlier dates.

---

## Models

### Seasonal Naive Baseline

The baseline model uses the value from the same weekday one week earlier:

[
\hat{y}*{t} = y*{t-7}
]

This baseline is meaningful because ACF/PACF diagnostics identified a moderate weekly pattern.

---

### Ridge Regression

Ridge Regression combines lag features, rolling means, and calendar features.

It uses `StandardScaler` before Ridge Regression and selects the regularization strength through chronological cross-validation.

The best regularization parameter on the final training data was:

```text
alpha = 1000.0
```

A strong Ridge penalty helps reduce overfitting to highly variable historical observations.

---

## Final Test Results

The final untouched test period covered:

```text
2025-06-04 to 2025-11-30
```

with 180 daily observations.

| Model                             | MAE, min | RMSE, min |    sMAPE |
| --------------------------------- | -------: | --------: | -------: |
| Ridge Regression (`alpha=1000.0`) |  111.104 |   150.887 | 106.666% |
| Seasonal Naive (`lag_7`)          |  134.345 |   198.484 | 124.905% |

Ridge Regression improved over the Seasonal Naive baseline:

* MAE reduction: approximately 17.3%;
* RMSE reduction: approximately 24.0%;
* no negative Ridge predictions were produced.

The primary comparison metric is MAE because percentage metrics can become unstable on zero-duration or very low-duration days.

Output files:

```text
reports/forecast_metrics.csv
reports/ridge_test_predictions.csv
reports/ridge_feature_coefficients.csv
reports/figures/07_kyiv_city_ridge_vs_baseline.png
```

---

## Rolling-Origin Backtesting

Rolling-origin backtesting was used to test whether the Ridge model performs better than the baseline across several historical periods, not only in the final 180-day test interval.

Configuration:

```text
Number of historical folds: 3
Test horizon per fold: 120 days
Training window: expanding
Final 180-day test period: excluded completely
```

### MAE Results by Historical Fold

| Fold | Ridge MAE, min | Seasonal Naive MAE, min | Ridge improvement |
| ---: | -------------: | ----------------------: | ----------------: |
|    1 |         73.380 |                  96.699 | 23.319 min, 24.1% |
|    2 |        102.195 |                 139.301 | 37.105 min, 26.6% |
|    3 |        102.060 |                 130.579 | 28.519 min, 21.8% |

### Average Backtesting Performance

| Model            | Mean MAE, min | Mean RMSE, min | Mean sMAPE |
| ---------------- | ------------: | -------------: | ---------: |
| Ridge Regression |        92.545 |        131.780 |    91.343% |
| Seasonal Naive   |       122.193 |        167.836 |   112.823% |

Ridge Regression outperformed Seasonal Naive in all three historical folds.

This indicates that its improvement is not limited to one final test period. However, the absolute error level changes between periods, which is expected for a non-stationary series affected by changing external conditions.

Output files:

```text
reports/rolling_origin_backtest_metrics.csv
reports/rolling_origin_backtest_fold_summary.csv
reports/figures/11_rolling_origin_backtest_mae.png
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
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

This order is important because each stage creates processed files required by the following stage.

---

## Running Tests

From the project root directory:

```bash
python -m pytest -q
```

The test suite checks:

* preprocessing and duplicate removal;
* daily aggregation and cross-midnight splitting;
* visualization outputs;
* time-series continuity checks;
* ACF/PACF diagnostics;
* weekday-pattern calculations;
* Seasonal Naive evaluation;
* Ridge Regression;
* rolling-origin backtesting;
* saved report figures.

---

## Limitations

This project has important limitations:

* it uses historical alert-duration data only;
* it does not include causes of alerts, military activity, weather, intelligence information, or geopolitical events;
* air raid alert dynamics are non-stationary and can change abruptly;
* relationships found in past data may not remain stable in future periods;
* the model produces a retrospective statistical forecast, not a reliable real-time warning;
* the model must not be used for personal safety decisions, emergency planning, or operational use.

The results should be interpreted as an educational demonstration of time series analysis and forecasting methodology.

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
Jupyter Notebook
```

---

## Key Methodological Principles

* event-level data are converted into a continuous daily time series;
* all final evaluation periods are chronological;
* future data are excluded from feature justification and training;
* ACF/PACF diagnostics are calculated on training data only;
* the Ridge model is compared against a meaningful Seasonal Naive baseline;
* model stability is checked through rolling-origin backtesting;
* conclusions are limited to retrospective statistical performance.
