from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# src/config.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


# ---------------------------------------------------------------------
# Input and processed data files
# ---------------------------------------------------------------------

RAW_ALERTS_FILE = RAW_DATA_DIR / "official_data_en.csv"

CLEANED_ALERTS_FILE = PROCESSED_DATA_DIR / "alerts_cleaned.csv"
DAILY_REGION_METRICS_FILE = PROCESSED_DATA_DIR / "daily_region_metrics.csv"
DATA_QUALITY_REPORT_FILE = PROCESSED_DATA_DIR / "data_quality_report.csv"

KYIV_CITY_FORECAST_FEATURES_FILE = (
    PROCESSED_DATA_DIR / "kyiv_city_forecast_features.csv"
)

KYIV_CITY_FORECAST_TRAIN_FILE = (
    PROCESSED_DATA_DIR / "kyiv_city_forecast_train.csv"
)

KYIV_CITY_FORECAST_TEST_FILE = (
    PROCESSED_DATA_DIR / "kyiv_city_forecast_test.csv"
)

# Names already used in the forecasting notebook.
FORECAST_FEATURES_FILE = KYIV_CITY_FORECAST_FEATURES_FILE
FORECAST_TRAIN_FILE = KYIV_CITY_FORECAST_TRAIN_FILE
FORECAST_TEST_FILE = KYIV_CITY_FORECAST_TEST_FILE


# ---------------------------------------------------------------------
# Exploratory-analysis outputs
# ---------------------------------------------------------------------

KYIV_CITY_ACF_PACF_LAGS_FILE = (
    PROCESSED_DATA_DIR / "kyiv_city_acf_pacf_lags.csv"
)

KYIV_CITY_WEEKDAY_PATTERN_FILE = (
    PROCESSED_DATA_DIR / "kyiv_city_weekday_pattern.csv"
)

KYIV_CITY_STL_COMPONENTS_FILE = (
    PROCESSED_DATA_DIR / "kyiv_city_stl_components.csv"
)


# ---------------------------------------------------------------------
# Forecasting report files
# ---------------------------------------------------------------------

FORECAST_METRICS_FILE = REPORTS_DIR / "forecast_metrics.csv"

SEASONAL_NAIVE_TEST_PREDICTIONS_FILE = (
    REPORTS_DIR / "seasonal_naive_test_predictions.csv"
)

RIDGE_TEST_PREDICTIONS_FILE = (
    REPORTS_DIR / "ridge_test_predictions.csv"
)

RIDGE_FEATURE_COEFFICIENTS_FILE = (
    REPORTS_DIR / "ridge_feature_coefficients.csv"
)

ROLLING_ORIGIN_BACKTEST_METRICS_FILE = (
    REPORTS_DIR / "rolling_origin_backtest_metrics.csv"
)

ROLLING_ORIGIN_BACKTEST_FOLD_SUMMARY_FILE = (
    REPORTS_DIR / "rolling_origin_backtest_fold_summary.csv"
)

ARIMA_SARIMA_VALIDATION_METRICS_FILE = (
    REPORTS_DIR / "arima_sarima_validation_metrics.csv"
)

ARIMA_SARIMA_TEST_PREDICTIONS_FILE = (
    REPORTS_DIR / "arima_sarima_test_predictions.csv"
)


# ---------------------------------------------------------------------
# Figure paths
# ---------------------------------------------------------------------

KYIV_CITY_ROLLING_MEANS_FIGURE = (
    FIGURES_DIR / "08_kyiv_city_daily_duration_with_rolling_means.png"
)

KYIV_CITY_ACF_PACF_FIGURE = (
    FIGURES_DIR / "09_kyiv_city_acf_pacf_lags.png"
)

KYIV_CITY_WEEKDAY_PATTERN_FIGURE = (
    FIGURES_DIR / "10_kyiv_city_weekday_pattern.png"
)

ROLLING_ORIGIN_BACKTEST_FIGURE = (
    FIGURES_DIR / "11_rolling_origin_backtest_mae.png"
)

KYIV_CITY_STL_DECOMPOSITION_FIGURE = (
    FIGURES_DIR / "12_kyiv_city_stl_decomposition.png"
)

KYIV_CITY_FORECAST_COMPARISON_FIGURE = (
    FIGURES_DIR / "13_kyiv_city_seasonal_naive_arima_sarima.png"
)


# ---------------------------------------------------------------------
# Dataset and analysis settings
# ---------------------------------------------------------------------

RAW_TIMEZONE = "UTC"
ANALYSIS_TIMEZONE = "Europe/Kyiv"

TARGET_GEOGRAPHIC_LEVEL = "oblast"

ANALYSIS_START_DATE = "2022-03-15"
ANALYSIS_END_DATE = "2025-11-30"

SELECTED_REGIONS = [
    "Kyiv City",
    "Lvivska oblast",
    "Odeska oblast",
    "Kharkivska oblast",
    "Dnipropetrovska oblast",
]


# ---------------------------------------------------------------------
# Forecasting settings
# ---------------------------------------------------------------------

FORECAST_REGION = "Kyiv City"
FORECAST_TARGET_COLUMN = "total_duration_min"

FORECAST_LAGS = [1, 7, 14]
FORECAST_ROLLING_WINDOWS = [7, 14, 28]

FORECAST_TEST_SIZE_DAYS = 180

RIDGE_ALPHA_CANDIDATES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

RIDGE_TIME_SERIES_SPLITS = 5

ROLLING_ORIGIN_N_FOLDS = 3
ROLLING_ORIGIN_TEST_SIZE_DAYS = 120


# ---------------------------------------------------------------------
# Classical time-series model settings
# ---------------------------------------------------------------------

STL_PERIOD = 7
STL_SEASONAL = 13
STL_ROBUST = True

ARIMA_ORDER_CANDIDATES = [
    (0, 0, 0),
    (1, 0, 0),
    (0, 0, 1),
    (1, 0, 1),
    (2, 0, 0),
    (0, 1, 1),
    (1, 1, 0),
    (1, 1, 1),
]

SARIMA_ORDER_CANDIDATES = [
    ((0, 0, 0), (0, 1, 1, 7)),
    ((1, 0, 0), (0, 1, 1, 7)),
    ((0, 0, 1), (0, 1, 1, 7)),
    ((1, 0, 1), (0, 1, 1, 7)),
    ((1, 1, 0), (0, 1, 1, 7)),
    ((1, 1, 1), (0, 1, 1, 7)),
]

CLASSICAL_MODEL_VALIDATION_FOLDS = 3
CLASSICAL_MODEL_VALIDATION_SIZE_DAYS = 120

USE_LOG1P_TRANSFORM = True


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_STATE = 42