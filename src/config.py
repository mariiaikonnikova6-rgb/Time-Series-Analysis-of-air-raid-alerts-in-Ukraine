from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# src/config.py -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"


KYIV_CITY_ACF_PACF_LAGS_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_acf_pacf_lags.csv"
)

KYIV_CITY_ACF_PACF_FIGURE = (
    FIGURES_DIR
    / "09_kyiv_city_acf_pacf_lags.png"
)


KYIV_CITY_WEEKDAY_PATTERN_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_weekday_pattern.csv"
)

KYIV_CITY_WEEKDAY_PATTERN_FIGURE = (
    FIGURES_DIR
    / "10_kyiv_city_weekday_pattern.png"
)

# ---------------------------------------------------------------------
# Input / output files
# ---------------------------------------------------------------------

RAW_ALERTS_FILE = RAW_DATA_DIR / "official_data_en.csv"

CLEANED_ALERTS_FILE = PROCESSED_DATA_DIR / "alerts_cleaned.csv"
DAILY_REGION_METRICS_FILE = PROCESSED_DATA_DIR / "daily_region_metrics.csv"
FORECAST_TEST_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_forecast_test.csv"
)


FORECAST_FEATURES_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_forecast_features.csv"
)

DATA_QUALITY_REPORT_FILE = PROCESSED_DATA_DIR / "data_quality_report.csv"

FORECAST_METRICS_FILE = REPORTS_DIR / "forecast_metrics.csv"


# ---------------------------------------------------------------------
# Dataset and analysis settings
# ---------------------------------------------------------------------

# The raw CSV stores timestamps in UTC.
RAW_TIMEZONE = "UTC"

# All calendar-based analysis and visualizations will use Kyiv local time.
ANALYSIS_TIMEZONE = "Europe/Kyiv"

# We analyse only oblast-level records.
TARGET_GEOGRAPHIC_LEVEL = "oblast"

# Use a stable oblast-level period for the MVP.
ANALYSIS_START_DATE = "2022-03-15"
ANALYSIS_END_DATE = "2025-11-30"


# ---------------------------------------------------------------------
# Regions for analysis
# ---------------------------------------------------------------------

# Historical forecasting target.
# Important: Kyiv City and Kyivska oblast are separate regions.
FORECAST_REGION = "Kyiv City"


FORECAST_LAGS = [1, 7, 14]

FORECAST_ROLLING_WINDOWS = [7, 14, 28]
# Five regions for line plots and distribution comparisons.
SELECTED_REGIONS = [
    "Kyiv City",
    "Lvivska oblast",
    "Odeska oblast",
    "Kharkivska oblast",
    "Dnipropetrovska oblast",
]

ROLLING_ORIGIN_N_FOLDS = 3
ROLLING_ORIGIN_TEST_SIZE_DAYS = 120


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

RANDOM_STATE = 42



ROLLING_ORIGIN_BACKTEST_METRICS_FILE = (
    REPORTS_DIR
    / "rolling_origin_backtest_metrics.csv"
)

ROLLING_ORIGIN_BACKTEST_FOLD_SUMMARY_FILE = (
    REPORTS_DIR
    / "rolling_origin_backtest_fold_summary.csv"
)

ROLLING_ORIGIN_BACKTEST_FIGURE = (
    FIGURES_DIR
    / "11_rolling_origin_backtest_mae.png"
)



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