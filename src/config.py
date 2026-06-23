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

# ---------------------------------------------------------------------
# Input / output files
# ---------------------------------------------------------------------

RAW_ALERTS_FILE = RAW_DATA_DIR / "official_data_en.csv"

CLEANED_ALERTS_FILE = PROCESSED_DATA_DIR / "alerts_cleaned.csv"
DAILY_REGION_METRICS_FILE = PROCESSED_DATA_DIR / "daily_region_metrics.csv"
DATA_QUALITY_REPORT_FILE = PROCESSED_DATA_DIR / "data_quality_report.csv"

FORECAST_METRICS_FILE = REPORTS_DIR / "forecast_metrics.csv"

RIDGE_TEST_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "ridge_test_predictions.csv"
)

RIDGE_FEATURE_COEFFICIENTS_FILE = (
    REPORTS_DIR
    / "ridge_feature_coefficients.csv"
)

# ---------------------------------------------------------------------
# Analysis settings
# ---------------------------------------------------------------------

ANALYSIS_START_DATE = "2022-03-15"
ANALYSIS_END_DATE = "2025-11-30"

TARGET_GEOGRAPHIC_LEVEL = "oblast"

ANALYSIS_TIMEZONE = "Europe/Kyiv"

# This is only a preliminary value.
# We will verify the exact region name after the first data audit.
FORECAST_REGION = "Kyiv City"

# Regions planned for visual comparison.
# Exact names will be checked and adjusted after the audit.
SELECTED_REGIONS = [
    "Kyiv City",
    "Lvivska oblast",
    "Odeska oblast",
    "Kharkivska oblast",
    "Dnipropetrovska oblast",
]

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

FORECAST_TARGET_COLUMN = "total_duration_min"

FORECAST_LAGS = [1, 7, 14]

FORECAST_ROLLING_WINDOWS = [7, 14, 28]

FORECAST_TEST_SIZE_DAYS = 180



RANDOM_STATE = 42

CLEANED_ALERTS_FILE = PROCESSED_DATA_DIR / "alerts_cleaned.csv"
DAILY_REGION_METRICS_FILE = PROCESSED_DATA_DIR / "daily_region_metrics.csv"
DATA_QUALITY_REPORT_FILE = PROCESSED_DATA_DIR / "data_quality_report.csv"

DAILY_AGGREGATION_QUALITY_REPORT_FILE = (
    PROCESSED_DATA_DIR / "daily_aggregation_quality_report.csv"
)

FORECAST_METRICS_FILE = REPORTS_DIR / "forecast_metrics.csv"



FORECAST_FEATURES_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_forecast_features.csv"
)

FORECAST_TRAIN_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_forecast_train.csv"
)

FORECAST_TEST_FILE = (
    PROCESSED_DATA_DIR
    / "kyiv_city_forecast_test.csv"
)


RIDGE_ALPHA_CANDIDATES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

RIDGE_TIME_SERIES_SPLITS = 5

RIDGE_ALPHA_CANDIDATES = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

RIDGE_TIME_SERIES_SPLITS = 5



