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

# Five regions for line plots and distribution comparisons.
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

RANDOM_STATE = 42