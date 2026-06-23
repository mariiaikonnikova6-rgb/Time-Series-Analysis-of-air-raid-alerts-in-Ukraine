from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_RAW_COLUMNS = [
    "oblast",
    "level",
    "started_at",
    "finished_at",
    "source",
]

EVENT_KEY_COLUMNS = [
    "region",
    "started_at_utc",
    "finished_at_utc",
    "source",
]


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Check that a DataFrame contains all required columns.

    Parameters
    ----------
    dataframe:
        Input pandas DataFrame.
    required_columns:
        List of required column names.

    Raises
    ------
    KeyError:
        If one or more required columns are missing.
    """
    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            "The input DataFrame is missing required columns: "
            f"{missing_columns}. "
            f"Available columns: {dataframe.columns.tolist()}"
        )


# ---------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------

def preprocess_oblast_alert_events(
    raw_dataframe: pd.DataFrame,
    target_geographic_level: str,
    analysis_start_date: str,
    analysis_end_date: str,
    analysis_timezone: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a cleaned oblast-level dataset of air-raid alert events.

    Processing steps
    ----------------
    1. Keep only records on the requested geographic level.
    2. Rename 'oblast' to 'region'.
    3. Parse timestamps as UTC.
    4. Convert timestamps to the selected local timezone.
    5. Calculate event duration in minutes.
    6. Restrict records to the selected historical analysis period.
    7. Remove rows with invalid timestamps or non-positive durations.
    8. Remove exact duplicate events.

    Parameters
    ----------
    raw_dataframe:
        Raw dataset loaded from official_data_en.csv.
    target_geographic_level:
        Geographic level to keep, for example 'oblast'.
    analysis_start_date:
        First included local calendar date in YYYY-MM-DD format.
    analysis_end_date:
        Last included local calendar date in YYYY-MM-DD format.
    analysis_timezone:
        Timezone used for calendar-based analysis, for example
        'Europe/Kyiv'.

    Returns
    -------
    cleaned_dataframe:
        Cleaned event-level DataFrame.
    quality_report:
        DataFrame with processing and data-quality metrics.
    """
    validate_required_columns(
        dataframe=raw_dataframe,
        required_columns=REQUIRED_RAW_COLUMNS,
    )

    raw_rows = len(raw_dataframe)

    # -------------------------------------------------------------
    # Keep only oblast-level rows and standardize the region column.
    # -------------------------------------------------------------

    working_dataframe = (
        raw_dataframe.loc[
            raw_dataframe["level"].eq(target_geographic_level),
            [
                "oblast",
                "level",
                "started_at",
                "finished_at",
                "source",
            ],
        ]
        .copy()
        .rename(columns={"oblast": "region"})
    )

    rows_after_level_filter = len(working_dataframe)

    # -------------------------------------------------------------
    # Parse UTC timestamps and create local-time columns.
    # -------------------------------------------------------------

    working_dataframe["started_at_utc"] = pd.to_datetime(
        working_dataframe["started_at"],
        utc=True,
        errors="coerce",
    )

    working_dataframe["finished_at_utc"] = pd.to_datetime(
        working_dataframe["finished_at"],
        utc=True,
        errors="coerce",
    )

    invalid_timestamp_mask = (
        working_dataframe["started_at_utc"].isna()
        | working_dataframe["finished_at_utc"].isna()
    )

    invalid_timestamp_rows_removed = int(invalid_timestamp_mask.sum())

    working_dataframe = working_dataframe.loc[
        ~invalid_timestamp_mask
    ].copy()

    working_dataframe["started_at_local"] = (
        working_dataframe["started_at_utc"]
        .dt.tz_convert(analysis_timezone)
    )

    working_dataframe["finished_at_local"] = (
        working_dataframe["finished_at_utc"]
        .dt.tz_convert(analysis_timezone)
    )

    # -------------------------------------------------------------
    # Calculate event duration.
    # -------------------------------------------------------------

    working_dataframe["duration_min"] = (
        working_dataframe["finished_at_utc"]
        - working_dataframe["started_at_utc"]
    ).dt.total_seconds() / 60

    # -------------------------------------------------------------
    # Filter selected historical period.
    # The upper bound is exclusive, so the entire final date is included.
    # -------------------------------------------------------------

    analysis_start = pd.Timestamp(
        analysis_start_date,
        tz=analysis_timezone,
    )

    analysis_end_exclusive = (
        pd.Timestamp(
            analysis_end_date,
            tz=analysis_timezone,
        )
        + pd.Timedelta(days=1)
    )

    period_mask = (
        working_dataframe["started_at_local"].ge(analysis_start)
        & working_dataframe["started_at_local"].lt(analysis_end_exclusive)
    )

    working_dataframe = working_dataframe.loc[period_mask].copy()

    rows_in_analysis_period = len(working_dataframe)

    # -------------------------------------------------------------
    # Remove records with non-positive duration.
    # -------------------------------------------------------------

    invalid_duration_mask = working_dataframe["duration_min"].le(0)

    nonpositive_duration_rows_removed = int(
        invalid_duration_mask.sum()
    )

    working_dataframe = working_dataframe.loc[
        ~invalid_duration_mask
    ].copy()

    rows_before_deduplication = len(working_dataframe)

    # -------------------------------------------------------------
    # Remove technical duplicates.
    # Each real oblast-level event occurs twice in this dataset.
    # -------------------------------------------------------------

    cleaned_dataframe = (
        working_dataframe
        .drop_duplicates(
            subset=EVENT_KEY_COLUMNS,
            keep="first",
        )
        .sort_values(
            by=[
                "region",
                "started_at_utc",
                "finished_at_utc",
            ],
        )
        .reset_index(drop=True)
    )

    rows_after_deduplication = len(cleaned_dataframe)

    rows_removed_as_duplicates = (
        rows_before_deduplication
        - rows_after_deduplication
    )

    remaining_duplicate_rows = int(
        cleaned_dataframe.duplicated(
            subset=EVENT_KEY_COLUMNS,
            keep=False,
        ).sum()
    )

    # -------------------------------------------------------------
    # Create a compact data-quality report.
    # -------------------------------------------------------------

    quality_report = pd.DataFrame(
        {
            "metric": [
                "raw_rows",
                "rows_after_level_filter",
                "invalid_timestamp_rows_removed",
                "rows_in_analysis_period",
                "nonpositive_duration_rows_removed",
                "rows_before_deduplication",
                "rows_removed_as_duplicates",
                "rows_after_deduplication",
                "remaining_duplicate_rows",
                "unique_regions_after_cleaning",
            ],
            "value": [
                raw_rows,
                rows_after_level_filter,
                invalid_timestamp_rows_removed,
                rows_in_analysis_period,
                nonpositive_duration_rows_removed,
                rows_before_deduplication,
                rows_removed_as_duplicates,
                rows_after_deduplication,
                remaining_duplicate_rows,
                cleaned_dataframe["region"].nunique(),
            ],
        }
    )

    return cleaned_dataframe, quality_report