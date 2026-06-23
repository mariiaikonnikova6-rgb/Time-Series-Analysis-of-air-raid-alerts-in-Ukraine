from __future__ import annotations

from datetime import timedelta

import pandas as pd


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_CLEANED_COLUMNS = [
    "region",
    "started_at_local",
    "finished_at_local",
    "duration_min",
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
# Event splitting
# ---------------------------------------------------------------------

def split_event_into_daily_segments(
    event_id: int,
    region: str,
    started_at_local: pd.Timestamp,
    finished_at_local: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end_exclusive: pd.Timestamp,
    timezone_name: str,
) -> list[dict]:
    """
    Split one alert event into one or more local calendar-day segments.

    An event that spans midnight is divided between the relevant dates.
    Before splitting, the event is clipped to the selected analysis period.

    Parameters
    ----------
    event_id:
        Stable identifier of the event.
    region:
        Region name.
    started_at_local:
        Event start timestamp in local timezone.
    finished_at_local:
        Event finish timestamp in local timezone.
    period_start:
        Inclusive beginning of the analysis period.
    period_end_exclusive:
        Exclusive end of the analysis period.
    timezone_name:
        Timezone name, for example 'Europe/Kyiv'.

    Returns
    -------
    list[dict]
        One dictionary per local calendar-day segment.
    """
    clipped_start = max(started_at_local, period_start)
    clipped_end = min(finished_at_local, period_end_exclusive)

    if clipped_end <= clipped_start:
        return []

    segment_rows = []
    current_date = clipped_start.date()

    while True:
        day_start = pd.Timestamp(
            current_date,
            tz=timezone_name,
        )

        next_day_start = pd.Timestamp(
            current_date + timedelta(days=1),
            tz=timezone_name,
        )

        segment_start = max(clipped_start, day_start)
        segment_end = min(clipped_end, next_day_start)

        if segment_end > segment_start:
            segment_duration_min = (
                segment_end - segment_start
            ).total_seconds() / 60

            segment_rows.append(
                {
                    "event_id": event_id,
                    "region": region,
                    "date": pd.Timestamp(current_date),
                    "segment_start_local": segment_start,
                    "segment_end_local": segment_end,
                    "segment_duration_min": segment_duration_min,
                }
            )

        if next_day_start >= clipped_end:
            break

        current_date = current_date + timedelta(days=1)

    return segment_rows


# ---------------------------------------------------------------------
# Main aggregation function
# ---------------------------------------------------------------------

def aggregate_alert_events_daily(
    cleaned_dataframe: pd.DataFrame,
    analysis_start_date: str,
    analysis_end_date: str,
    analysis_timezone: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert cleaned event-level alert data into daily region-level metrics.

    Processing steps
    ----------------
    1. Clip every event to the selected analysis period.
    2. Split events that cross local midnight.
    3. Aggregate total alert duration by date and region.
    4. Count alert starts and active events per date and region.
    5. Create a complete date × region grid with zero-filled missing days.
    6. Build a compact quality report.

    Parameters
    ----------
    cleaned_dataframe:
        Output of preprocess_oblast_alert_events().
    analysis_start_date:
        First included local calendar date in YYYY-MM-DD format.
    analysis_end_date:
        Last included local calendar date in YYYY-MM-DD format.
    analysis_timezone:
        Timezone used for calendar-based analysis.

    Returns
    -------
    daily_metrics:
        Daily metrics for each region.
    quality_report:
        Aggregation checks and summary values.
    """
    validate_required_columns(
        dataframe=cleaned_dataframe,
        required_columns=REQUIRED_CLEANED_COLUMNS,
    )

    events = (
        cleaned_dataframe[
            [
                "region",
                "started_at_local",
                "finished_at_local",
                "duration_min",
            ]
        ]
        .copy()
        .reset_index(drop=True)
    )

    if events.empty:
        raise ValueError(
            "The cleaned DataFrame is empty. "
            "Daily aggregation cannot be performed."
        )

    events.insert(
        loc=0,
        column="event_id",
        value=events.index,
    )

    analysis_start_local = pd.Timestamp(
        analysis_start_date,
        tz=analysis_timezone,
    )

    analysis_end_exclusive_local = (
        pd.Timestamp(
            analysis_end_date,
            tz=analysis_timezone,
        )
        + pd.DateOffset(days=1)
    )

    cross_midnight_events = int(
        (
            events["started_at_local"].dt.date
            != events["finished_at_local"].dt.date
        ).sum()
    )

    # -------------------------------------------------------------
    # Split all events into local daily segments.
    # -------------------------------------------------------------

    all_segments = []

    for event in events.itertuples(index=False):
        event_segments = split_event_into_daily_segments(
            event_id=event.event_id,
            region=event.region,
            started_at_local=event.started_at_local,
            finished_at_local=event.finished_at_local,
            period_start=analysis_start_local,
            period_end_exclusive=analysis_end_exclusive_local,
            timezone_name=analysis_timezone,
        )

        all_segments.extend(event_segments)

    event_day_segments = pd.DataFrame(all_segments)

    if event_day_segments.empty:
        raise ValueError(
            "No event-day segments were created. "
            "Check the analysis period and input timestamps."
        )

    # -------------------------------------------------------------
    # Aggregate duration and active events by date and region.
    # -------------------------------------------------------------

    daily_duration_metrics = (
        event_day_segments
        .groupby(
            ["date", "region"],
            as_index=False,
        )
        .agg(
            total_duration_min=(
                "segment_duration_min",
                "sum",
            ),
            active_alert_count=(
                "event_id",
                "nunique",
            ),
        )
    )

    # Count events that began on each local day.
    # It differs from active_alert_count because an alert can continue
    # after midnight despite starting on the previous date.
    daily_start_metrics = (
        events.loc[
            events["started_at_local"].ge(analysis_start_local)
            & events["started_at_local"].lt(
                analysis_end_exclusive_local
            )
        ]
        .assign(
            date=lambda dataframe: (
                dataframe["started_at_local"]
                .dt.tz_localize(None)
                .dt.normalize()
            )
        )
        .groupby(
            ["date", "region"],
            as_index=False,
        )
        .agg(
            alert_start_count=(
                "event_id",
                "nunique",
            )
        )
    )

    # -------------------------------------------------------------
    # Build a complete date × region grid.
    # Zero-alert days must exist explicitly for time-series analysis.
    # -------------------------------------------------------------

    analysis_dates = pd.date_range(
        start=pd.Timestamp(analysis_start_date),
        end=pd.Timestamp(analysis_end_date),
        freq="D",
    )

    all_regions = sorted(events["region"].unique())

    full_daily_grid = (
        pd.MultiIndex.from_product(
            [analysis_dates, all_regions],
            names=["date", "region"],
        )
        .to_frame(index=False)
    )

    daily_metrics = (
        full_daily_grid
        .merge(
            daily_duration_metrics,
            on=["date", "region"],
            how="left",
        )
        .merge(
            daily_start_metrics,
            on=["date", "region"],
            how="left",
        )
    )

    daily_metrics["total_duration_min"] = (
        daily_metrics["total_duration_min"]
        .fillna(0.0)
        .astype(float)
    )

    daily_metrics["active_alert_count"] = (
        daily_metrics["active_alert_count"]
        .fillna(0)
        .astype(int)
    )

    daily_metrics["alert_start_count"] = (
        daily_metrics["alert_start_count"]
        .fillna(0)
        .astype(int)
    )

    daily_metrics["mean_active_duration_min"] = 0.0

    active_alert_mask = (
        daily_metrics["active_alert_count"] > 0
    )

    daily_metrics.loc[
        active_alert_mask,
        "mean_active_duration_min",
    ] = (
        daily_metrics.loc[
            active_alert_mask,
            "total_duration_min",
        ]
        / daily_metrics.loc[
            active_alert_mask,
            "active_alert_count",
        ]
    )

    daily_metrics = (
        daily_metrics
        .sort_values(["region", "date"])
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------
    # Duration-conservation check after splitting events.
    # -------------------------------------------------------------

    clipped_start = events["started_at_local"].where(
        events["started_at_local"].ge(analysis_start_local),
        analysis_start_local,
    )

    clipped_end = events["finished_at_local"].where(
        events["finished_at_local"].lt(
            analysis_end_exclusive_local
        ),
        analysis_end_exclusive_local,
    )

    expected_duration_min = float(
        (
            (clipped_end - clipped_start)
            .dt.total_seconds()
            .div(60)
            .clip(lower=0)
            .sum()
        )
    )

    allocated_duration_min = float(
        event_day_segments["segment_duration_min"].sum()
    )

    duration_difference_min = (
        allocated_duration_min - expected_duration_min
    )

    expected_daily_rows = len(analysis_dates) * len(all_regions)

    duplicate_daily_rows = int(
        daily_metrics.duplicated(
            subset=["date", "region"],
            keep=False,
        ).sum()
    )

    if len(daily_metrics) != expected_daily_rows:
        raise RuntimeError(
            "The date × region grid is incomplete."
        )

    if duplicate_daily_rows != 0:
        raise RuntimeError(
            "Duplicate date-region rows were created."
        )

    if abs(duration_difference_min) > 1e-8:
        raise RuntimeError(
            "Duration was not conserved while splitting "
            "events across calendar days."
        )

    # -------------------------------------------------------------
    # Aggregation quality report.
    # -------------------------------------------------------------

    quality_report = pd.DataFrame(
        {
            "metric": [
                "input_cleaned_events",
                "cross_midnight_events",
                "event_day_segments_created",
                "daily_metrics_rows",
                "expected_daily_metrics_rows",
                "unique_regions",
                "analysis_days",
                "duplicate_date_region_rows",
                "expected_duration_min",
                "allocated_duration_min",
                "duration_difference_min",
            ],
            "value": [
                len(events),
                cross_midnight_events,
                len(event_day_segments),
                len(daily_metrics),
                expected_daily_rows,
                len(all_regions),
                len(analysis_dates),
                duplicate_daily_rows,
                expected_duration_min,
                allocated_duration_min,
                duration_difference_min,
            ],
        }
    )

    return daily_metrics, quality_report