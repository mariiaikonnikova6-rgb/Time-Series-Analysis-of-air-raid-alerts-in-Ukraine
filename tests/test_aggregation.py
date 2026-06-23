import pandas as pd
import pytest

from src.aggregation import aggregate_alert_events_daily


TIMEZONE = "Europe/Kyiv"


def get_report_value(
    quality_report: pd.DataFrame,
    metric_name: str,
):
    """
    Return one value from the aggregation quality report.
    """
    return quality_report.loc[
        quality_report["metric"].eq(metric_name),
        "value",
    ].iloc[0]


def make_cleaned_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Create a small cleaned event-level DataFrame for tests.
    """
    return pd.DataFrame(rows)


def test_event_crossing_midnight_is_split_correctly():
    cleaned_dataframe = make_cleaned_dataframe(
        [
            {
                "region": "Kyiv City",
                "started_at_local": pd.Timestamp(
                    "2024-01-01 23:30:00",
                    tz=TIMEZONE,
                ),
                "finished_at_local": pd.Timestamp(
                    "2024-01-02 01:20:00",
                    tz=TIMEZONE,
                ),
                "duration_min": 110.0,
            }
        ]
    )

    daily_metrics, quality_report = aggregate_alert_events_daily(
        cleaned_dataframe=cleaned_dataframe,
        analysis_start_date="2024-01-01",
        analysis_end_date="2024-01-02",
        analysis_timezone=TIMEZONE,
    )

    first_day = daily_metrics.loc[
        daily_metrics["date"].eq(pd.Timestamp("2024-01-01"))
    ].iloc[0]

    second_day = daily_metrics.loc[
        daily_metrics["date"].eq(pd.Timestamp("2024-01-02"))
    ].iloc[0]

    assert first_day["total_duration_min"] == pytest.approx(30.0)
    assert second_day["total_duration_min"] == pytest.approx(80.0)

    assert first_day["alert_start_count"] == 1
    assert second_day["alert_start_count"] == 0

    assert first_day["active_alert_count"] == 1
    assert second_day["active_alert_count"] == 1

    assert get_report_value(
        quality_report,
        "allocated_duration_min",
    ) == pytest.approx(110.0)

    assert get_report_value(
        quality_report,
        "duration_difference_min",
    ) == pytest.approx(0.0)


def test_daily_grid_contains_zero_alert_days():
    cleaned_dataframe = make_cleaned_dataframe(
        [
            {
                "region": "Kyiv City",
                "started_at_local": pd.Timestamp(
                    "2024-01-01 10:00:00",
                    tz=TIMEZONE,
                ),
                "finished_at_local": pd.Timestamp(
                    "2024-01-01 11:00:00",
                    tz=TIMEZONE,
                ),
                "duration_min": 60.0,
            },
            {
                "region": "Lvivska oblast",
                "started_at_local": pd.Timestamp(
                    "2024-01-01 12:00:00",
                    tz=TIMEZONE,
                ),
                "finished_at_local": pd.Timestamp(
                    "2024-01-01 12:30:00",
                    tz=TIMEZONE,
                ),
                "duration_min": 30.0,
            },
        ]
    )

    daily_metrics, quality_report = aggregate_alert_events_daily(
        cleaned_dataframe=cleaned_dataframe,
        analysis_start_date="2024-01-01",
        analysis_end_date="2024-01-02",
        analysis_timezone=TIMEZONE,
    )

    # 2 dates × 2 regions = 4 rows.
    assert len(daily_metrics) == 4

    lviv_second_day = daily_metrics.loc[
        daily_metrics["date"].eq(pd.Timestamp("2024-01-02"))
        & daily_metrics["region"].eq("Lvivska oblast")
    ].iloc[0]

    assert lviv_second_day["total_duration_min"] == 0.0
    assert lviv_second_day["alert_start_count"] == 0
    assert lviv_second_day["active_alert_count"] == 0

    assert get_report_value(
        quality_report,
        "expected_daily_metrics_rows",
    ) == 4

    assert get_report_value(
        quality_report,
        "duplicate_date_region_rows",
    ) == 0


def test_event_is_clipped_to_analysis_period():
    cleaned_dataframe = make_cleaned_dataframe(
        [
            {
                "region": "Kyiv City",
                "started_at_local": pd.Timestamp(
                    "2024-01-01 23:30:00",
                    tz=TIMEZONE,
                ),
                "finished_at_local": pd.Timestamp(
                    "2024-01-02 01:30:00",
                    tz=TIMEZONE,
                ),
                "duration_min": 120.0,
            }
        ]
    )

    daily_metrics, quality_report = aggregate_alert_events_daily(
        cleaned_dataframe=cleaned_dataframe,
        analysis_start_date="2024-01-02",
        analysis_end_date="2024-01-02",
        analysis_timezone=TIMEZONE,
    )

    assert len(daily_metrics) == 1

    only_day = daily_metrics.iloc[0]

    # Only the segment from 00:00 to 01:30 belongs to 2024-01-02.
    assert only_day["total_duration_min"] == pytest.approx(90.0)
    assert only_day["alert_start_count"] == 0
    assert only_day["active_alert_count"] == 1

    assert get_report_value(
        quality_report,
        "expected_duration_min",
    ) == pytest.approx(90.0)

    assert get_report_value(
        quality_report,
        "allocated_duration_min",
    ) == pytest.approx(90.0)