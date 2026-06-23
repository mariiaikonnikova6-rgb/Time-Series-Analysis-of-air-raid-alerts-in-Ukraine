import pandas as pd

from src.preprocessing import preprocess_oblast_alert_events


def make_raw_dataframe(rows: list[dict]) -> pd.DataFrame:
    """
    Create a minimal DataFrame compatible with the raw dataset structure.
    """
    return pd.DataFrame(rows)


def get_report_value(
    quality_report: pd.DataFrame,
    metric_name: str,
):
    """
    Get one metric value from the generated quality report.
    """
    return quality_report.loc[
        quality_report["metric"].eq(metric_name),
        "value",
    ].iloc[0]


def test_duration_is_calculated_correctly():
    raw_dataframe = make_raw_dataframe(
        [
            {
                "oblast": "Kyiv City",
                "level": "oblast",
                "started_at": "2024-01-01 10:00:00+00:00",
                "finished_at": "2024-01-01 11:30:00+00:00",
                "source": "official",
            }
        ]
    )

    cleaned_dataframe, _ = preprocess_oblast_alert_events(
        raw_dataframe=raw_dataframe,
        target_geographic_level="oblast",
        analysis_start_date="2024-01-01",
        analysis_end_date="2024-01-01",
        analysis_timezone="Europe/Kyiv",
    )

    assert len(cleaned_dataframe) == 1
    assert cleaned_dataframe.loc[0, "duration_min"] == 90.0

    expected_local_start = pd.Timestamp(
        "2024-01-01 12:00:00",
        tz="Europe/Kyiv",
    )

    assert (
        cleaned_dataframe.loc[0, "started_at_local"]
        == expected_local_start
    )


def test_exact_duplicate_events_are_removed():
    duplicated_event = {
        "oblast": "Kyiv City",
        "level": "oblast",
        "started_at": "2024-02-01 10:00:00+00:00",
        "finished_at": "2024-02-01 11:00:00+00:00",
        "source": "official",
    }

    raw_dataframe = make_raw_dataframe(
        [
            duplicated_event,
            duplicated_event.copy(),
        ]
    )

    cleaned_dataframe, quality_report = preprocess_oblast_alert_events(
        raw_dataframe=raw_dataframe,
        target_geographic_level="oblast",
        analysis_start_date="2024-02-01",
        analysis_end_date="2024-02-01",
        analysis_timezone="Europe/Kyiv",
    )

    assert len(cleaned_dataframe) == 1

    assert get_report_value(
        quality_report,
        "rows_removed_as_duplicates",
    ) == 1

    assert get_report_value(
        quality_report,
        "remaining_duplicate_rows",
    ) == 0


def test_nonpositive_duration_events_are_removed():
    raw_dataframe = make_raw_dataframe(
        [
            {
                "oblast": "Kyiv City",
                "level": "oblast",
                "started_at": "2024-03-01 10:00:00+00:00",
                "finished_at": "2024-03-01 10:00:00+00:00",
                "source": "official",
            },
            {
                "oblast": "Kyiv City",
                "level": "oblast",
                "started_at": "2024-03-01 12:00:00+00:00",
                "finished_at": "2024-03-01 13:00:00+00:00",
                "source": "official",
            },
        ]
    )

    cleaned_dataframe, quality_report = preprocess_oblast_alert_events(
        raw_dataframe=raw_dataframe,
        target_geographic_level="oblast",
        analysis_start_date="2024-03-01",
        analysis_end_date="2024-03-01",
        analysis_timezone="Europe/Kyiv",
    )

    assert len(cleaned_dataframe) == 1
    assert cleaned_dataframe.loc[0, "duration_min"] == 60.0

    assert get_report_value(
        quality_report,
        "nonpositive_duration_rows_removed",
    ) == 1