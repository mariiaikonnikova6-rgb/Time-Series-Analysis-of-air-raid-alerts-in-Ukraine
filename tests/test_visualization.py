import pandas as pd
import pytest

from src.visualization import prepare_monthly_duration


def test_monthly_duration_is_aggregated_correctly():
    daily_metrics = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Kyiv City",
                "total_duration_min": 60.0,
                "alert_start_count": 1,
            },
            {
                "date": pd.Timestamp("2024-01-15"),
                "region": "Kyiv City",
                "total_duration_min": 120.0,
                "alert_start_count": 2,
            },
            {
                "date": pd.Timestamp("2024-02-01"),
                "region": "Kyiv City",
                "total_duration_min": 30.0,
                "alert_start_count": 1,
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Lvivska oblast",
                "total_duration_min": 90.0,
                "alert_start_count": 1,
            },
        ]
    )

    monthly_duration = prepare_monthly_duration(
        daily_metrics=daily_metrics,
        selected_regions=[
            "Kyiv City",
            "Lvivska oblast",
        ],
    )

    kyiv_january = monthly_duration.loc[
        monthly_duration["month"].eq(
            pd.Timestamp("2024-01-01")
        )
        & monthly_duration["region"].eq("Kyiv City")
    ].iloc[0]

    kyiv_february = monthly_duration.loc[
        monthly_duration["month"].eq(
            pd.Timestamp("2024-02-01")
        )
        & monthly_duration["region"].eq("Kyiv City")
    ].iloc[0]

    assert kyiv_january["total_duration_min"] == 180.0
    assert kyiv_january["total_duration_hours"] == 3.0
    assert kyiv_january["alert_start_count"] == 3

    assert kyiv_february["total_duration_min"] == 30.0
    assert kyiv_february["total_duration_hours"] == 0.5


def test_missing_selected_region_raises_error():
    daily_metrics = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Kyiv City",
                "total_duration_min": 60.0,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Selected regions were not found",
    ):
        prepare_monthly_duration(
            daily_metrics=daily_metrics,
            selected_regions=[
                "Kyiv City",
                "Imaginary Oblast",
            ],
        )


import matplotlib.pyplot as plt

from src.visualization import (
    plot_top_regions_by_total_duration,
    prepare_top_regions_by_total_duration,
)


def test_top_regions_are_ranked_by_total_duration():
    daily_metrics = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Kyiv City",
                "total_duration_min": 120.0,
                "alert_start_count": 2,
                "active_alert_count": 2,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "region": "Kyiv City",
                "total_duration_min": 180.0,
                "alert_start_count": 1,
                "active_alert_count": 1,
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Lvivska oblast",
                "total_duration_min": 60.0,
                "alert_start_count": 1,
                "active_alert_count": 1,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "region": "Kharkivska oblast",
                "total_duration_min": 480.0,
                "alert_start_count": 3,
                "active_alert_count": 3,
            },
        ]
    )

    top_regions = prepare_top_regions_by_total_duration(
        daily_metrics=daily_metrics,
        top_n=2,
    )

    assert len(top_regions) == 2
    assert top_regions["region"].tolist() == [
        "Kyiv City",
        "Kharkivska oblast",
    ]

    kyiv_row = top_regions.loc[
        top_regions["region"].eq("Kyiv City")
    ].iloc[0]

    kharkiv_row = top_regions.loc[
        top_regions["region"].eq("Kharkivska oblast")
    ].iloc[0]

    assert kyiv_row["total_duration_min"] == 300.0
    assert kyiv_row["total_duration_hours"] == 5.0
    assert kyiv_row["total_active_alert_days"] == 2

    assert kharkiv_row["total_duration_min"] == 480.0
    assert kharkiv_row["total_duration_hours"] == 8.0


def test_top_regions_plot_is_saved(tmp_path):
    top_regions = pd.DataFrame(
        {
            "region": [
                "Kyiv City",
                "Kharkivska oblast",
            ],
            "total_duration_hours": [
                100.0,
                200.0,
            ],
        }
    )

    output_path = tmp_path / "top_regions.png"

    figure, axes = plot_top_regions_by_total_duration(
        top_regions=top_regions,
        output_path=output_path,
    )

    assert output_path.exists()
    assert axes.get_xlabel() == "Total alert duration, hours"
    assert axes.get_ylabel() == "Region"

    plt.close(figure)





from src.visualization import (
    plot_alert_duration_distribution,
    prepare_selected_event_durations,
)


def test_event_duration_summary_is_calculated_correctly():
    cleaned_events = pd.DataFrame(
        [
            {
                "region": "Kyiv City",
                "duration_min": 30.0,
            },
            {
                "region": "Kyiv City",
                "duration_min": 90.0,
            },
            {
                "region": "Kyiv City",
                "duration_min": 150.0,
            },
            {
                "region": "Lvivska oblast",
                "duration_min": 60.0,
            },
            {
                "region": "Lvivska oblast",
                "duration_min": 120.0,
            },
        ]
    )

    selected_events, duration_summary = (
        prepare_selected_event_durations(
            cleaned_events=cleaned_events,
            selected_regions=[
                "Kyiv City",
                "Lvivska oblast",
            ],
        )
    )

    assert len(selected_events) == 5
    assert duration_summary["region"].tolist() == [
        "Kyiv City",
        "Lvivska oblast",
    ]

    kyiv_row = duration_summary.loc[
        duration_summary["region"].eq("Kyiv City")
    ].iloc[0]

    lviv_row = duration_summary.loc[
        duration_summary["region"].eq("Lvivska oblast")
    ].iloc[0]

    assert kyiv_row["event_count"] == 3
    assert kyiv_row["mean_duration_min"] == pytest.approx(90.0)
    assert kyiv_row["median_duration_min"] == pytest.approx(90.0)
    assert kyiv_row["p25_duration_min"] == pytest.approx(60.0)
    assert kyiv_row["p75_duration_min"] == pytest.approx(120.0)
    assert kyiv_row["max_duration_min"] == pytest.approx(150.0)

    assert lviv_row["event_count"] == 2
    assert lviv_row["mean_duration_min"] == pytest.approx(90.0)
    assert lviv_row["median_duration_min"] == pytest.approx(90.0)


def test_alert_duration_boxplot_is_saved(tmp_path):
    selected_events = pd.DataFrame(
        [
            {
                "region": "Kyiv City",
                "duration_min": 30.0,
            },
            {
                "region": "Kyiv City",
                "duration_min": 90.0,
            },
            {
                "region": "Lvivska oblast",
                "duration_min": 60.0,
            },
            {
                "region": "Lvivska oblast",
                "duration_min": 120.0,
            },
        ]
    )

    output_path = tmp_path / "alert_duration_distribution.png"

    figure, axes = plot_alert_duration_distribution(
        selected_events=selected_events,
        selected_regions=[
            "Kyiv City",
            "Lvivska oblast",
        ],
        output_path=output_path,
    )

    assert output_path.exists()
    assert axes.get_xlabel() == "Region"
    assert axes.get_ylabel() == "Alert duration, minutes"

    plt.close(figure)


from src.visualization import (
    plot_alert_start_month_hour_heatmap,
    prepare_month_hour_alert_start_counts,
)

def test_month_hour_alert_start_counts_are_correct():
    cleaned_events = pd.DataFrame(
        [
            {
                "region": "Kyiv City",
                "started_at_local": "2024-01-15 00:10:00+02:00",
            },
            {
                "region": "Kyiv City",
                "started_at_local": "2024-01-20 00:40:00+02:00",
            },
            {
                "region": "Kyiv City",
                "started_at_local": "2024-02-05 23:15:00+02:00",
            },
            {
                "region": "Lvivska oblast",
                "started_at_local": "2024-01-10 12:00:00+02:00",
            },
        ]
    )

    month_hour_counts, event_count = (
        prepare_month_hour_alert_start_counts(
            cleaned_events=cleaned_events,
            region="Kyiv City",
            analysis_timezone="Europe/Kyiv",
        )
    )

    assert month_hour_counts.shape == (12, 24)
    assert event_count == 3

    assert month_hour_counts.loc["January", 0] == 2
    assert month_hour_counts.loc["February", 23] == 1

    assert int(month_hour_counts.to_numpy().sum()) == 3


def test_month_hour_heatmap_is_saved(tmp_path):
    month_labels = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    month_hour_counts = pd.DataFrame(
        0,
        index=month_labels,
        columns=range(24),
    )

    month_hour_counts.loc["January", 0] = 5
    month_hour_counts.loc["February", 23] = 3

    output_path = tmp_path / "kyiv_heatmap.png"

    figure, axes = plot_alert_start_month_hour_heatmap(
        month_hour_counts=month_hour_counts,
        region="Kyiv City",
        output_path=output_path,
    )

    assert output_path.exists()
    assert axes.get_xlabel() == "Local hour of day"
    assert axes.get_ylabel() == "Month"

    plt.close(figure)



from src.visualization import (
    plot_daily_duration_correlation_heatmap,
    prepare_daily_duration_correlation,
)


def test_daily_duration_correlation_is_calculated_correctly():
    daily_metrics = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Kyiv City",
                "total_duration_min": 60.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "region": "Kyiv City",
                "total_duration_min": 120.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "region": "Kyiv City",
                "total_duration_min": 180.0,
            },
            {
                "date": pd.Timestamp("2024-01-01"),
                "region": "Lvivska oblast",
                "total_duration_min": 30.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "region": "Lvivska oblast",
                "total_duration_min": 60.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "region": "Lvivska oblast",
                "total_duration_min": 90.0,
            },
        ]
    )

    daily_duration_matrix, correlation_matrix = (
        prepare_daily_duration_correlation(
            daily_metrics=daily_metrics,
            selected_regions=[
                "Kyiv City",
                "Lvivska oblast",
            ],
        )
    )

    assert daily_duration_matrix.shape == (3, 2)
    assert correlation_matrix.shape == (2, 2)

    assert correlation_matrix.loc[
        "Kyiv City",
        "Kyiv City",
    ] == pytest.approx(1.0)

    assert correlation_matrix.loc[
        "Lvivska oblast",
        "Lvivska oblast",
    ] == pytest.approx(1.0)

    assert correlation_matrix.loc[
        "Kyiv City",
        "Lvivska oblast",
    ] == pytest.approx(1.0)


def test_daily_duration_correlation_heatmap_is_saved(tmp_path):
    correlation_matrix = pd.DataFrame(
        {
            "Kyiv City": [1.0, 0.65],
            "Lvivska oblast": [0.65, 1.0],
        },
        index=[
            "Kyiv City",
            "Lvivska oblast",
        ],
    )

    output_path = tmp_path / "correlation_heatmap.png"

    figure, axes = plot_daily_duration_correlation_heatmap(
        correlation_matrix=correlation_matrix,
        output_path=output_path,
    )

    assert output_path.exists()
    assert axes.get_xlabel() == "Region"
    assert axes.get_ylabel() == "Region"

    plt.close(figure)



