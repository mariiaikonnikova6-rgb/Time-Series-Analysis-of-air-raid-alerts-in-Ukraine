import matplotlib.pyplot as plt
import pandas as pd
import pytest
import numpy as np
from src.forecasting import (
    evaluate_seasonal_naive_baseline,
    plot_seasonal_naive_baseline,
)
from src.forecasting import (
    plot_ridge_vs_baseline,
    train_and_evaluate_ridge_regression,
)

def test_seasonal_naive_metrics_are_calculated_correctly():
    test_data = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "target": 0.0,
                "seasonal_naive_prediction": 0.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "target": 60.0,
                "seasonal_naive_prediction": 30.0,
            },
            {
                "date": pd.Timestamp("2024-01-03"),
                "target": 120.0,
                "seasonal_naive_prediction": 60.0,
            },
        ]
    )

    baseline_evaluation, metrics_table = (
        evaluate_seasonal_naive_baseline(
            test_data=test_data,
            forecast_region="Kyiv City",
        )
    )

    assert len(baseline_evaluation) == 3

    assert baseline_evaluation[
        "absolute_error_min"
    ].tolist() == [0.0, 30.0, 60.0]

    metrics_row = metrics_table.iloc[0]

    assert metrics_row["mae_min"] == pytest.approx(30.0)

    assert metrics_row["rmse_min"] == pytest.approx(
        (1500.0 / 1) ** 0.5
    )

    assert metrics_row["smape_percent"] == pytest.approx(
        44.444444,
        rel=1e-5,
    )

    assert metrics_row[
        "zero_duration_days_in_test"
    ] == 1


def test_seasonal_naive_plot_is_saved(tmp_path):
    baseline_evaluation = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-01"),
                "actual_duration_min": 60.0,
                "seasonal_naive_duration_min": 30.0,
            },
            {
                "date": pd.Timestamp("2024-01-02"),
                "actual_duration_min": 90.0,
                "seasonal_naive_duration_min": 120.0,
            },
        ]
    )

    output_path = tmp_path / "seasonal_naive.png"

    figure, axes = plot_seasonal_naive_baseline(
        baseline_evaluation=baseline_evaluation,
        output_path=output_path,
        plot_days=2,
    )

    assert output_path.exists()
    assert axes.get_xlabel() == "Date"
    assert axes.get_ylabel() == "Total alert duration, minutes"

    plt.close(figure)




def test_ridge_regression_trains_and_evaluates():
    train_dates = pd.date_range(
        start="2024-01-01",
        periods=20,
        freq="D",
    )

    test_dates = pd.date_range(
        start="2024-01-21",
        periods=5,
        freq="D",
    )

    train_data = pd.DataFrame(
        {
            "date": train_dates,
            "target": [
                20.0 + 4.0 * value
                for value in range(20)
            ],
            "lag_1": [
                10.0 + 3.0 * value
                for value in range(20)
            ],
            "is_weekend": [
                int(date.dayofweek >= 5)
                for date in train_dates
            ],
        }
    )

    test_data = pd.DataFrame(
        {
            "date": test_dates,
            "target": [
                20.0 + 4.0 * value
                for value in range(20, 25)
            ],
            "lag_1": [
                10.0 + 3.0 * value
                for value in range(20, 25)
            ],
            "is_weekend": [
                int(date.dayofweek >= 5)
                for date in test_dates
            ],
        }
    )

    (
        ridge_evaluation,
        ridge_metrics_table,
        ridge_coefficients,
        cv_results,
        best_ridge_model,
    ) = train_and_evaluate_ridge_regression(
        train_data=train_data,
        test_data=test_data,
        feature_columns=[
            "lag_1",
            "is_weekend",
        ],
        forecast_region="Kyiv City",
        alpha_candidates=[
            0.1,
            1.0,
            10.0,
        ],
        time_series_splits=2,
    )

    assert len(ridge_evaluation) == 5
    assert (
        ridge_evaluation["ridge_duration_min"] >= 0
    ).all()

    assert len(ridge_metrics_table) == 1
    assert ridge_metrics_table.iloc[0][
        "model"
    ].startswith("Ridge Regression")

    assert len(ridge_coefficients) == 2
    assert set(ridge_coefficients["feature"]) == {
        "lag_1",
        "is_weekend",
    }

    assert len(cv_results) == 3
    assert (
        cv_results["mean_validation_mae_min"]
        .is_monotonic_increasing
    )

    assert best_ridge_model.named_steps["ridge"] is not None


def test_ridge_vs_baseline_plot_is_saved(tmp_path):
    ridge_evaluation = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "actual_duration_min": [
                60.0,
                90.0,
                120.0,
            ],
            "ridge_duration_min": [
                55.0,
                85.0,
                125.0,
            ],
        }
    )

    baseline_evaluation = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "seasonal_naive_duration_min": [
                30.0,
                120.0,
                100.0,
            ],
        }
    )

    output_path = tmp_path / "ridge_vs_baseline.png"

    figure, axes = plot_ridge_vs_baseline(
        ridge_evaluation=ridge_evaluation,
        baseline_evaluation=baseline_evaluation,
        best_alpha=10.0,
        output_path=output_path,
        plot_days=3,
    )

    assert output_path.exists()
    assert axes.get_xlabel() == "Date"
    assert axes.get_ylabel() == "Total alert duration, minutes"

    plt.close(figure)