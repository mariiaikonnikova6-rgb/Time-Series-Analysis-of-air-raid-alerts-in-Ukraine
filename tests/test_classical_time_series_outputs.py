from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def test_stl_components_are_complete_and_additive():
    """
    Check that saved STL decomposition components are valid,
    continuous and reconstruct the observed series.
    """
    stl_file = config.KYIV_CITY_STL_COMPONENTS_FILE

    assert stl_file.exists(), (
        "STL components CSV was not found. "
        "Run the STL cell in 02_exploratory_analysis.ipynb first."
    )

    stl_data = pd.read_csv(
        stl_file,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "observed",
        "trend",
        "seasonal",
        "residual",
    }

    assert required_columns.issubset(stl_data.columns), (
        "STL CSV is missing required columns."
    )

    assert not stl_data.empty
    assert not stl_data["date"].duplicated().any()
    assert stl_data["date"].is_monotonic_increasing

    expected_dates = pd.date_range(
        start=stl_data["date"].min(),
        end=stl_data["date"].max(),
        freq="D",
    )

    assert pd.DatetimeIndex(stl_data["date"]).equals(
        expected_dates
    ), "STL dates are not a continuous daily sequence."

    component_columns = [
        "observed",
        "trend",
        "seasonal",
        "residual",
    ]

    assert not stl_data[component_columns].isna().any().any()

    assert np.isfinite(
        stl_data[component_columns].to_numpy()
    ).all()

    assert (stl_data["observed"] >= 0).all()

    reconstructed_observed = (
        stl_data["trend"]
        + stl_data["seasonal"]
        + stl_data["residual"]
    )

    assert np.allclose(
        stl_data["observed"].to_numpy(),
        reconstructed_observed.to_numpy(),
        rtol=1e-7,
        atol=1e-7,
    ), (
        "Observed STL values do not equal "
        "trend + seasonal + residual."
    )


def test_classical_predictions_match_final_test_period():
    """
    Check that ARIMA/SARIMA predictions use exactly the same
    chronological final test dates as the existing forecast dataset.
    """
    prediction_file = config.ARIMA_SARIMA_TEST_PREDICTIONS_FILE
    forecast_test_file = config.FORECAST_TEST_FILE

    assert prediction_file.exists(), (
        "ARIMA/SARIMA prediction CSV was not found. "
        "Run the final ARIMA/SARIMA forecast cell first."
    )

    assert forecast_test_file.exists(), (
        "Forecast test CSV was not found."
    )

    predictions = pd.read_csv(
        prediction_file,
        parse_dates=["date"],
    )

    forecast_test = pd.read_csv(
        forecast_test_file,
        parse_dates=["date"],
    )

    required_columns = {
        "date",
        "actual_duration_min",
        "seasonal_naive_duration_min",
        "arima_duration_min",
        "sarima_duration_min",
        "sarima_lower_95_min",
        "sarima_upper_95_min",
    }

    assert required_columns.issubset(predictions.columns), (
        "Prediction CSV is missing required columns."
    )

    assert len(predictions) == config.FORECAST_TEST_SIZE_DAYS
    assert len(predictions) == len(forecast_test)

    assert not predictions["date"].duplicated().any()
    assert predictions["date"].is_monotonic_increasing

    expected_dates = pd.date_range(
        start=predictions["date"].min(),
        end=predictions["date"].max(),
        freq="D",
    )

    assert pd.DatetimeIndex(predictions["date"]).equals(
        expected_dates
    ), "Prediction dates are not continuous daily observations."

    assert predictions["date"].equals(
        forecast_test["date"]
    ), (
        "ARIMA/SARIMA prediction dates do not match "
        "the established final test dates."
    )

    assert np.allclose(
        predictions["actual_duration_min"].to_numpy(),
        forecast_test["target"].to_numpy(),
    ), (
        "Actual values in the ARIMA/SARIMA prediction file "
        "do not match the final test target."
    )

    numeric_columns = [
        "actual_duration_min",
        "seasonal_naive_duration_min",
        "arima_duration_min",
        "sarima_duration_min",
        "sarima_lower_95_min",
        "sarima_upper_95_min",
    ]

    assert np.isfinite(
        predictions[numeric_columns].to_numpy()
    ).all()

    assert (
        predictions[numeric_columns] >= 0
    ).all().all()

    assert (
        predictions["sarima_lower_95_min"]
        <= predictions["sarima_upper_95_min"]
    ).all(), (
        "Some SARIMA lower prediction bounds exceed upper bounds."
    )


def test_forecast_metrics_include_all_model_families():
    """
    Check that the final metric table contains baseline,
    Ridge, ARIMA and SARIMA results.
    """
    metrics_file = config.FORECAST_METRICS_FILE

    assert metrics_file.exists(), (
        "Forecast metrics CSV was not found."
    )

    metrics = pd.read_csv(metrics_file)

    required_columns = {
        "model",
        "mae_min",
        "rmse_min",
        "smape_percent",
        "zero_duration_days_in_test",
        "negative_raw_predictions_clipped_to_zero",
    }

    assert required_columns.issubset(metrics.columns), (
        "Forecast metrics CSV is missing required columns."
    )

    assert not metrics.empty
    assert metrics["model"].notna().all()

    model_names = metrics["model"].astype(str)

    assert (
        model_names == "Seasonal Naive (lag 7)"
    ).any(), "Seasonal Naive result is missing."

    assert model_names.str.startswith(
        "Ridge Regression"
    ).any(), "Ridge Regression result is missing."

    assert model_names.str.startswith(
        "ARIMA("
    ).any(), "ARIMA result is missing."

    assert model_names.str.startswith(
        "SARIMA("
    ).any(), "SARIMA result is missing."

    metric_columns = [
        "mae_min",
        "rmse_min",
        "smape_percent",
    ]

    assert np.isfinite(
        metrics[metric_columns].to_numpy()
    ).all()

    assert (
        metrics[metric_columns] >= 0
    ).all().all()

    assert (
        metrics["zero_duration_days_in_test"]
        == 28
    ).all(), (
        "All models should use the same final test period "
        "with 28 zero-duration days."
    )


def test_final_stl_and_forecast_figures_exist():
    """
    Check that the two new final PNG figures were created.
    """
    required_figures = [
        config.KYIV_CITY_STL_DECOMPOSITION_FIGURE,
        config.KYIV_CITY_FORECAST_COMPARISON_FIGURE,
    ]

    for figure_file in required_figures:
        assert figure_file.exists(), (
            f"Required figure was not found: {figure_file}"
        )

        assert figure_file.stat().st_size > 0, (
            f"Figure file is empty: {figure_file}"
        )