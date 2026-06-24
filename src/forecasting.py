from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_BASELINE_COLUMNS = [
    "date",
    "target",
    "seasonal_naive_prediction",
]

REQUIRED_BASELINE_EVALUATION_COLUMNS = [
    "date",
    "actual_duration_min",
    "seasonal_naive_duration_min",
]


# ---------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------

def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Check that a DataFrame contains all required columns.
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
# Forecast metrics
# ---------------------------------------------------------------------

def calculate_forecast_metrics(
    actual_values: pd.Series,
    predicted_values: pd.Series,
) -> dict[str, float]:
    """
    Calculate MAE, RMSE and sMAPE for forecast values.

    Parameters
    ----------
    actual_values:
        Actual target values.
    predicted_values:
        Forecasted target values.

    Returns
    -------
    dict[str, float]
        Dictionary with mae_min, rmse_min and smape_percent.
    """
    actual = pd.Series(
        actual_values,
        dtype="float64",
    )

    predicted = pd.Series(
        predicted_values,
        dtype="float64",
    )

    if len(actual) == 0:
        raise ValueError(
            "Forecast metric calculation requires at least one row."
        )

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted values must have equal length. "
            f"Actual: {len(actual)}, predicted: {len(predicted)}."
        )

    if actual.isna().any() or predicted.isna().any():
        raise ValueError(
            "Actual or predicted values contain missing values."
        )

    if not np.isfinite(actual.to_numpy()).all():
        raise ValueError(
            "Actual values contain non-finite values."
        )

    if not np.isfinite(predicted.to_numpy()).all():
        raise ValueError(
            "Predicted values contain non-finite values."
        )

    if (actual < 0).any():
        raise ValueError(
            "Actual duration values cannot be negative."
        )

    absolute_error = (
        actual
        - predicted
    ).abs()

    squared_error = (
        actual
        - predicted
    ) ** 2

    smape_denominator = (
        actual.abs()
        + predicted.abs()
    )

    smape_components = np.where(
        smape_denominator.eq(0),
        0.0,
        2 * absolute_error / smape_denominator,
    )

    return {
        "mae_min": float(absolute_error.mean()),
        "rmse_min": float(
            np.sqrt(squared_error.mean())
        ),
        "smape_percent": float(
            np.mean(smape_components) * 100
        ),
    }


# ---------------------------------------------------------------------
# Seasonal Naive baseline
# ---------------------------------------------------------------------

def evaluate_seasonal_naive_baseline(
    test_data: pd.DataFrame,
    forecast_region: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a Seasonal Naive baseline forecast.

    The forecast for day t equals the actual target value from day t - 7.
    The required prediction must already be stored in
    seasonal_naive_prediction.

    Parameters
    ----------
    test_data:
        Chronological test dataset created during feature preparation.
    forecast_region:
        Region name used in reports.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        baseline_evaluation:
            Day-level actual values, predictions and errors.
        metrics_table:
            One-row table with forecast metrics.
    """
    validate_required_columns(
        dataframe=test_data,
        required_columns=REQUIRED_BASELINE_COLUMNS,
    )

    baseline_evaluation = (
        test_data[
            REQUIRED_BASELINE_COLUMNS
        ]
        .copy()
    )

    baseline_evaluation["date"] = pd.to_datetime(
        baseline_evaluation["date"],
        errors="coerce",
    )

    if baseline_evaluation["date"].isna().any():
        invalid_date_count = int(
            baseline_evaluation["date"].isna().sum()
        )

        raise ValueError(
            "Some test dates could not be parsed correctly. "
            f"Invalid date count: {invalid_date_count}"
        )

    if baseline_evaluation["date"].duplicated().any():
        duplicate_date_count = int(
            baseline_evaluation["date"].duplicated().sum()
        )

        raise ValueError(
            "Duplicate test dates were found. "
            f"Duplicate count: {duplicate_date_count}"
        )

    baseline_evaluation = (
        baseline_evaluation
        .sort_values("date")
        .reset_index(drop=True)
    )

    baseline_evaluation["target"] = pd.to_numeric(
        baseline_evaluation["target"],
        errors="coerce",
    )

    baseline_evaluation[
        "seasonal_naive_prediction"
    ] = pd.to_numeric(
        baseline_evaluation["seasonal_naive_prediction"],
        errors="coerce",
    )

    numeric_columns = [
        "target",
        "seasonal_naive_prediction",
    ]

    if baseline_evaluation[numeric_columns].isna().any().any():
        raise ValueError(
            "The test dataset contains invalid target values "
            "or baseline predictions."
        )

    if (
        baseline_evaluation[numeric_columns] < 0
    ).any().any():
        raise ValueError(
            "Target values or Seasonal Naive predictions "
            "cannot be negative."
        )

    baseline_evaluation = (
        baseline_evaluation
        .rename(
            columns={
                "target": "actual_duration_min",
                "seasonal_naive_prediction": (
                    "seasonal_naive_duration_min"
                ),
            }
        )
    )

    baseline_evaluation["absolute_error_min"] = (
        baseline_evaluation["actual_duration_min"]
        - baseline_evaluation[
            "seasonal_naive_duration_min"
        ]
    ).abs()

    baseline_evaluation["squared_error_min2"] = (
        baseline_evaluation["actual_duration_min"]
        - baseline_evaluation[
            "seasonal_naive_duration_min"
        ]
    ) ** 2

    smape_denominator = (
        baseline_evaluation["actual_duration_min"].abs()
        + baseline_evaluation[
            "seasonal_naive_duration_min"
        ].abs()
    )

    baseline_evaluation["smape_component"] = np.where(
        smape_denominator.eq(0),
        0.0,
        2
        * baseline_evaluation["absolute_error_min"]
        / smape_denominator,
    )

    metrics = calculate_forecast_metrics(
        actual_values=baseline_evaluation[
            "actual_duration_min"
        ],
        predicted_values=baseline_evaluation[
            "seasonal_naive_duration_min"
        ],
    )

    metrics_table = pd.DataFrame(
        [
            {
                "model": "Seasonal Naive (lag 7)",
                "forecast_region": forecast_region,
                "test_start_date": (
                    baseline_evaluation["date"]
                    .min()
                    .date()
                ),
                "test_end_date": (
                    baseline_evaluation["date"]
                    .max()
                    .date()
                ),
                "test_rows": len(baseline_evaluation),
                "mae_min": metrics["mae_min"],
                "rmse_min": metrics["rmse_min"],
                "smape_percent": metrics[
                    "smape_percent"
                ],
                "zero_duration_days_in_test": int(
                    baseline_evaluation[
                        "actual_duration_min"
                    ]
                    .eq(0)
                    .sum()
                ),
            }
        ]
    )

    return baseline_evaluation, metrics_table


# ---------------------------------------------------------------------
# Baseline visualization
# ---------------------------------------------------------------------

def plot_seasonal_naive_baseline(
    baseline_evaluation: pd.DataFrame,
    output_path: str | Path | None = None,
    plot_days: int = 60,
):
    """
    Plot actual values and Seasonal Naive predictions.

    Parameters
    ----------
    baseline_evaluation:
        Output of evaluate_seasonal_naive_baseline().
    output_path:
        Optional path for saving the PNG figure.
    plot_days:
        Number of final test days shown in the chart.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    validate_required_columns(
        dataframe=baseline_evaluation,
        required_columns=REQUIRED_BASELINE_EVALUATION_COLUMNS,
    )

    if plot_days <= 0:
        raise ValueError(
            "plot_days must be a positive integer."
        )

    plot_data = (
        baseline_evaluation
        .sort_values("date")
        .tail(plot_days)
        .copy()
    )

    if plot_data.empty:
        raise ValueError(
            "The baseline evaluation dataset is empty."
        )

    figure, axes = plt.subplots(
        figsize=(14, 7),
    )

    axes.plot(
        plot_data["date"],
        plot_data["actual_duration_min"],
        label="Actual daily duration",
        linewidth=2,
    )

    axes.plot(
        plot_data["date"],
        plot_data["seasonal_naive_duration_min"],
        label="Seasonal Naive prediction (t-7)",
        linewidth=1.8,
    )

    axes.set_title(
        "Kyiv City: Seasonal Naive Baseline Forecast "
        f"(Last {len(plot_data)} Test Days)",
        fontsize=14,
    )

    axes.set_xlabel("Date")
    axes.set_ylabel("Total alert duration, minutes")

    axes.grid(
        True,
        alpha=0.3,
    )

    axes.legend()

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    return figure, axes



# ---------------------------------------------------------------------
# Ridge Regression forecast
# ---------------------------------------------------------------------

REQUIRED_RIDGE_COLUMNS = [
    "date",
    "target",
]


def train_and_evaluate_ridge_regression(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    forecast_region: str,
    alpha_candidates: list[float],
    time_series_splits: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Pipeline,
]:
    """
    Train Ridge Regression using chronological cross-validation and
    evaluate it on an untouched chronological test period.

    Parameters
    ----------
    train_data:
        Forecast training dataset with date, target and feature columns.
    test_data:
        Forecast test dataset with date, target and feature columns.
    feature_columns:
        Ordered names of numerical features used by the Ridge model.
    forecast_region:
        Region name used in output reports.
    alpha_candidates:
        Candidate Ridge regularization strengths.
    time_series_splits:
        Number of chronological validation splits inside train_data.

    Returns
    -------
    tuple
        ridge_evaluation:
            Test-day actual values, raw predictions, clipped predictions
            and error columns.
        ridge_metrics_table:
            One-row metrics table for Ridge Regression.
        ridge_coefficients:
            Standardized Ridge coefficients sorted by absolute magnitude.
        cv_results:
            Validation results for all alpha candidates.
        best_ridge_model:
            Fitted StandardScaler + Ridge pipeline.
    """
    validate_required_columns(
        dataframe=train_data,
        required_columns=[
            *REQUIRED_RIDGE_COLUMNS,
            *feature_columns,
        ],
    )

    validate_required_columns(
        dataframe=test_data,
        required_columns=[
            *REQUIRED_RIDGE_COLUMNS,
            *feature_columns,
        ],
    )

    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty."
        )

    if time_series_splits < 2:
        raise ValueError(
            "time_series_splits must be at least 2."
        )

    if not alpha_candidates:
        raise ValueError(
            "alpha_candidates cannot be empty."
        )

    if any(alpha <= 0 for alpha in alpha_candidates):
        raise ValueError(
            "All Ridge alpha candidates must be positive."
        )

    train_copy = train_data.copy()
    test_copy = test_data.copy()

    for dataframe_name, dataframe in [
        ("train_data", train_copy),
        ("test_data", test_copy),
    ]:
        dataframe["date"] = pd.to_datetime(
            dataframe["date"],
            errors="coerce",
        )

        if dataframe["date"].isna().any():
            invalid_date_count = int(
                dataframe["date"].isna().sum()
            )

            raise ValueError(
                f"{dataframe_name} contains invalid dates. "
                f"Invalid count: {invalid_date_count}"
            )

        numeric_columns = [
            "target",
            *feature_columns,
        ]

        dataframe[numeric_columns] = (
            dataframe[numeric_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        if dataframe[numeric_columns].isna().any().any():
            raise ValueError(
                f"{dataframe_name} contains missing or invalid "
                "target/feature values."
            )

        if not np.isfinite(
            dataframe[numeric_columns].to_numpy()
        ).all():
            raise ValueError(
                f"{dataframe_name} contains non-finite values."
            )

        if (dataframe["target"] < 0).any():
            raise ValueError(
                f"{dataframe_name} contains negative target values."
            )

    if train_copy["date"].duplicated().any():
        raise ValueError(
            "train_data contains duplicate dates."
        )

    if test_copy["date"].duplicated().any():
        raise ValueError(
            "test_data contains duplicate dates."
        )

    train_copy = (
        train_copy
        .sort_values("date")
        .reset_index(drop=True)
    )

    test_copy = (
        test_copy
        .sort_values("date")
        .reset_index(drop=True)
    )

    if train_copy["date"].max() >= test_copy["date"].min():
        raise ValueError(
            "Training and test periods overlap."
        )

    if len(train_copy) <= time_series_splits:
        raise ValueError(
            "Training data is too short for the requested "
            "number of TimeSeriesSplit folds."
        )

    X_train = train_copy[feature_columns].copy()
    y_train = train_copy["target"].copy()

    X_test = test_copy[feature_columns].copy()
    y_test = test_copy["target"].copy()

    time_series_cv = TimeSeriesSplit(
        n_splits=time_series_splits,
    )

    ridge_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "ridge",
                Ridge(),
            ),
        ]
    )

    ridge_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid={
            "ridge__alpha": alpha_candidates,
        },
        scoring="neg_mean_absolute_error",
        cv=time_series_cv,
        refit=True,
        return_train_score=True,
    )

    ridge_search.fit(
        X_train,
        y_train,
    )

    best_ridge_model = ridge_search.best_estimator_
    best_alpha = float(
        ridge_search.best_params_["ridge__alpha"]
    )

    ridge_raw_predictions = best_ridge_model.predict(
        X_test
    )

    # Daily duration cannot be negative.
    ridge_predictions = np.clip(
        ridge_raw_predictions,
        a_min=0,
        a_max=None,
    )

    ridge_evaluation = pd.DataFrame(
        {
            "date": test_copy["date"].to_numpy(),
            "actual_duration_min": y_test.to_numpy(),
            "ridge_raw_prediction_min": ridge_raw_predictions,
            "ridge_duration_min": ridge_predictions,
        }
    )

    ridge_evaluation["absolute_error_min"] = (
        ridge_evaluation["actual_duration_min"]
        - ridge_evaluation["ridge_duration_min"]
    ).abs()

    ridge_evaluation["squared_error_min2"] = (
        ridge_evaluation["actual_duration_min"]
        - ridge_evaluation["ridge_duration_min"]
    ) ** 2

    smape_denominator = (
        ridge_evaluation["actual_duration_min"].abs()
        + ridge_evaluation["ridge_duration_min"].abs()
    )

    ridge_evaluation["smape_component"] = np.where(
        smape_denominator.eq(0),
        0.0,
        2
        * ridge_evaluation["absolute_error_min"]
        / smape_denominator,
    )

    metrics = calculate_forecast_metrics(
        actual_values=ridge_evaluation[
            "actual_duration_min"
        ],
        predicted_values=ridge_evaluation[
            "ridge_duration_min"
        ],
    )

    negative_raw_prediction_count = int(
        (
            ridge_evaluation[
                "ridge_raw_prediction_min"
            ] < 0
        ).sum()
    )

    ridge_metrics_table = pd.DataFrame(
        [
            {
                "model": (
                    f"Ridge Regression "
                    f"(alpha={best_alpha})"
                ),
                "forecast_region": forecast_region,
                "test_start_date": (
                    ridge_evaluation["date"]
                    .min()
                    .date()
                ),
                "test_end_date": (
                    ridge_evaluation["date"]
                    .max()
                    .date()
                ),
                "test_rows": len(ridge_evaluation),
                "mae_min": metrics["mae_min"],
                "rmse_min": metrics["rmse_min"],
                "smape_percent": metrics[
                    "smape_percent"
                ],
                "zero_duration_days_in_test": int(
                    ridge_evaluation[
                        "actual_duration_min"
                    ]
                    .eq(0)
                    .sum()
                ),
                "negative_raw_predictions_clipped_to_zero": (
                    negative_raw_prediction_count
                ),
            }
        ]
    )

    ridge_coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "standardized_coefficient": (
                best_ridge_model
                .named_steps["ridge"]
                .coef_
            ),
        }
    )

    ridge_coefficients["absolute_coefficient"] = (
        ridge_coefficients[
            "standardized_coefficient"
        ].abs()
    )

    ridge_coefficients = (
        ridge_coefficients
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    cv_results = (
        pd.DataFrame(
            ridge_search.cv_results_
        )
        .loc[
            :,
            [
                "param_ridge__alpha",
                "mean_test_score",
                "std_test_score",
                "mean_train_score",
            ],
        ]
        .copy()
    )

    cv_results = cv_results.rename(
        columns={
            "param_ridge__alpha": "alpha",
            "mean_test_score": (
                "mean_validation_mae_negative"
            ),
            "std_test_score": (
                "validation_mae_std_negative"
            ),
            "mean_train_score": (
                "mean_training_mae_negative"
            ),
        }
    )

    cv_results["mean_validation_mae_min"] = (
        -cv_results["mean_validation_mae_negative"]
    )

    cv_results["validation_mae_std_min"] = (
        cv_results[
            "validation_mae_std_negative"
        ].abs()
    )

    cv_results["mean_training_mae_min"] = (
        -cv_results["mean_training_mae_negative"]
    )

    cv_results = (
        cv_results
        .sort_values("mean_validation_mae_min")
        .reset_index(drop=True)
    )

    return (
        ridge_evaluation,
        ridge_metrics_table,
        ridge_coefficients,
        cv_results,
        best_ridge_model,
    )


def plot_ridge_vs_baseline(
    ridge_evaluation: pd.DataFrame,
    baseline_evaluation: pd.DataFrame,
    best_alpha: float,
    output_path: str | Path | None = None,
    plot_days: int = 60,
):
    """
    Plot actual test values together with Seasonal Naive and Ridge
    predictions for the final test days.

    Parameters
    ----------
    ridge_evaluation:
        Output of train_and_evaluate_ridge_regression().
    baseline_evaluation:
        Output of evaluate_seasonal_naive_baseline().
    best_alpha:
        Selected Ridge alpha shown in the legend.
    output_path:
        Optional PNG output path.
    plot_days:
        Number of final test days displayed.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    validate_required_columns(
        dataframe=ridge_evaluation,
        required_columns=[
            "date",
            "actual_duration_min",
            "ridge_duration_min",
        ],
    )

    validate_required_columns(
        dataframe=baseline_evaluation,
        required_columns=[
            "date",
            "seasonal_naive_duration_min",
        ],
    )

    if plot_days <= 0:
        raise ValueError(
            "plot_days must be a positive integer."
        )

    ridge_plot_data = (
        ridge_evaluation
        .sort_values("date")
        .tail(plot_days)
        .copy()
    )

    baseline_plot_data = (
        baseline_evaluation[
            [
                "date",
                "seasonal_naive_duration_min",
            ]
        ]
        .copy()
    )

    comparison_plot_data = ridge_plot_data.merge(
        baseline_plot_data,
        on="date",
        how="left",
        validate="one_to_one",
    )

    if comparison_plot_data[
        "seasonal_naive_duration_min"
    ].isna().any():
        raise ValueError(
            "Baseline predictions could not be aligned "
            "with Ridge test dates."
        )

    figure, axes = plt.subplots(
        figsize=(14, 7),
    )

    axes.plot(
        comparison_plot_data["date"],
        comparison_plot_data["actual_duration_min"],
        label="Actual daily duration",
        linewidth=2,
    )

    axes.plot(
        comparison_plot_data["date"],
        comparison_plot_data[
            "seasonal_naive_duration_min"
        ],
        label="Seasonal Naive prediction (t-7)",
        linewidth=1.7,
    )

    axes.plot(
        comparison_plot_data["date"],
        comparison_plot_data["ridge_duration_min"],
        label=f"Ridge prediction (alpha={best_alpha})",
        linewidth=1.8,
    )

    axes.set_title(
        "Kyiv City: Ridge Regression vs Seasonal Naive "
        f"(Last {len(comparison_plot_data)} Test Days)",
        fontsize=14,
    )

    axes.set_xlabel("Date")
    axes.set_ylabel("Total alert duration, minutes")

    axes.grid(
        True,
        alpha=0.3,
    )

    axes.legend()

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    return figure, axes



def run_rolling_origin_backtest(
    feature_data: pd.DataFrame,
    forecast_region: str,
    feature_columns: list[str],
    final_test_start_date: str | pd.Timestamp,
    alpha_candidates: list[float],
    time_series_splits: int,
    n_folds: int,
    test_size_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run expanding-window rolling-origin backtesting for Ridge Regression
    and the Seasonal Naive baseline.

    The final test period is excluded completely. Historical folds use
    only observations available before their own test window.
    """
    required_columns = [
        "date",
        "target",
        "seasonal_naive_prediction",
        *feature_columns,
    ]

    validate_required_columns(
        dataframe=feature_data,
        required_columns=required_columns,
    )

    if not feature_columns:
        raise ValueError(
            "feature_columns cannot be empty."
        )

    if n_folds <= 0:
        raise ValueError(
            "n_folds must be a positive integer."
        )

    if test_size_days <= 0:
        raise ValueError(
            "test_size_days must be a positive integer."
        )

    final_test_start_date = pd.to_datetime(
        final_test_start_date,
        errors="coerce",
    )

    if pd.isna(final_test_start_date):
        raise ValueError(
            "final_test_start_date must be a valid date."
        )

    final_test_start_date = final_test_start_date.normalize()

    backtest_data = feature_data.copy()

    backtest_data["date"] = pd.to_datetime(
        backtest_data["date"],
        errors="coerce",
    ).dt.normalize()

    numeric_columns = [
        "target",
        "seasonal_naive_prediction",
        *feature_columns,
    ]

    backtest_data[numeric_columns] = (
        backtest_data[numeric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if backtest_data["date"].isna().any():
        raise ValueError(
            "feature_data contains invalid dates."
        )

    if backtest_data[numeric_columns].isna().any().any():
        raise ValueError(
            "feature_data contains missing or invalid "
            "target, baseline, or feature values."
        )

    if not np.isfinite(
        backtest_data[numeric_columns].to_numpy()
    ).all():
        raise ValueError(
            "feature_data contains non-finite values."
        )

    if (backtest_data["target"] < 0).any():
        raise ValueError(
            "feature_data contains negative target values."
        )

    backtest_data = (
        backtest_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if backtest_data["date"].duplicated().any():
        raise ValueError(
            "feature_data contains duplicate dates."
        )

    if final_test_start_date not in pd.DatetimeIndex(
        backtest_data["date"]
    ):
        raise ValueError(
            "final_test_start_date is not present "
            "in feature_data."
        )

    historical_data = (
        backtest_data.loc[
            backtest_data["date"].lt(final_test_start_date)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if historical_data.empty:
        raise ValueError(
            "No historical observations remain before "
            "the final test period."
        )

    required_test_rows = n_folds * test_size_days

    if len(historical_data) <= required_test_rows:
        raise ValueError(
            "Not enough historical observations for the "
            "requested rolling-origin folds."
        )

    first_test_start_index = (
        len(historical_data)
        - required_test_rows
    )

    if first_test_start_index <= time_series_splits:
        raise ValueError(
            "The first training window is too short for "
            "the requested TimeSeriesSplit configuration."
        )

    fold_metric_frames = []

    for fold_id in range(1, n_folds + 1):
        test_start_index = (
            first_test_start_index
            + (fold_id - 1) * test_size_days
        )

        test_end_index = (
            test_start_index
            + test_size_days
        )

        fold_train_data = (
            historical_data
            .iloc[:test_start_index]
            .copy()
            .reset_index(drop=True)
        )

        fold_test_data = (
            historical_data
            .iloc[test_start_index:test_end_index]
            .copy()
            .reset_index(drop=True)
        )

        if fold_train_data.empty or fold_test_data.empty:
            raise ValueError(
                f"Fold {fold_id} has an empty train or test dataset."
            )

        if (
            fold_train_data["date"].max()
            >= fold_test_data["date"].min()
        ):
            raise ValueError(
                f"Fold {fold_id} has overlapping train and test dates."
            )

        (
            baseline_evaluation,
            baseline_metrics,
        ) = evaluate_seasonal_naive_baseline(
            test_data=fold_test_data,
            forecast_region=forecast_region,
        )

        (
            ridge_evaluation,
            ridge_metrics,
            ridge_coefficients,
            ridge_cv_results,
            best_ridge_model,
        ) = train_and_evaluate_ridge_regression(
            train_data=fold_train_data,
            test_data=fold_test_data,
            feature_columns=feature_columns,
            forecast_region=forecast_region,
            alpha_candidates=alpha_candidates,
            time_series_splits=time_series_splits,
        )

        selected_alpha = float(
            best_ridge_model
            .named_steps["ridge"]
            .alpha
        )

        baseline_metrics = baseline_metrics.copy()
        ridge_metrics = ridge_metrics.copy()

        baseline_metrics["model_family"] = "Seasonal Naive"
        baseline_metrics["selected_alpha"] = np.nan

        ridge_metrics["model_family"] = "Ridge Regression"
        ridge_metrics["selected_alpha"] = selected_alpha

        if (
            "negative_raw_predictions_clipped_to_zero"
            not in baseline_metrics.columns
        ):
            baseline_metrics[
                "negative_raw_predictions_clipped_to_zero"
            ] = 0

        fold_metadata = {
            "fold_id": fold_id,
            "train_start_date": (
                fold_train_data["date"].min().date()
            ),
            "train_end_date": (
                fold_train_data["date"].max().date()
            ),
            "train_rows": len(fold_train_data),
            "test_start_date": (
                fold_test_data["date"].min().date()
            ),
            "test_end_date": (
                fold_test_data["date"].max().date()
            ),
            "test_rows": len(fold_test_data),
        }

        for column_name, value in fold_metadata.items():
            baseline_metrics[column_name] = value
            ridge_metrics[column_name] = value

        fold_metric_frames.extend(
            [
                baseline_metrics,
                ridge_metrics,
            ]
        )

    backtest_metrics = pd.concat(
        fold_metric_frames,
        ignore_index=True,
    )

    backtest_metrics = (
        backtest_metrics
        .sort_values(
            ["fold_id", "model_family"]
        )
        .reset_index(drop=True)
    )

    fold_summary = (
        backtest_metrics
        .pivot_table(
            index=[
                "fold_id",
                "train_start_date",
                "train_end_date",
                "train_rows",
                "test_start_date",
                "test_end_date",
                "test_rows",
            ],
            columns="model_family",
            values="mae_min",
            aggfunc="first",
        )
        .reset_index()
    )

    fold_summary.columns.name = None

    required_model_columns = [
        "Ridge Regression",
        "Seasonal Naive",
    ]

    missing_model_columns = [
        column
        for column in required_model_columns
        if column not in fold_summary.columns
    ]

    if missing_model_columns:
        raise ValueError(
            "Backtest results are missing model families: "
            f"{missing_model_columns}"
        )

    fold_summary["ridge_mae_improvement_min"] = (
        fold_summary["Seasonal Naive"]
        - fold_summary["Ridge Regression"]
    )

    fold_summary["ridge_mae_improvement_percent"] = (
        100
        * fold_summary["ridge_mae_improvement_min"]
        / fold_summary["Seasonal Naive"]
    )

    model_summary = (
        backtest_metrics
        .groupby(
            "model_family",
            as_index=False,
        )
        .agg(
            folds=("fold_id", "nunique"),
            mean_mae_min=("mae_min", "mean"),
            median_mae_min=("mae_min", "median"),
            std_mae_min=("mae_min", "std"),
            mean_rmse_min=("rmse_min", "mean"),
            mean_smape_percent=("smape_percent", "mean"),
        )
        .sort_values("mean_mae_min")
        .reset_index(drop=True)
    )

    return (
        backtest_metrics,
        fold_summary,
        model_summary,
    )


def plot_rolling_origin_backtest_mae(
    backtest_metrics: pd.DataFrame,
    forecast_region: str,
    output_path: str | Path | None = None,
):
    """
    Plot MAE for each model across rolling-origin backtest folds.
    """
    required_columns = [
        "fold_id",
        "model_family",
        "mae_min",
    ]

    validate_required_columns(
        dataframe=backtest_metrics,
        required_columns=required_columns,
    )

    plot_data = backtest_metrics.copy()

    plot_data["fold_id"] = pd.to_numeric(
        plot_data["fold_id"],
        errors="coerce",
    )

    plot_data["mae_min"] = pd.to_numeric(
        plot_data["mae_min"],
        errors="coerce",
    )

    if plot_data["fold_id"].isna().any():
        raise ValueError(
            "backtest_metrics contains invalid fold IDs."
        )

    if plot_data["mae_min"].isna().any():
        raise ValueError(
            "backtest_metrics contains invalid MAE values."
        )

    if (plot_data["mae_min"] < 0).any():
        raise ValueError(
            "backtest_metrics contains negative MAE values."
        )

    if plot_data["model_family"].isna().any():
        raise ValueError(
            "backtest_metrics contains missing model-family values."
        )

    if plot_data.duplicated(
        subset=["fold_id", "model_family"]
    ).any():
        raise ValueError(
            "Each model must have exactly one MAE value per fold."
        )

    figure, axes = plt.subplots(
        figsize=(11, 7),
    )

    for model_family, model_data in plot_data.groupby(
        "model_family",
        sort=True,
    ):
        model_data = (
            model_data
            .sort_values("fold_id")
            .reset_index(drop=True)
        )

        axes.plot(
            model_data["fold_id"],
            model_data["mae_min"],
            marker="o",
            linewidth=2,
            label=model_family,
        )

    fold_ids = sorted(
        plot_data["fold_id"].unique()
    )

    axes.set_title(
        f"{forecast_region}: Rolling-Origin Backtesting MAE",
        fontsize=14,
    )

    axes.set_xlabel("Historical backtest fold")
    axes.set_ylabel("MAE, minutes")

    axes.set_xticks(fold_ids)

    axes.grid(
        True,
        alpha=0.3,
    )

    axes.legend()

    figure.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

    return figure, axes