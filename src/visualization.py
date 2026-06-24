from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

REQUIRED_DAILY_COLUMNS = [
    "date",
    "region",
    "total_duration_min",
]

REQUIRED_MONTHLY_COLUMNS = [
    "month",
    "region",
    "total_duration_min",
    "total_duration_hours",
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
# Monthly aggregation
# ---------------------------------------------------------------------

def prepare_monthly_duration(
    daily_metrics: pd.DataFrame,
    selected_regions: list[str],
) -> pd.DataFrame:
    """
    Aggregate daily alert duration into monthly total duration.

    Parameters
    ----------
    daily_metrics:
        Daily region-level metrics created by aggregation.py.
    selected_regions:
        Regions to include in the monthly comparison.

    Returns
    -------
    pd.DataFrame
        Monthly duration table with columns:
        month, region, total_duration_min, alert_start_count,
        total_duration_hours.
    """
    validate_required_columns(
        dataframe=daily_metrics,
        required_columns=REQUIRED_DAILY_COLUMNS,
    )

    available_regions = set(daily_metrics["region"].dropna().unique())

    missing_regions = [
        region
        for region in selected_regions
        if region not in available_regions
    ]

    if missing_regions:
        raise ValueError(
            "Selected regions were not found in daily metrics: "
            f"{missing_regions}"
        )

    selected_daily_metrics = (
        daily_metrics.loc[
            daily_metrics["region"].isin(selected_regions)
        ]
        .copy()
    )

    if selected_daily_metrics.empty:
        raise ValueError(
            "No records were found for selected regions."
        )

    selected_daily_metrics["month"] = (
        selected_daily_metrics["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    aggregation_mapping = {
        "total_duration_min": "sum",
    }

    if "alert_start_count" in selected_daily_metrics.columns:
        aggregation_mapping["alert_start_count"] = "sum"

    monthly_duration = (
        selected_daily_metrics
        .groupby(
            ["month", "region"],
            as_index=False,
        )
        .agg(aggregation_mapping)
    )

    monthly_duration["total_duration_hours"] = (
        monthly_duration["total_duration_min"] / 60
    )

    monthly_duration = (
        monthly_duration
        .sort_values(["region", "month"])
        .reset_index(drop=True)
    )

    return monthly_duration


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_monthly_total_alert_duration(
    monthly_duration: pd.DataFrame,
    selected_regions: list[str],
    output_path: str | Path | None = None,
):
    """
    Plot monthly total air-raid alert duration for selected regions.

    Parameters
    ----------
    monthly_duration:
        Output of prepare_monthly_duration().
    selected_regions:
        Ordered region names used for plot lines and legend.
    output_path:
        Optional output PNG path. If provided, the figure is saved.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    validate_required_columns(
        dataframe=monthly_duration,
        required_columns=REQUIRED_MONTHLY_COLUMNS,
    )

    figure, axes = plt.subplots(figsize=(14, 7))

    for region in selected_regions:
        region_monthly_data = (
            monthly_duration.loc[
                monthly_duration["region"].eq(region)
            ]
            .sort_values("month")
        )

        if region_monthly_data.empty:
            raise ValueError(
                f"No monthly data found for region: {region}"
            )

        axes.plot(
            region_monthly_data["month"],
            region_monthly_data["total_duration_hours"],
            label=region,
            linewidth=1.8,
        )

    axes.set_title(
        "Monthly Total Air Raid Alert Duration in Selected Regions",
        fontsize=14,
    )

    axes.set_xlabel("Month")
    axes.set_ylabel("Total alert duration, hours")

    axes.grid(
        True,
        alpha=0.3,
    )

    axes.legend(
        title="Region",
        loc="upper left",
    )

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
# Regional total-duration ranking
# ---------------------------------------------------------------------

def prepare_top_regions_by_total_duration(
    daily_metrics: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank regions by total air-raid alert duration over the full period.

    Parameters
    ----------
    daily_metrics:
        Daily region-level metrics created by aggregation.py.
    top_n:
        Number of highest-ranked regions to return.

    Returns
    -------
    pd.DataFrame
        Regional summary table sorted in ascending order of total duration.
        Ascending order is intentional: it is convenient for a horizontal
        bar chart where the largest bar appears at the top.
    """
    validate_required_columns(
        dataframe=daily_metrics,
        required_columns=[
            "region",
            "total_duration_min",
            "alert_start_count",
            "active_alert_count",
        ],
    )

    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    region_total_duration = (
        daily_metrics
        .groupby(
            "region",
            as_index=False,
        )
        .agg(
            total_duration_min=(
                "total_duration_min",
                "sum",
            ),
            total_alert_starts=(
                "alert_start_count",
                "sum",
            ),
            total_active_alert_days=(
                "active_alert_count",
                lambda values: (values > 0).sum(),
            ),
        )
    )

    region_total_duration["total_duration_hours"] = (
        region_total_duration["total_duration_min"] / 60
    )

    region_total_duration["total_duration_days"] = (
        region_total_duration["total_duration_hours"] / 24
    )

    top_regions = (
        region_total_duration
        .sort_values(
            by="total_duration_hours",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            by="total_duration_hours",
            ascending=True,
        )
        .reset_index(drop=True)
    )

    return top_regions


def plot_top_regions_by_total_duration(
    top_regions: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """
    Create a horizontal bar chart for regions with the greatest
    total air-raid alert duration.

    Parameters
    ----------
    top_regions:
        Output of prepare_top_regions_by_total_duration().
    output_path:
        Optional PNG output path. If provided, the figure is saved.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    validate_required_columns(
        dataframe=top_regions,
        required_columns=[
            "region",
            "total_duration_hours",
        ],
    )

    if top_regions.empty:
        raise ValueError("The top_regions DataFrame is empty.")

    figure, axes = plt.subplots(figsize=(12, 7))

    axes.barh(
        top_regions["region"],
        top_regions["total_duration_hours"],
    )

    axes.set_title(
        "Top 10 Regions by Total Air Raid Alert Duration",
        fontsize=14,
    )

    axes.set_xlabel("Total alert duration, hours")
    axes.set_ylabel("Region")

    axes.grid(
        axis="x",
        alpha=0.3,
    )

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
# Individual alert-duration distribution
# ---------------------------------------------------------------------

REQUIRED_EVENT_COLUMNS = [
    "region",
    "duration_min",
]


def prepare_selected_event_durations(
    cleaned_events: pd.DataFrame,
    selected_regions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare individual alert-duration data and summary statistics
    for selected regions.

    Parameters
    ----------
    cleaned_events:
        Cleaned event-level dataset created by preprocessing.py.
    selected_regions:
        Ordered list of regions to include.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        selected_events:
            Event-level duration data for selected regions.
        duration_summary:
            Summary statistics by region.
    """
    validate_required_columns(
        dataframe=cleaned_events,
        required_columns=REQUIRED_EVENT_COLUMNS,
    )

    available_regions = set(
        cleaned_events["region"]
        .dropna()
        .unique()
    )

    missing_regions = [
        region
        for region in selected_regions
        if region not in available_regions
    ]

    if missing_regions:
        raise ValueError(
            "Selected regions were not found in cleaned event data: "
            f"{missing_regions}"
        )

    selected_events = (
        cleaned_events.loc[
            cleaned_events["region"].isin(selected_regions),
            ["region", "duration_min"],
        ]
        .copy()
    )

    selected_events["duration_min"] = pd.to_numeric(
        selected_events["duration_min"],
        errors="coerce",
    )

    invalid_duration_count = int(
        selected_events["duration_min"].isna().sum()
        + selected_events["duration_min"].le(0).sum()
    )

    if invalid_duration_count > 0:
        raise ValueError(
            "Selected event data contains missing or non-positive "
            "alert durations."
        )

    selected_events = (
        selected_events
        .sort_values(
            by=["region", "duration_min"],
        )
        .reset_index(drop=True)
    )

    duration_summary = (
        selected_events
        .groupby(
            "region",
            as_index=False,
        )
        .agg(
            event_count=(
                "duration_min",
                "count",
            ),
            mean_duration_min=(
                "duration_min",
                "mean",
            ),
            median_duration_min=(
                "duration_min",
                "median",
            ),
            p25_duration_min=(
                "duration_min",
                lambda values: values.quantile(0.25),
            ),
            p75_duration_min=(
                "duration_min",
                lambda values: values.quantile(0.75),
            ),
            p95_duration_min=(
                "duration_min",
                lambda values: values.quantile(0.95),
            ),
            max_duration_min=(
                "duration_min",
                "max",
            ),
        )
    )

    duration_summary = (
        duration_summary
        .set_index("region")
        .reindex(selected_regions)
        .reset_index()
    )

    return selected_events, duration_summary


def plot_alert_duration_distribution(
    selected_events: pd.DataFrame,
    selected_regions: list[str],
    output_path: str | Path | None = None,
):
    """
    Create a boxplot of individual alert durations for selected regions.

    Outlier markers are hidden only for readability. The source data
    are neither removed nor modified.

    Parameters
    ----------
    selected_events:
        Event-level duration data returned by
        prepare_selected_event_durations().
    selected_regions:
        Ordered list of region labels for the plot.
    output_path:
        Optional output PNG path.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    validate_required_columns(
        dataframe=selected_events,
        required_columns=REQUIRED_EVENT_COLUMNS,
    )

    boxplot_data = []

    for region in selected_regions:
        region_durations = (
            selected_events.loc[
                selected_events["region"].eq(region),
                "duration_min",
            ]
            .dropna()
            .to_numpy()
        )

        if len(region_durations) == 0:
            raise ValueError(
                f"No event durations found for region: {region}"
            )

        boxplot_data.append(region_durations)

    figure, axes = plt.subplots(figsize=(13, 7))

    axes.boxplot(
        boxplot_data,
        tick_labels=selected_regions,
        showfliers=False,
        showmeans=True,
    )

    axes.set_title(
        "Distribution of Individual Air Raid Alert Durations",
        fontsize=14,
    )

    axes.set_xlabel("Region")
    axes.set_ylabel("Alert duration, minutes")

    axes.grid(
        axis="y",
        alpha=0.3,
    )

    axes.tick_params(
        axis="x",
        rotation=20,
    )

    figure.text(
        0.01,
        0.01,
        (
            "Outlier markers are hidden for readability; "
            "underlying data are unchanged."
        ),
        fontsize=9,
    )

    figure.tight_layout(
        rect=[0, 0.04, 1, 1],
    )

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
# Alert-start month-hour heatmap
# ---------------------------------------------------------------------

MONTH_LABELS = [
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

REQUIRED_ALERT_START_COLUMNS = [
    "region",
    "started_at_local",
]


def prepare_month_hour_alert_start_counts(
    cleaned_events: pd.DataFrame,
    region: str,
    analysis_timezone: str,
) -> tuple[pd.DataFrame, int]:
    """
    Count air-raid alert starts by month of year and local hour of day.

    Parameters
    ----------
    cleaned_events:
        Cleaned event-level dataset created by preprocessing.py.
    region:
        Region whose alert starts should be analysed.
    analysis_timezone:
        IANA timezone used for local-hour conversion, for example
        'Europe/Kyiv'.

    Returns
    -------
    tuple[pd.DataFrame, int]
        month_hour_counts:
            A 12 × 24 table. Rows are calendar months; columns are
            local hours from 0 to 23.
        region_event_count:
            Number of cleaned events included for the selected region.
    """
    validate_required_columns(
        dataframe=cleaned_events,
        required_columns=REQUIRED_ALERT_START_COLUMNS,
    )

    region_events = (
        cleaned_events.loc[
            cleaned_events["region"].eq(region),
            ["region", "started_at_local"],
        ]
        .copy()
    )

    if region_events.empty:
        raise ValueError(
            f"No events were found for region: {region}"
        )

    region_events["started_at_local"] = (
        pd.to_datetime(
            region_events["started_at_local"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(analysis_timezone)
    )

    invalid_start_count = int(
        region_events["started_at_local"].isna().sum()
    )

    if invalid_start_count > 0:
        raise ValueError(
            "Some alert-start timestamps could not be parsed correctly. "
            f"Invalid timestamp count: {invalid_start_count}"
        )

    region_events["month_number"] = (
        region_events["started_at_local"]
        .dt.month
    )

    region_events["hour"] = (
        region_events["started_at_local"]
        .dt.hour
    )

    month_hour_counts = pd.crosstab(
        region_events["month_number"],
        region_events["hour"],
    )

    month_hour_counts = (
        month_hour_counts
        .reindex(
            index=range(1, 13),
            columns=range(0, 24),
            fill_value=0,
        )
    )

    month_hour_counts.index = MONTH_LABELS
    month_hour_counts.index.name = "month"
    month_hour_counts.columns.name = "hour"

    return month_hour_counts, len(region_events)


def plot_alert_start_month_hour_heatmap(
    month_hour_counts: pd.DataFrame,
    region: str,
    output_path: str | Path | None = None,
):
    """
    Plot a heatmap of air-raid alert starts by local month and hour.

    Parameters
    ----------
    month_hour_counts:
        Table returned by prepare_month_hour_alert_start_counts().
    region:
        Region name shown in the chart title.
    output_path:
        Optional PNG output path.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    if month_hour_counts.shape != (12, 24):
        raise ValueError(
            "month_hour_counts must have shape (12, 24). "
            f"Received: {month_hour_counts.shape}"
        )

    if (month_hour_counts.to_numpy() < 0).any():
        raise ValueError(
            "month_hour_counts cannot contain negative values."
        )

    figure, axes = plt.subplots(
        figsize=(15, 8),
    )

    heatmap = axes.imshow(
        month_hour_counts.to_numpy(),
        aspect="auto",
    )

    colorbar = figure.colorbar(
        heatmap,
        ax=axes,
    )

    colorbar.set_label(
        "Number of alert starts",
    )

    axes.set_title(
        f"{region}: Air Raid Alert Starts by Month and Local Hour",
        fontsize=14,
    )

    axes.set_xlabel(
        "Local hour of day",
    )

    axes.set_ylabel(
        "Month",
    )

    axes.set_xticks(
        range(len(month_hour_counts.columns)),
    )

    axes.set_xticklabels(
        month_hour_counts.columns.tolist(),
    )

    axes.set_yticks(
        range(len(month_hour_counts.index)),
    )

    axes.set_yticklabels(
        month_hour_counts.index.tolist(),
    )

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
# Daily alert-duration correlation heatmap
# ---------------------------------------------------------------------

REQUIRED_CORRELATION_COLUMNS = [
    "date",
    "region",
    "total_duration_min",
]


def prepare_daily_duration_correlation(
    daily_metrics: pd.DataFrame,
    selected_regions: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a date-by-region duration matrix and calculate the Pearson
    correlation matrix for daily total alert duration.

    Parameters
    ----------
    daily_metrics:
        Daily region-level metrics created by aggregation.py.
    selected_regions:
        Ordered list of regions to include in the correlation analysis.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        daily_duration_matrix:
            Rows are dates and columns are selected regions. Values are
            daily total alert duration in minutes.
        correlation_matrix:
            Pearson correlation matrix between selected regional series.
    """
    validate_required_columns(
        dataframe=daily_metrics,
        required_columns=REQUIRED_CORRELATION_COLUMNS,
    )

    available_regions = set(
        daily_metrics["region"]
        .dropna()
        .unique()
    )

    missing_regions = [
        region
        for region in selected_regions
        if region not in available_regions
    ]

    if missing_regions:
        raise ValueError(
            "Selected regions were not found in daily metrics: "
            f"{missing_regions}"
        )

    selected_daily_metrics = (
        daily_metrics.loc[
            daily_metrics["region"].isin(selected_regions),
            ["date", "region", "total_duration_min"],
        ]
        .copy()
    )

    selected_daily_metrics["date"] = pd.to_datetime(
        selected_daily_metrics["date"],
        errors="coerce",
    )

    invalid_date_count = int(
        selected_daily_metrics["date"].isna().sum()
    )

    if invalid_date_count > 0:
        raise ValueError(
            "Some daily metric dates could not be parsed correctly. "
            f"Invalid date count: {invalid_date_count}"
        )

    duplicate_date_region_rows = int(
        selected_daily_metrics.duplicated(
            subset=["date", "region"]
        ).sum()
    )

    if duplicate_date_region_rows > 0:
        raise ValueError(
            "Duplicate date-region rows were found. "
            f"Duplicate count: {duplicate_date_region_rows}"
        )

    daily_duration_matrix = (
        selected_daily_metrics
        .pivot(
            index="date",
            columns="region",
            values="total_duration_min",
        )
        .reindex(columns=selected_regions)
        .sort_index()
    )

    missing_values_by_region = (
        daily_duration_matrix
        .isna()
        .sum()
    )

    if missing_values_by_region.any():
        raise ValueError(
            "Missing daily values were found after pivoting:\n"
            f"{missing_values_by_region[
                missing_values_by_region > 0
            ]}"
        )

    correlation_matrix = daily_duration_matrix.corr(
        method="pearson"
    )

    if correlation_matrix.isna().any().any():
        raise ValueError(
            "Correlation matrix contains missing values. "
            "At least one regional time series may be constant."
        )

    return daily_duration_matrix, correlation_matrix


def plot_daily_duration_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """
    Plot a Pearson-correlation heatmap for regional daily alert duration.

    Parameters
    ----------
    correlation_matrix:
        Square correlation matrix returned by
        prepare_daily_duration_correlation().
    output_path:
        Optional PNG output path.

    Returns
    -------
    tuple
        Matplotlib figure and axes objects.
    """
    if correlation_matrix.empty:
        raise ValueError(
            "The correlation_matrix DataFrame is empty."
        )

    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError(
            "correlation_matrix must be square. "
            f"Received shape: {correlation_matrix.shape}"
        )

    if correlation_matrix.isna().any().any():
        raise ValueError(
            "correlation_matrix cannot contain missing values."
        )

    if not np.allclose(
        correlation_matrix.to_numpy(),
        correlation_matrix.to_numpy().T,
    ):
        raise ValueError(
            "correlation_matrix must be symmetric."
        )

    region_names = correlation_matrix.index.tolist()

    figure, axes = plt.subplots(
        figsize=(10, 8),
    )

    heatmap = axes.imshow(
        correlation_matrix.to_numpy(),
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    colorbar = figure.colorbar(
        heatmap,
        ax=axes,
    )

    colorbar.set_label(
        "Pearson correlation coefficient",
    )

    axes.set_title(
        "Correlation of Daily Total Alert Duration Between Selected Regions",
        fontsize=14,
    )

    axes.set_xlabel("Region")
    axes.set_ylabel("Region")

    axes.set_xticks(
        np.arange(len(region_names)),
    )

    axes.set_xticklabels(
        region_names,
        rotation=30,
        ha="right",
    )

    axes.set_yticks(
        np.arange(len(region_names)),
    )

    axes.set_yticklabels(
        region_names,
    )

    for row_index in range(len(region_names)):
        for column_index in range(len(region_names)):
            correlation_value = correlation_matrix.iloc[
                row_index,
                column_index,
            ]

            axes.text(
                column_index,
                row_index,
                f"{correlation_value:.2f}",
                ha="center",
                va="center",
                fontsize=10,
            )

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



def plot_daily_time_series_with_rolling_means(
    daily_metrics: pd.DataFrame,
    forecast_region: str,
    test_start_date: str | pd.Timestamp,
    output_path: str | Path | None = None,
):
    """
    Plot a complete daily time series for one region with trailing
    7-day and 28-day rolling means and a chronological train/test boundary.

    The rolling means are descriptive EDA values: they include the
    current day. Forecasting features remain leakage-free because
    their rolling means are shifted by one day separately.
    """
    required_columns = [
        "date",
        "region",
        "total_duration_min",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in daily_metrics.columns
    ]

    if missing_columns:
        raise KeyError(
            "daily_metrics is missing required columns: "
            f"{missing_columns}"
        )

    test_start_date = pd.to_datetime(
        test_start_date,
        errors="coerce",
    )

    if pd.isna(test_start_date):
        raise ValueError(
            "test_start_date must be a valid date."
        )

    time_series_data = (
        daily_metrics.loc[
            daily_metrics["region"].eq(forecast_region),
            ["date", "total_duration_min"],
        ]
        .copy()
    )

    if time_series_data.empty:
        raise ValueError(
            f"No observations found for region: {forecast_region}"
        )

    time_series_data["date"] = pd.to_datetime(
        time_series_data["date"],
        errors="coerce",
    )

    time_series_data["total_duration_min"] = pd.to_numeric(
        time_series_data["total_duration_min"],
        errors="coerce",
    )

    if time_series_data["date"].isna().any():
        raise ValueError(
            "The selected time series contains invalid dates."
        )

    if time_series_data["total_duration_min"].isna().any():
        raise ValueError(
            "The selected time series contains invalid durations."
        )

    if (time_series_data["total_duration_min"] < 0).any():
        raise ValueError(
            "The selected time series contains negative durations."
        )

    time_series_data = (
        time_series_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if time_series_data["date"].duplicated().any():
        raise ValueError(
            "The selected time series contains duplicate dates."
        )

    expected_dates = pd.date_range(
        start=time_series_data["date"].min(),
        end=time_series_data["date"].max(),
        freq="D",
    )

    if not pd.DatetimeIndex(
        time_series_data["date"]
    ).equals(expected_dates):
        raise ValueError(
            "The selected time series is not continuous by day."
        )

    if test_start_date not in pd.DatetimeIndex(
        time_series_data["date"]
    ):
        raise ValueError(
            "test_start_date is not present in the time series."
        )

    time_series_data["rolling_mean_7"] = (
        time_series_data["total_duration_min"]
        .rolling(
            window=7,
            min_periods=7,
        )
        .mean()
    )

    time_series_data["rolling_mean_28"] = (
        time_series_data["total_duration_min"]
        .rolling(
            window=28,
            min_periods=28,
        )
        .mean()
    )

    figure, axes = plt.subplots(
        figsize=(15, 8),
    )

    axes.plot(
        time_series_data["date"],
        time_series_data["total_duration_min"],
        label="Daily total duration",
        alpha=0.35,
        linewidth=1,
    )

    axes.plot(
        time_series_data["date"],
        time_series_data["rolling_mean_7"],
        label="7-day trailing mean",
        linewidth=1.8,
    )

    axes.plot(
        time_series_data["date"],
        time_series_data["rolling_mean_28"],
        label="28-day trailing mean",
        linewidth=2,
    )

    axes.axvline(
        test_start_date,
        label="Test period begins",
        linewidth=1.5,
    )

    maximum_duration = time_series_data[
        "total_duration_min"
    ].max()

    axes.text(
        time_series_data["date"].min(),
        maximum_duration * 0.96,
        "Training period",
        ha="left",
        va="top",
    )

    axes.text(
        test_start_date + pd.Timedelta(days=7),
        maximum_duration * 0.96,
        "Test period",
        ha="left",
        va="top",
    )

    axes.set_title(
        f"{forecast_region}: Daily Total Air Raid Alert Duration "
        "with Trailing Rolling Means",
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

    return time_series_data, figure, axes



def prepare_acf_pacf_lag_diagnostics(
    daily_metrics: pd.DataFrame,
    forecast_region: str,
    test_start_date: str | pd.Timestamp,
    max_lags: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare train-only ACF/PACF diagnostics for one daily time series.

    The final test period is excluded so that lag selection and feature
    justification use only observations available before forecasting.
    """
    required_columns = [
        "date",
        "region",
        "total_duration_min",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in daily_metrics.columns
    ]

    if missing_columns:
        raise KeyError(
            "daily_metrics is missing required columns: "
            f"{missing_columns}"
        )

    if max_lags <= 0:
        raise ValueError(
            "max_lags must be a positive integer."
        )

    test_start_date = pd.to_datetime(
        test_start_date,
        errors="coerce",
    )

    if pd.isna(test_start_date):
        raise ValueError(
            "test_start_date must be a valid date."
        )

    test_start_date = test_start_date.normalize()

    time_series_data = (
        daily_metrics.loc[
            daily_metrics["region"].eq(forecast_region),
            ["date", "total_duration_min"],
        ]
        .copy()
    )

    if time_series_data.empty:
        raise ValueError(
            f"No observations found for region: {forecast_region}"
        )

    time_series_data["date"] = pd.to_datetime(
        time_series_data["date"],
        errors="coerce",
    ).dt.normalize()

    time_series_data["total_duration_min"] = pd.to_numeric(
        time_series_data["total_duration_min"],
        errors="coerce",
    )

    if time_series_data["date"].isna().any():
        raise ValueError(
            "The selected time series contains invalid dates."
        )

    if time_series_data["total_duration_min"].isna().any():
        raise ValueError(
            "The selected time series contains invalid durations."
        )

    if (time_series_data["total_duration_min"] < 0).any():
        raise ValueError(
            "The selected time series contains negative durations."
        )

    time_series_data = (
        time_series_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if time_series_data["date"].duplicated().any():
        raise ValueError(
            "The selected time series contains duplicate dates."
        )

    expected_dates = pd.date_range(
        start=time_series_data["date"].min(),
        end=time_series_data["date"].max(),
        freq="D",
    )

    if not pd.DatetimeIndex(
        time_series_data["date"]
    ).equals(expected_dates):
        raise ValueError(
            "The selected time series is not continuous by day."
        )

    if test_start_date not in pd.DatetimeIndex(
        time_series_data["date"]
    ):
        raise ValueError(
            "test_start_date is not present in the time series."
        )

    training_series = (
        time_series_data.loc[
            time_series_data["date"].lt(test_start_date)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if training_series.empty:
        raise ValueError(
            "No training observations remain after excluding "
            "the final test period."
        )

    # The Yule-Walker PACF estimator requires the lag count to remain
    # below half of the available observations.
    if max_lags >= len(training_series) / 2:
        raise ValueError(
            "max_lags must be smaller than half the number of "
            "training observations."
        )

    duration_values = training_series[
        "total_duration_min"
    ].to_numpy(dtype=float)

    acf_values, acf_confidence_intervals = acf(
        duration_values,
        nlags=max_lags,
        alpha=0.05,
        fft=True,
    )

    pacf_values, pacf_confidence_intervals = pacf(
        duration_values,
        nlags=max_lags,
        alpha=0.05,
        method="ywm",
    )

    lag_diagnostics = pd.DataFrame(
        {
            "lag_days": range(max_lags + 1),
            "acf": acf_values,
            "acf_lower_95": acf_confidence_intervals[:, 0],
            "acf_upper_95": acf_confidence_intervals[:, 1],
            "pacf": pacf_values,
            "pacf_lower_95": pacf_confidence_intervals[:, 0],
            "pacf_upper_95": pacf_confidence_intervals[:, 1],
        }
    )

    lag_diagnostics["acf_significant_95"] = (
        lag_diagnostics["acf_lower_95"].gt(0)
        | lag_diagnostics["acf_upper_95"].lt(0)
    )

    lag_diagnostics["pacf_significant_95"] = (
        lag_diagnostics["pacf_lower_95"].gt(0)
        | lag_diagnostics["pacf_upper_95"].lt(0)
    )

    lag_diagnostics["approximate_zero_threshold"] = (
        1.96 / np.sqrt(len(training_series))
    )

    return training_series, lag_diagnostics


def plot_acf_pacf_lag_diagnostics(
    training_series: pd.DataFrame,
    max_lags: int,
    selected_lags: list[int],
    output_path: str | Path | None = None,
):
    """
    Plot ACF and PACF for a train-only daily time series.

    Selected lags are marked with vertical dashed lines to highlight
    forecast-related features such as lag_1, lag_7 and lag_14.
    """
    required_columns = [
        "date",
        "total_duration_min",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in training_series.columns
    ]

    if missing_columns:
        raise KeyError(
            "training_series is missing required columns: "
            f"{missing_columns}"
        )

    if max_lags <= 0:
        raise ValueError(
            "max_lags must be a positive integer."
        )

    if not selected_lags:
        raise ValueError(
            "selected_lags cannot be empty."
        )

    if any(
        lag <= 0 or lag > max_lags
        for lag in selected_lags
    ):
        raise ValueError(
            "Each selected lag must be between 1 and max_lags."
        )

    plot_data = training_series.copy()

    plot_data["date"] = pd.to_datetime(
        plot_data["date"],
        errors="coerce",
    )

    plot_data["total_duration_min"] = pd.to_numeric(
        plot_data["total_duration_min"],
        errors="coerce",
    )

    if plot_data["date"].isna().any():
        raise ValueError(
            "training_series contains invalid dates."
        )

    if plot_data["total_duration_min"].isna().any():
        raise ValueError(
            "training_series contains invalid durations."
        )

    if (plot_data["total_duration_min"] < 0).any():
        raise ValueError(
            "training_series contains negative durations."
        )

    plot_data = (
        plot_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if max_lags >= len(plot_data) / 2:
        raise ValueError(
            "max_lags must be smaller than half the number of "
            "training observations."
        )

    duration_values = plot_data[
        "total_duration_min"
    ].to_numpy(dtype=float)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 11),
        sharex=True,
    )

    plot_acf(
        duration_values,
        lags=max_lags,
        alpha=0.05,
        zero=False,
        fft=True,
        ax=axes[0],
    )

    plot_pacf(
        duration_values,
        lags=max_lags,
        alpha=0.05,
        zero=False,
        method="ywm",
        ax=axes[1],
    )

    for axes_item in axes:
        for lag in selected_lags:
            axes_item.axvline(
                lag,
                linestyle="--",
                alpha=0.45,
            )

        axes_item.set_xlabel("Lag, days")

        axes_item.grid(
            True,
            alpha=0.25,
        )

    axes[0].set_title(
        "Autocorrelation Function (ACF)",
        fontsize=14,
    )

    axes[1].set_title(
        "Partial Autocorrelation Function (PACF)",
        fontsize=14,
    )

    figure.suptitle(
        "Kyiv City: Daily Total Alert Duration Lag Diagnostics",
        fontsize=16,
        y=0.99,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

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



WEEKDAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def prepare_weekday_pattern(
    daily_metrics: pd.DataFrame,
    forecast_region: str,
    test_start_date: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare a train-only weekday summary for one region.

    The final test period is excluded so that calendar-feature analysis
    uses only observations available before forecasting.
    """
    required_columns = [
        "date",
        "region",
        "total_duration_min",
    ]

    validate_required_columns(
        dataframe=daily_metrics,
        required_columns=required_columns,
    )

    test_start_date = pd.to_datetime(
        test_start_date,
        errors="coerce",
    )

    if pd.isna(test_start_date):
        raise ValueError(
            "test_start_date must be a valid date."
        )

    test_start_date = test_start_date.normalize()

    weekday_data = (
        daily_metrics.loc[
            daily_metrics["region"].eq(forecast_region),
            ["date", "total_duration_min"],
        ]
        .copy()
    )

    if weekday_data.empty:
        raise ValueError(
            f"No observations found for region: {forecast_region}"
        )

    weekday_data["date"] = pd.to_datetime(
        weekday_data["date"],
        errors="coerce",
    ).dt.normalize()

    weekday_data["total_duration_min"] = pd.to_numeric(
        weekday_data["total_duration_min"],
        errors="coerce",
    )

    if weekday_data["date"].isna().any():
        raise ValueError(
            "The selected time series contains invalid dates."
        )

    if weekday_data["total_duration_min"].isna().any():
        raise ValueError(
            "The selected time series contains invalid durations."
        )

    if (weekday_data["total_duration_min"] < 0).any():
        raise ValueError(
            "The selected time series contains negative durations."
        )

    weekday_data = (
        weekday_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    if weekday_data["date"].duplicated().any():
        raise ValueError(
            "The selected time series contains duplicate dates."
        )

    expected_dates = pd.date_range(
        start=weekday_data["date"].min(),
        end=weekday_data["date"].max(),
        freq="D",
    )

    if not pd.DatetimeIndex(
        weekday_data["date"]
    ).equals(expected_dates):
        raise ValueError(
            "The selected time series is not continuous by day."
        )

    if test_start_date not in pd.DatetimeIndex(
        weekday_data["date"]
    ):
        raise ValueError(
            "test_start_date is not present in the time series."
        )

    training_weekday_data = (
        weekday_data.loc[
            weekday_data["date"].lt(test_start_date)
        ]
        .copy()
        .reset_index(drop=True)
    )

    if training_weekday_data.empty:
        raise ValueError(
            "No training observations remain after excluding "
            "the final test period."
        )

    training_weekday_data["weekday"] = (
        training_weekday_data["date"]
        .dt.day_name()
    )

    training_weekday_data["weekday"] = pd.Categorical(
        training_weekday_data["weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )

    weekday_summary = (
        training_weekday_data
        .groupby(
            "weekday",
            observed=True,
            as_index=False,
        )
        .agg(
            days=("date", "count"),
            mean_duration_min=(
                "total_duration_min",
                "mean",
            ),
            median_duration_min=(
                "total_duration_min",
                "median",
            ),
            std_duration_min=(
                "total_duration_min",
                "std",
            ),
            min_duration_min=(
                "total_duration_min",
                "min",
            ),
            max_duration_min=(
                "total_duration_min",
                "max",
            ),
            zero_duration_days=(
                "total_duration_min",
                lambda values: int(values.eq(0).sum()),
            ),
            zero_duration_share_percent=(
                "total_duration_min",
                lambda values: values.eq(0).mean() * 100,
            ),
        )
        .sort_values("weekday")
        .reset_index(drop=True)
    )

    if len(weekday_summary) != 7:
        raise ValueError(
            "Expected all seven weekdays in the training period."
        )

    return training_weekday_data, weekday_summary


def plot_weekday_pattern(
    weekday_summary: pd.DataFrame,
    forecast_region: str,
    output_path: str | Path | None = None,
):
    """
    Plot mean and median daily alert duration by weekday.
    """
    required_columns = [
        "weekday",
        "mean_duration_min",
        "median_duration_min",
    ]

    validate_required_columns(
        dataframe=weekday_summary,
        required_columns=required_columns,
    )

    plot_data = weekday_summary.copy()

    plot_data["weekday"] = pd.Categorical(
        plot_data["weekday"],
        categories=WEEKDAY_ORDER,
        ordered=True,
    )

    plot_data["mean_duration_min"] = pd.to_numeric(
        plot_data["mean_duration_min"],
        errors="coerce",
    )

    plot_data["median_duration_min"] = pd.to_numeric(
        plot_data["median_duration_min"],
        errors="coerce",
    )

    if plot_data["weekday"].isna().any():
        raise ValueError(
            "weekday_summary contains invalid weekday names."
        )

    if plot_data[
        ["mean_duration_min", "median_duration_min"]
    ].isna().any().any():
        raise ValueError(
            "weekday_summary contains invalid duration values."
        )

    if (
        plot_data[
            ["mean_duration_min", "median_duration_min"]
        ] < 0
    ).any().any():
        raise ValueError(
            "weekday_summary contains negative duration values."
        )

    plot_data = (
        plot_data
        .sort_values("weekday")
        .reset_index(drop=True)
    )

    if plot_data["weekday"].tolist() != WEEKDAY_ORDER:
        raise ValueError(
            "weekday_summary must contain Monday through Sunday "
            "exactly once and in the standard order."
        )

    figure, axes = plt.subplots(
        figsize=(13, 7),
    )

    axes.bar(
        plot_data["weekday"],
        plot_data["mean_duration_min"],
        label="Mean daily duration",
    )

    axes.plot(
        plot_data["weekday"],
        plot_data["median_duration_min"],
        marker="o",
        linewidth=2,
        label="Median daily duration",
    )

    axes.set_title(
        f"{forecast_region}: Daily Alert Duration by Day of Week "
        "(Training Period)",
        fontsize=14,
    )

    axes.set_xlabel("Day of week")
    axes.set_ylabel("Total alert duration, minutes")

    axes.grid(
        axis="y",
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