import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import os
import pytz


# %%
def visualize_score_comparisons_zoomed(
    postgres_data,
    pandas_metrics,
    pandas_status,
    postgres_status,
    save_dir=None,
    video_id="test",
    pattern_type="default",
    n_timestamps=100,
):
    """
    Create a zoomed-in visualization of the DS score comparison focused on the last n_timestamps.
    This function creates a single plot focusing on recent data, including the current window
    and prediction.

    Args:
        postgres_data (pandas.DataFrame): Data from PostgreSQL
        pandas_metrics (pandas.DataFrame): Metrics calculated with pandas
        pandas_status (dict): Hot or not status from pandas processing
        postgres_status (dict): Hot or not status from PostgreSQL processing
        save_dir (str): Directory to save visualizations, if None, only displays them
        video_id (str): The ID of the video for title information
        pattern_type (str): Type of activity pattern for title information
        n_timestamps (int): Number of most recent timestamps to include

    Returns:
        str: Path to saved figure file if save_dir is provided, otherwise None
    """
    import logging

    if postgres_data.empty:
        logging.warning("No PostgreSQL data available for zoom visualization")
        return None

    # Create a title with pattern and video information if provided
    title_prefix = ""
    if pattern_type and video_id:
        # Use plain text for hot status (no emojis)
        if postgres_status and postgres_status.get("hot_or_not") is not None:
            hot_status = "HOT" if postgres_status["hot_or_not"] else "NOT HOT"
            title_prefix = (
                f"{pattern_type.capitalize()} Pattern - {video_id} - {hot_status}\n"
            )
        else:
            title_prefix = f"{pattern_type.capitalize()} Pattern - {video_id}\n"

    # Create a unique file prefix for this video and pattern
    file_prefix = f"{video_id}_{pattern_type}_"

    # Create a figure for the zoomed view
    plt.figure(figsize=(15, 8))

    # Get the last n_timestamps from PostgreSQL data
    postgres_data_sorted = postgres_data.sort_values("timestamp_mnt")
    if len(postgres_data_sorted) > n_timestamps:
        postgres_zoomed = postgres_data_sorted.iloc[-n_timestamps:]
    else:
        postgres_zoomed = postgres_data_sorted
        logging.info(
            f"Only {len(postgres_data)} PostgreSQL timestamps available, showing all"
        )

    # Get the time range of the zoomed data
    min_time = postgres_zoomed["timestamp_mnt"].min()
    max_time = postgres_zoomed["timestamp_mnt"].max()

    logging.info(f"Zoomed view time range: {min_time} to {max_time}")

    # Plot PostgreSQL data
    sns.lineplot(
        x=postgres_zoomed["timestamp_mnt"],
        y=postgres_zoomed["ds_score"],
        label="PostgreSQL DS Score (Historical)",
        color="navy",
    )

    # Filter to match most recent PostgreSQL data
    five_mins_ago = max_time - timedelta(minutes=5)
    postgres_recent = postgres_data_sorted[
        postgres_data_sorted["timestamp_mnt"] >= five_mins_ago
    ]

    # Plot recent PostgreSQL data with more emphasis
    if not postgres_recent.empty:
        sns.lineplot(
            x=postgres_recent["timestamp_mnt"],
            y=postgres_recent["ds_score"],
            label="PostgreSQL DS Score (Recent 5min)",
            color="gold",
            alpha=1.0,
            linewidth=3,
            zorder=5,
        )

    # Check if pandas timestamps have the same timezone awareness as postgres
    if not pandas_metrics.empty:
        pg_has_tz = postgres_data["timestamp_mnt"].iloc[0].tzinfo is not None
        pd_has_tz = pandas_metrics["timestamp_mnt"].iloc[0].tzinfo is not None

        # Make pandas timestamps timezone-aware/naive to match postgres timestamps
        if pd_has_tz != pg_has_tz:
            pandas_metrics = pandas_metrics.copy()
            if pg_has_tz and not pd_has_tz:
                # Make pandas timestamps timezone-aware
                pandas_metrics["timestamp_mnt"] = pandas_metrics["timestamp_mnt"].apply(
                    lambda x: x.replace(tzinfo=pytz.UTC)
                )
            elif not pg_has_tz and pd_has_tz:
                # Make pandas timestamps timezone-naive
                pandas_metrics["timestamp_mnt"] = pandas_metrics["timestamp_mnt"].apply(
                    lambda x: x.replace(tzinfo=None)
                )

        # Filter pandas data to match the zoomed range
        pandas_in_range = pandas_metrics[
            (pandas_metrics["timestamp_mnt"] >= min_time)
            & (pandas_metrics["timestamp_mnt"] <= max_time)
        ]

        if not pandas_in_range.empty:
            sns.lineplot(
                x=pandas_in_range["timestamp_mnt"],
                y=pandas_in_range["ds_score"],
                label="Pandas DS Score",
                color="crimson",
                linestyle="--",
            )
            logging.info(f"Plotted {len(pandas_in_range)} pandas points in zoomed view")
        else:
            logging.info("No pandas data points within the zoomed time range")

            # Since pandas data might be right after the PostgreSQL data,
            # still include it in the plot if it's close to the max time
            if (
                pandas_metrics["timestamp_mnt"].min() - max_time
            ).total_seconds() < 3600:  # Within 1 hour
                sns.lineplot(
                    x=pandas_metrics["timestamp_mnt"],
                    y=pandas_metrics["ds_score"],
                    label="Pandas DS Score (After Zoom Range)",
                    color="crimson",
                    linestyle="--",
                )
                logging.info(
                    f"Plotted {len(pandas_metrics)} pandas points after zoomed range"
                )

    # If we have hot status information, add a reference line
    if postgres_status and postgres_status.get("current_avg_ds_score") is not None:
        current_avg = postgres_status["current_avg_ds_score"]
        plt.axhline(
            y=current_avg,
            color="blue",
            linestyle=":",
            label=f"Current Avg DS Score: {current_avg:.4f}",
        )

        # Add predicted score line if available
        if postgres_status.get("reference_predicted_avg_ds_score") is not None:
            pred_score = postgres_status["reference_predicted_avg_ds_score"]
            plt.axhline(
                y=pred_score,
                color="red",
                linestyle=":",
                label=f"Predicted DS Score: {pred_score:.4f}",
            )

            # Shade the current window
            plt.axvspan(
                five_mins_ago,
                max_time,
                alpha=0.2,
                color="lightblue",
                label="Current Window (5 min)",
            )

    # Set titles and labels
    plt.title(
        f"{title_prefix}DS Score Trend (Last {n_timestamps} Timestamps)", fontsize=14
    )
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("DS Score", fontsize=12)
    plt.legend(loc="best")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha="right")

    # Adjust layout
    plt.tight_layout()

    # Save the zoomed figure
    if save_dir:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        zoomed_file = os.path.join(
            save_dir, f"{file_prefix}zoomed_{n_timestamps}_timestamps_{timestamp}.png"
        )
        plt.savefig(zoomed_file, dpi=300)
        logging.info(f"Saved zoomed visualization to: {zoomed_file}")
        plt.close()
        return zoomed_file
    else:
        plt.close()
        return None


# %%
def visualize_score_comparisons(
    postgres_data,
    pandas_metrics,
    pandas_status,
    postgres_status,
    save_dir=None,
    video_id="test",
    pattern_type="default",
):
    """
    Create visualizations comparing PostgreSQL and pandas processing results.

    Args:
        postgres_data (pandas.DataFrame): Data retrieved from PostgreSQL
        pandas_metrics (pandas.DataFrame): Metrics calculated with pandas
        pandas_status (dict): Hot or not status calculated with pandas
        postgres_status (dict): Hot or not status calculated with PostgreSQL
        save_dir (str): Directory to save visualizations
        video_id (str): Video ID for naming files
        pattern_type (str): Activity pattern type for naming files

    Returns:
        list: Paths to saved visualizations
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    import numpy as np
    import os
    from datetime import datetime, timedelta
    import pytz
    import logging

    # Check if we have enough data to visualize
    if postgres_data.empty or pandas_metrics.empty:
        print("Not enough data for visualization")
        return []

    # Create output directory if it doesn't exist
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set up plot style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("Set2")

    # Check timezone settings of timestamps
    pg_has_tz = postgres_data["timestamp_mnt"].iloc[0].tzinfo is not None
    pd_has_tz = pandas_metrics["timestamp_mnt"].iloc[0].tzinfo is not None

    print(f"PostgreSQL timestamps timezone-aware: {pg_has_tz}")
    print(f"Pandas timestamps timezone-aware: {pd_has_tz}")

    # Make pandas timestamps timezone-aware/naive to match postgres timestamps
    if pd_has_tz != pg_has_tz:
        pandas_metrics = pandas_metrics.copy()
        if pg_has_tz and not pd_has_tz:
            # Make pandas timestamps timezone-aware
            pandas_metrics["timestamp_mnt"] = pandas_metrics["timestamp_mnt"].apply(
                lambda x: x.replace(tzinfo=pytz.UTC)
            )
        elif not pg_has_tz and pd_has_tz:
            # Make pandas timestamps timezone-naive
            pandas_metrics["timestamp_mnt"] = pandas_metrics["timestamp_mnt"].apply(
                lambda x: x.replace(tzinfo=None)
            )

    # Figure 1: DS Score over time with hot status
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot PostgreSQL data
    ax.plot(
        postgres_data["timestamp_mnt"],
        postgres_data["ds_score"],
        marker="o",
        linestyle="-",
        alpha=0.7,
        label="PostgreSQL DS Score",
    )

    # Plot pandas data
    ax.plot(
        pandas_metrics["timestamp_mnt"],
        pandas_metrics["ds_score"],
        marker="x",
        linestyle="--",
        alpha=0.7,
        label="Pandas DS Score",
    )

    # Mark current window and hot/not hot status based on PostgreSQL data timestamps
    if postgres_status and postgres_status.get("current_avg_ds_score") is not None:
        # For the current window, find the most recent timestamps in postgres_data
        postgres_data_sorted = postgres_data.sort_values("timestamp_mnt")
        latest_timestamp = postgres_data_sorted["timestamp_mnt"].max()
        window_start = latest_timestamp - timedelta(minutes=5)

        current_window_pg = postgres_data[
            (postgres_data["timestamp_mnt"] >= window_start)
            & (postgres_data["timestamp_mnt"] <= latest_timestamp)
        ]

        if not current_window_pg.empty:
            # Highlight current window
            ax.axvspan(
                current_window_pg["timestamp_mnt"].min(),
                current_window_pg["timestamp_mnt"].max(),
                alpha=0.2,
                color="lightblue",
                label="Current Window (5 min)",
            )

            # Add horizontal line for current avg DS score
            ax.axhline(
                y=postgres_status["current_avg_ds_score"],
                color="blue",
                linestyle=":",
                label=f'Current Avg DS Score: {postgres_status["current_avg_ds_score"]:.4f}',
            )

            # Add horizontal line for reference predicted DS score
            if postgres_status["reference_predicted_avg_ds_score"] is not None:
                ax.axhline(
                    y=postgres_status["reference_predicted_avg_ds_score"],
                    color="red",
                    linestyle=":",
                    label=f'Reference Predicted DS Score: {postgres_status["reference_predicted_avg_ds_score"]:.4f}',
                )

    # Add hot status to title (without emojis to avoid font issues)
    title = f"DS Score Over Time - {pattern_type.capitalize()} Pattern\n"
    if postgres_status and postgres_status.get("hot_or_not") is not None:
        hot_status = "HOT" if postgres_status["hot_or_not"] else "NOT HOT"
        title += f"PostgreSQL Status: {hot_status}"

    if pandas_status and pandas_status.get("hot_or_not") is not None:
        pandas_hot = "HOT" if pandas_status["hot_or_not"] else "NOT HOT"
        title += f" | Pandas Status: {pandas_hot}"

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("DS Score")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Format x-axis dates
    date_format = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)

    plt.tight_layout()

    # Save the plot if directory is provided
    saved_files = []
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            save_dir, f"{video_id}_{pattern_type}_ds_score_{timestamp}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        saved_files.append(filename)
        print(f"Saved DS score plot to {filename}")

    # Figure 2: Cumulative metrics over time
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    # Plot 1: Cumulative Like CTR
    axes[0].plot(
        postgres_data["timestamp_mnt"],
        postgres_data["cumulative_like_ctr"],
        marker="o",
        linestyle="-",
        alpha=0.7,
        label="PostgreSQL",
    )
    axes[0].plot(
        pandas_metrics["timestamp_mnt"],
        pandas_metrics["cumulative_like_ctr"],
        marker="x",
        linestyle="--",
        alpha=0.7,
        label="Pandas",
    )
    axes[0].set_title("Cumulative Like CTR")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("CTR (%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cumulative Average Watch Percentage
    axes[1].plot(
        postgres_data["timestamp_mnt"],
        postgres_data["cumulative_average_percentage_watched"],
        marker="o",
        linestyle="-",
        alpha=0.7,
        label="PostgreSQL",
    )
    axes[1].plot(
        pandas_metrics["timestamp_mnt"],
        pandas_metrics["cumulative_average_percentage_watched"],
        marker="x",
        linestyle="--",
        alpha=0.7,
        label="Pandas",
    )
    axes[1].set_title("Cumulative Average Watch Percentage")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Watch Percentage")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Normalized Metrics
    axes[2].plot(
        postgres_data["timestamp_mnt"],
        postgres_data["normalized_cumulative_like_ctr"],
        marker="o",
        linestyle="-",
        alpha=0.7,
        label="PostgreSQL Norm CTR",
    )
    axes[2].plot(
        postgres_data["timestamp_mnt"],
        postgres_data["normalized_cumulative_watch_percentage"],
        marker="s",
        linestyle="-",
        alpha=0.7,
        label="PostgreSQL Norm Watch %",
    )
    axes[2].plot(
        pandas_metrics["timestamp_mnt"],
        pandas_metrics["normalized_cumulative_like_ctr"],
        marker="x",
        linestyle="--",
        alpha=0.7,
        label="Pandas Norm CTR",
    )
    axes[2].plot(
        pandas_metrics["timestamp_mnt"],
        pandas_metrics["normalized_cumulative_watch_percentage"],
        marker="+",
        linestyle="--",
        alpha=0.7,
        label="Pandas Norm Watch %",
    )
    axes[2].set_title("Normalized Metrics")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Normalized Value")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Regression for Reference Period
    # Get the reference period data (1 day ago to 5 mins ago)
    postgres_data_sorted = postgres_data.sort_values("timestamp_mnt")
    latest_timestamp = postgres_data_sorted["timestamp_mnt"].max()
    five_mins_ago = latest_timestamp - timedelta(minutes=5)
    one_day_ago = latest_timestamp - timedelta(days=1)

    reference_period_pg = postgres_data[
        (postgres_data["timestamp_mnt"] >= one_day_ago)
        & (postgres_data["timestamp_mnt"] < five_mins_ago)
    ]

    if len(reference_period_pg) >= 2:
        # Convert timestamps to epoch seconds for regression
        reference_period_pg = reference_period_pg.copy()
        reference_period_pg["timestamp_seconds"] = reference_period_pg[
            "timestamp_mnt"
        ].apply(lambda x: x.timestamp())

        # Perform simple linear regression
        from scipy import stats

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            reference_period_pg["timestamp_seconds"], reference_period_pg["ds_score"]
        )

        # Generate points for regression line
        x_range = np.linspace(
            reference_period_pg["timestamp_seconds"].min(),
            reference_period_pg["timestamp_seconds"].max(),
            100,
        )
        y_pred = slope * x_range + intercept

        # Convert back to datetime for plotting
        x_dates = [datetime.fromtimestamp(x) for x in x_range]
        if pg_has_tz:
            x_dates = [x.replace(tzinfo=pytz.UTC) for x in x_dates]

        # Plot reference period data points
        axes[3].scatter(
            reference_period_pg["timestamp_mnt"],
            reference_period_pg["ds_score"],
            marker="o",
            alpha=0.7,
            label="Reference Period Data",
        )

        # Plot regression line
        axes[3].plot(
            x_dates, y_pred, "r-", label=f"Regression Line (slope={slope:.6f})"
        )

        # If we have current scores, plot them for comparison
        if postgres_status and postgres_status.get("current_avg_ds_score") is not None:
            # Calculate midpoint of current window
            midpoint_time = five_mins_ago + (latest_timestamp - five_mins_ago) / 2

            # Calculate predicted value at midpoint
            midpoint_seconds = midpoint_time.timestamp()
            predicted_value = slope * midpoint_seconds + intercept

            # Plot predicted value
            axes[3].scatter(
                [midpoint_time],
                [predicted_value],
                marker="*",
                s=150,
                color="red",
                label=f"Predicted Value: {predicted_value:.4f}",
            )

            # Plot actual average
            axes[3].scatter(
                [midpoint_time],
                [postgres_status["current_avg_ds_score"]],
                marker="*",
                s=150,
                color="blue",
                label=f'Actual Avg: {postgres_status["current_avg_ds_score"]:.4f}',
            )

        axes[3].set_title("Reference Period Regression")
        axes[3].set_xlabel("Time")
        axes[3].set_ylabel("DS Score")
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(
            0.5,
            0.5,
            "Not enough data points for regression",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axes[3].transAxes,
        )

    # Format x-axis dates for all subplots
    for ax in axes:
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # Save the plot if directory is provided
    if save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            save_dir, f"{video_id}_{pattern_type}_metrics_{timestamp}.png"
        )
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        saved_files.append(filename)
        print(f"Saved metrics plot to {filename}")

    plt.close("all")  # Close all figures to prevent memory leaks

    return saved_files
