import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
import os


# %%
def visualize_score_comparisons_zoomed(
    postgres_data,
    pandas_metrics,
    pandas_status,
    postgres_status,
    save_dir=None,
    video_id=None,
    pattern_type=None,
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
        title_prefix = f"{pattern_type.capitalize()} Pattern - {video_id}\n"

    # Create a unique file prefix for this video and pattern
    file_prefix = f"{video_id}_{pattern_type}_"

    # Create a figure for the zoomed view
    plt.figure(figsize=(15, 8))

    # Get the last n_timestamps from PostgreSQL data
    if len(postgres_data) > n_timestamps:
        postgres_zoomed = postgres_data.iloc[-n_timestamps:]
    else:
        postgres_zoomed = postgres_data
        logging.info(
            f"Only {len(postgres_data)} PostgreSQL timestamps available, showing all"
        )

    # Get the time range of the zoomed data
    min_time = postgres_zoomed["timestamp_mnt"].min()
    max_time = postgres_zoomed["timestamp_mnt"].max()

    # IMPORTANT: Adjust time range to include pandas data if it exists
    # This ensures pandas data will be visible even if it's after PostgreSQL data
    if not pandas_metrics.empty:
        pandas_min_time = pandas_metrics["timestamp_mnt"].min()
        pandas_max_time = pandas_metrics["timestamp_mnt"].max()

        # Extend max_time to include pandas data if needed
        if pandas_max_time > max_time:
            max_time = pandas_max_time

        # Log the adjusted time range
        logging.info(
            f"Adjusted zoomed view time range to include pandas data: {min_time} to {max_time}"
        )
    else:
        logging.info(f"Original zoomed view time range: {min_time} to {max_time}")

    # Plot PostgreSQL data
    sns.lineplot(
        x=postgres_zoomed["timestamp_mnt"],
        y=postgres_zoomed["ds_score"],
        label="historical_ds_score",
        color="navy",
    )

    # Filter to match most recent PostgreSQL data
    postgres_recent = postgres_data[
        postgres_data["timestamp_mnt"]
        >= (postgres_data["timestamp_mnt"].max() - timedelta(minutes=5))
    ]

    # Plot recent PostgreSQL data with more emphasis
    if not postgres_recent.empty:
        sns.lineplot(
            x=postgres_recent["timestamp_mnt"],
            y=postgres_recent["ds_score"],
            label="src_postgres_ds_score",
            color="gold",
            alpha=1.0,
            linewidth=3,
            zorder=5,
        )

    # Plot pandas data if available - ALWAYS plot it when it exists
    if not pandas_metrics.empty:
        # Always plot the pandas data - don't filter by range
        sns.lineplot(
            x=pandas_metrics["timestamp_mnt"],
            y=pandas_metrics["ds_score"],
            label="sim_pandas_ds_score",
            color="crimson",
            linestyle="--",
            linewidth=2.5,
            zorder=6,
        )
        logging.info(f"Plotted {len(pandas_metrics)} pandas data points")

        # Print details of pandas data for debugging
        logging.info(
            f"Pandas data time range: {pandas_metrics['timestamp_mnt'].min()} to {pandas_metrics['timestamp_mnt'].max()}"
        )
        logging.info(
            f"Pandas DS scores: {pandas_metrics['ds_score'].min()} to {pandas_metrics['ds_score'].max()}"
        )

    # Plot the predicted score if available - ALWAYS plot when it exists
    if pandas_status["reference_predicted_avg_ds_score"] is not None:
        pred_score = pandas_status["reference_predicted_avg_ds_score"]

        # If we have pandas data, use its midpoint for the prediction star
        if not pandas_metrics.empty:
            pandas_min_time = pandas_metrics["timestamp_mnt"].min()
            pandas_max_time = pandas_metrics["timestamp_mnt"].max()

            # Define window edges for shading and prediction point
            now = pandas_max_time
            five_mins_ago = max(pandas_min_time, now - timedelta(minutes=5))

            # Calculate midpoint of the current window
            midpoint_timestamp = five_mins_ago + (now - five_mins_ago) / 2

            # ALWAYS plot the prediction point - don't check if it's in range
            plt.scatter(
                midpoint_timestamp,
                pred_score,
                color="limegreen",
                s=150,  # Increased size for better visibility
                marker="*",
                label="predicted_ds_score_regression",
                zorder=10,
            )

            # Add annotation
            plt.annotate(
                f"Predicted: {pred_score:.2f}",
                xy=(midpoint_timestamp, pred_score),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=10,  # Increased font size for better visibility
            )

            # Shade the current window
            plt.axvspan(
                five_mins_ago,
                now,
                alpha=0.2,
                color="limegreen",
                label="current_window_data_range",
            )

            # Print info for debugging
            print(f"\n=== Zoomed View Prediction Details ===")
            print(f"Current window: {five_mins_ago} to {now}")
            print(f"Midpoint for prediction: {midpoint_timestamp}")
            print(f"Predicted score: {pred_score}")
            print("=" * 50)
        else:
            # If no pandas data, use the latest PostgreSQL timestamp
            midpoint_timestamp = max_time - timedelta(minutes=2.5)

            plt.scatter(
                midpoint_timestamp,
                pred_score,
                color="limegreen",
                s=150,
                marker="*",
                label="predicted_ds_score_regression",
                zorder=10,
            )

            # Add annotation
            plt.annotate(
                f"Predicted: {pred_score:.2f}",
                xy=(midpoint_timestamp, pred_score),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                fontsize=10,
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

    # Ensure plot limits include ALL data
    if not pandas_metrics.empty:
        # Get axis instance
        ax = plt.gca()

        # Set xlim to include all data
        current_xlim = ax.get_xlim()
        ax.set_xlim(min_time, max_time)

        # Ensure y-axis has some padding
        y_data_points = list(postgres_zoomed["ds_score"])
        if not pandas_metrics.empty:
            y_data_points.extend(list(pandas_metrics["ds_score"]))

        if pandas_status["reference_predicted_avg_ds_score"] is not None:
            y_data_points.append(pandas_status["reference_predicted_avg_ds_score"])

        if y_data_points:
            min_y = min(y_data_points) * 0.99
            max_y = max(y_data_points) * 1.01
            ax.set_ylim(min_y, max_y)

    # Adjust layout
    plt.tight_layout()

    # Save the zoomed figure
    if save_dir:
        zoomed_file = os.path.join(
            save_dir, f"{file_prefix}zoomed_{n_timestamps}_timestamps.png"
        )
        plt.savefig(zoomed_file, dpi=300)
        logging.info(f"Saved zoomed visualization to: {zoomed_file}")
        return zoomed_file
    else:
        return None


# %%
def visualize_score_comparisons(
    postgres_data,
    pandas_metrics,
    pandas_status,
    postgres_status,
    save_dir=None,
    video_id=None,
    pattern_type=None,
):
    """
    Create visualizations comparing PostgreSQL and pandas data and save to files.

    Args:
        postgres_data (pandas.DataFrame): Data from PostgreSQL
        pandas_metrics (pandas.DataFrame): Metrics calculated with pandas
        pandas_status (dict): Hot or not status from pandas processing
        postgres_status (dict): Hot or not status from PostgreSQL processing
        save_dir (str): Directory to save visualizations, if None, only displays them
        video_id (str): The ID of the video for title information
        pattern_type (str): Type of activity pattern for title information

    Returns:
        list: Paths to saved figure files if save_dir is provided
    """
    import logging

    # Create save directory if it doesn't exist
    saved_files = []
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set seaborn style
    sns.set_style("whitegrid")

    # Log debug information about why reference_ds_score might be None
    logging.info(f"Debug reference_ds_score calculation:")
    if pandas_status["reference_predicted_avg_ds_score"] is None:
        logging.info("  Pandas reference_ds_score is None")
        # Check reference period data
        one_day_ago = pandas_metrics["timestamp_mnt"].min() - timedelta(days=1)
        five_mins_ago = pandas_metrics["timestamp_mnt"].max() - timedelta(minutes=5)
        reference_period_data = postgres_data[
            (postgres_data["timestamp_mnt"] >= one_day_ago)
            & (postgres_data["timestamp_mnt"] < five_mins_ago)
        ]
        logging.info(f"  Reference period has {len(reference_period_data)} rows")
        if len(reference_period_data) < 2:
            logging.info(
                "  Not enough data points in reference period for regression (need at least 2)"
            )
        else:
            logging.info("  Enough data points but regression may have failed")
    else:
        logging.info(
            f"  Pandas reference_ds_score = {pandas_status['reference_predicted_avg_ds_score']}"
        )

    if postgres_status["reference_predicted_avg_ds_score"] is None:
        logging.info("  PostgreSQL reference_ds_score is None")
    else:
        logging.info(
            f"  PostgreSQL reference_ds_score = {postgres_status['reference_predicted_avg_ds_score']}"
        )

    # Create a title with pattern and video information if provided
    title_prefix = ""
    if pattern_type and video_id:
        title_prefix = f"{pattern_type.capitalize()} Pattern - {video_id}\n"

    # Create a unique file prefix for this video and pattern
    file_prefix = f"{video_id}_{pattern_type}_"

    # ===== Create a combined figure with both plots =====
    # Create figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Left subplot: DS scores over time trend line ---
    # Plot PostgreSQL data
    sns.lineplot(
        x=postgres_data["timestamp_mnt"],
        y=postgres_data["ds_score"],
        label="historical_ds_score",
        color="navy",
        ax=ax1,
    )

    # Print pandas data points before plotting
    if not pandas_metrics.empty:
        # Print the pandas data points that will be plotted in red
        print("\n=== Pandas DS Score Data Points (Crimson Line) ===")
        print("Number of data points:", len(pandas_metrics))
        print("\nTimestamp and DS Score values:")
        for idx, row in pandas_metrics.iterrows():
            print(f"Timestamp: {row['timestamp_mnt']}, DS Score: {row['ds_score']}")

        # Print any additional relevant variables
        print("\nPandas Status Variables:")
        for key, value in pandas_status.items():
            print(f"{key}: {value}")
        print("=" * 50)

        # Get the time range of pandas data
        pandas_min_time = pandas_metrics["timestamp_mnt"].min()
        pandas_max_time = pandas_metrics["timestamp_mnt"].max()

        # Extract minute-level timestamps from pandas data for better matching
        # with PostgreSQL's minute-level granularity
        pandas_min_minute = pandas_min_time.replace(second=0, microsecond=0)
        pandas_max_minute = (pandas_max_time + timedelta(minutes=1)).replace(
            second=0, microsecond=0
        )

        print(f"\n=== Looking for PostgreSQL data at minute level ===")
        print(f"Original pandas time range: {pandas_min_time} to {pandas_max_time}")
        print(f"Minute-level search range: {pandas_min_minute} to {pandas_max_minute}")

        # Filter PostgreSQL data for the minute-level time range
        postgres_recent = postgres_data[
            (postgres_data["timestamp_mnt"] >= pandas_min_minute)
            & (postgres_data["timestamp_mnt"] <= pandas_max_minute)
        ]

        # If we don't find matching PostgreSQL data, plot the most recent 5 minutes instead
        if postgres_recent.empty:
            print(
                "\nNo PostgreSQL data points found for the same time range as pandas data"
            )
            print("Plotting the most recent 5 minutes of PostgreSQL data instead...")

            # Find the most recent timestamp in PostgreSQL data
            postgres_max_time = postgres_data["timestamp_mnt"].max()
            postgres_min_time = postgres_max_time - timedelta(minutes=5)

            # Get PostgreSQL data for the most recent 5 minutes
            postgres_recent = postgres_data[
                postgres_data["timestamp_mnt"] >= postgres_min_time
            ]

            print(
                f"Using PostgreSQL data from {postgres_min_time} to {postgres_max_time}"
            )
            print(f"Found {len(postgres_recent)} points in this range")

            # Print a sample of the data to verify
            if not postgres_recent.empty:
                print("\nSample of PostgreSQL data points being plotted:")
                for idx, row in postgres_recent.head(3).iterrows():
                    print(
                        f"  Timestamp: {row['timestamp_mnt']}, DS Score: {row['ds_score']}"
                    )

        # Plot PostgreSQL data for the selected time period
        if not postgres_recent.empty:
            sns.lineplot(
                x=postgres_recent["timestamp_mnt"],
                y=postgres_recent["ds_score"],
                label="src_postgres_ds_score",
                color="gold",
                alpha=1.0,
                linewidth=3,
                zorder=5,
                ax=ax1,
            )
            print(f"\nPlotted {len(postgres_recent)} PostgreSQL points")
        else:
            print("\nNo PostgreSQL data points found to plot")

        # Plot pandas data
        sns.lineplot(
            x=pandas_metrics["timestamp_mnt"],
            y=pandas_metrics["ds_score"],
            label="sim_pandas_ds_score",
            color="crimson",
            linestyle="--",
            ax=ax1,
        )

        # Plot the predicted score from linear regression if available
        if pandas_status["reference_predicted_avg_ds_score"] is not None:
            # Get the reference predicted score
            pred_score = pandas_status["reference_predicted_avg_ds_score"]

            # Use the actual pandas data timestamps instead of system time
            # Get the min and max timestamps from pandas_metrics
            if not pandas_metrics.empty:
                # Use the actual pandas data time range
                min_time = pandas_metrics["timestamp_mnt"].min()
                max_time = pandas_metrics["timestamp_mnt"].max()

                # Calculate midpoint based on the actual data instead of system time
                midpoint_timestamp = min_time + (max_time - min_time) / 2

                # Define a five minute window around our data
                now = max_time
                five_mins_ago = max(min_time, now - timedelta(minutes=5))

                print("\n=== Actual Pandas Data Time Range ===")
                print(f"Min timestamp: {min_time}")
                print(f"Max timestamp: {max_time}")
                print(f"Using midpoint: {midpoint_timestamp}")
                print("=" * 50)

                # Plot a specific point at the midpoint timestamp and predicted score
                ax1.scatter(
                    midpoint_timestamp,
                    pred_score,
                    color="limegreen",
                    s=100,  # Size of marker
                    marker="*",  # Star marker
                    label="predicted_ds_score_regression",
                    zorder=10,  # Make sure it's on top
                )

                # Add annotation showing exact values
                ax1.annotate(
                    f"Predicted: {pred_score:.2f}",
                    xy=(midpoint_timestamp, pred_score),
                    xytext=(10, 10),  # Offset text by 10 points
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    fontsize=9,
                )

                # Shade the data range where this prediction applies
                # Add light green shading for the current window
                ax1.axvspan(
                    five_mins_ago,
                    now,
                    alpha=0.2,
                    color="limegreen",
                    label="current_window_data_range",
                )

                # Print the current window details
                print("\n=== Prediction Window ===")
                print(f"Start: {five_mins_ago}")
                print(f"End: {now}")
                print(f"Midpoint: {midpoint_timestamp}")
                print(f"Predicted DS Score: {pred_score}")
                print("=" * 50)
            else:
                # Fallback for empty pandas data
                print("\nNo pandas data points to align prediction with.")

                # If no pandas data, use postgres data for visualization
                min_time = postgres_data["timestamp_mnt"].min()
                max_time = postgres_data["timestamp_mnt"].max()
                midpoint_timestamp = (
                    min_time + (max_time - min_time) * 0.9
                )  # Near the end

                ax1.scatter(
                    midpoint_timestamp,
                    pred_score,
                    color="limegreen",
                    s=100,  # Size of marker
                    marker="*",  # Star marker
                    label="predicted_ds_score_regression",
                    zorder=10,  # Make sure it's on top
                )

                # Add annotation
                ax1.annotate(
                    f"Predicted: {pred_score:.2f}",
                    xy=(midpoint_timestamp, pred_score),
                    xytext=(10, 10),  # Offset text by 10 points
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    fontsize=9,
                )
        else:
            print("\nNo predicted score available from linear regression.")

    ax1.set_title(f"{title_prefix}DS Score Trend Comparison", fontsize=14)
    ax1.set_xlabel("Time", fontsize=12)
    ax1.set_ylabel("DS Score", fontsize=12)
    ax1.legend(loc="best")

    # Rotate x-axis labels for better readability
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # --- Right subplot: Bar comparison for current and reference scores ---
    # Check if we have data for bar chart
    if (
        pandas_status["current_avg_ds_score"] is not None
        and postgres_status["current_avg_ds_score"] is not None
    ):
        # Create DataFrame for better seaborn integration
        comp_data = pd.DataFrame(
            {
                "Metric": [
                    "current_ds_score",
                    "current_ds_score",
                    "reference_ds_score",
                    "reference_ds_score",
                ],
                "Implementation": [
                    "sim_pandas",
                    "postgresql",
                    "sim_pandas",
                    "postgresql",
                ],
                "Value": [
                    pandas_status["current_avg_ds_score"],
                    postgres_status["current_avg_ds_score"],
                    (
                        pandas_status["reference_predicted_avg_ds_score"]
                        if pandas_status["reference_predicted_avg_ds_score"] is not None
                        else 0
                    ),
                    (
                        postgres_status["reference_predicted_avg_ds_score"]
                        if postgres_status["reference_predicted_avg_ds_score"]
                        is not None
                        else 0
                    ),
                ],
            }
        )

        # Create grouped bar chart
        bar_plot = sns.barplot(
            x="Metric",
            y="Value",
            hue="Implementation",
            data=comp_data,
            palette=["crimson", "navy"],
            ax=ax2,
        )

        # Add values on top of the bars
        for bar in bar_plot.containers:
            bar_plot.bar_label(bar, fmt="%.2f")

        # Annotate None values
        for i, val in enumerate(comp_data["Value"]):
            if i >= 2 and (  # Only for reference DS scores
                (i == 2 and pandas_status["reference_predicted_avg_ds_score"] is None)
                or (
                    i == 3
                    and postgres_status["reference_predicted_avg_ds_score"] is None
                )
            ):
                bar_plot.annotate(
                    "None",
                    xy=(i % 2, 0.1),  # Position at the bottom of the bar
                    ha="center",
                    va="bottom",
                    color="red",
                    fontweight="bold",
                )

        ax2.set_title(f"{title_prefix}current_vs_reference_ds_score", fontsize=14)
        ax2.set_ylabel("DS Score", fontsize=12)
        ax2.legend(title="Implementation", loc="upper right")
    else:
        # If no data for bar chart, display a message
        ax2.text(
            0.5,
            0.5,
            "No current DS score data available",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax2.transAxes,
            fontsize=14,
        )
        ax2.set_title(f"{title_prefix}current_vs_reference_ds_score", fontsize=14)

    # Adjust layout and save combined figure
    plt.tight_layout()

    if save_dir:
        combined_file = os.path.join(save_dir, f"{file_prefix}combined_comparison.png")
        plt.savefig(combined_file, dpi=300)
        saved_files.append(combined_file)

    # Call the zoomed visualization after creating the main visualization
    if save_dir:
        zoomed_file = visualize_score_comparisons_zoomed(
            postgres_data,
            pandas_metrics,
            pandas_status,
            postgres_status,
            save_dir=save_dir,
            video_id=video_id,
            pattern_type=pattern_type,
            n_timestamps=100,
        )
        if zoomed_file:
            saved_files.append(zoomed_file)

    return saved_files if save_dir else None
