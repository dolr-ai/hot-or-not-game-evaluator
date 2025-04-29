# %% [markdown]
# Hot or Not Game Evaluator - Deterministic Testing
# This notebook provides deterministic tests for the hot-or-not game evaluator by:
# 1. Creating controlled video engagement patterns
# 2. Implementing pandas-based calculations
# 3. Comparing with SQL implementation
# 4. Running assertion tests

# %%
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg
from dotenv import load_dotenv
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import time
from IPython.display import display


# Load environment variables
load_dotenv("/Users/sagar/work/yral/hot-or-not-game-evaluator/test/.env")

# Database connection parameters
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} sslmode=require"

# %% [markdown]
## 1. Define Video Engagement Patterns
# We'll create several deterministic patterns for video engagement:
# 1. Steady High Engagement
# 2. Steady Low Engagement
# 3. Rising Engagement
# 4. Falling Engagement
# 5. Spike in Middle
# 6. Dip in Middle

# %%
# Constants for normalization (from metric_const table)
LIKE_CTR_CENTER = 0
LIKE_CTR_RANGE = 0.2
WATCH_PERCENTAGE_CENTER = 0
WATCH_PERCENTAGE_RANGE = 0.7


class VideoPattern:
    def __init__(self, video_id, pattern_func, description):
        self.video_id = video_id
        self.pattern_func = pattern_func
        self.description = description

    def generate_engagement(self, timestamp, base_timestamp):
        """Generate engagement metrics for a given timestamp"""
        minutes_elapsed = (timestamp - base_timestamp).total_seconds() / 60
        return self.pattern_func(minutes_elapsed)


def steady_high(t):
    """Steady high engagement pattern"""
    return {"watch_percentage": 90.0, "like_probability": 0.8}


def steady_low(t):
    """Steady low engagement pattern"""
    return {"watch_percentage": 30.0, "like_probability": 0.2}


def rising(t):
    """Rising engagement pattern"""
    base = min(0.9, 0.2 + t * 0.02)  # Rises from 20% to 90% over time
    return {"watch_percentage": base * 100, "like_probability": base}


def falling(t):
    """Falling engagement pattern"""
    base = max(0.2, 0.9 - t * 0.02)  # Falls from 90% to 20% over time
    return {"watch_percentage": base * 100, "like_probability": base}


def spike_middle(t):
    """Spike in middle pattern"""
    if 20 <= t <= 40:
        base = 0.9
    else:
        base = 0.3
    return {"watch_percentage": base * 100, "like_probability": base}


def dip_middle(t):
    """Dip in middle pattern"""
    if 20 <= t <= 40:
        base = 0.3
    else:
        base = 0.9
    return {"watch_percentage": base * 100, "like_probability": base}


# Define test patterns
TEST_PATTERNS = [
    VideoPattern("sgx-steady_high", steady_high, "Steady High Engagement"),
    # VideoPattern("sgx-steady_low", steady_low, "Steady Low Engagement"),
    # VideoPattern("sgx-rising", rising, "Rising Engagement"),
    # VideoPattern("sgx-falling", falling, "Falling Engagement"),
    # VideoPattern("sgx-spike", spike_middle, "Spike in Middle"),
    # VideoPattern("sgx-dip", dip_middle, "Dip in Middle"),
]

# %% [markdown]
## 2. Generate Deterministic Engagement Data
# We'll generate 1.5 days of data with specified minute intervals for each pattern.


# %%
def generate_deterministic_data(
    patterns, end_time, duration_days=1.5, events_per_minute=10, minute_interval=5
):
    """Generate deterministic engagement data for all patterns

    Args:
        patterns: List of VideoPattern objects to generate data for
        end_time: The end time for data generation (now)
        duration_days: How many days of historical data to generate
        events_per_minute: How many events to generate per minute
        minute_interval: Gap between data points in minutes (1=every minute, 5=every 5 minutes)

    Returns:
        DataFrame with engagement data spanning from (end_time - duration_days) to end_time
    """
    np.random.seed(42)  # Set seed for reproducibility

    all_data = []

    # Calculate start time (e.g., 1.5 days ago)
    start_time = end_time - timedelta(days=duration_days)

    # Mark the 1-day point for hot-or-not calculation
    one_day_mark = end_time - timedelta(
        days=1
    )  # This is the boundary for after_one_day flag

    print(f"Data generation timespan: {start_time} to {end_time}")
    print(f"One day mark: {one_day_mark}")

    # Calculate total minutes to generate
    total_minutes = int(duration_days * 24 * 60)

    # Control parameters to ensure we get the expected distribution
    early_data_minutes = int(
        (duration_days - 1.0) * 24 * 60
    )  # Minutes before one_day_mark
    recent_data_minutes = int(1.0 * 24 * 60)  # Minutes in the last day

    print(
        f"Early data minutes: {early_data_minutes}, Recent data minutes: {recent_data_minutes}"
    )

    # Generate data for each pattern
    for pattern in patterns:
        # Generate data at specified intervals
        for minute_offset in range(0, total_minutes, minute_interval):
            timestamp = start_time + timedelta(minutes=minute_offset)

            # Skip generation if we've passed the end time
            if timestamp > end_time:
                continue

            # Calculate time from pattern start for engagement curve
            minutes_elapsed = (timestamp - start_time).total_seconds() / 60

            # Get base engagement metrics for this minute
            engagement = pattern.generate_engagement(timestamp, start_time)

            # Generate events_per_minute events
            for _ in range(events_per_minute):
                liked = np.random.random() < engagement["like_probability"]

                # Set after_one_day flag correctly (True only for data in the last day)
                after_one_day = timestamp >= one_day_mark

                all_data.append(
                    {
                        "video_id": pattern.video_id,
                        "timestamp": timestamp,
                        "liked": liked,
                        "watch_percentage": engagement["watch_percentage"],
                        "after_one_day": after_one_day,
                    }
                )

    # Create DataFrame and sort by timestamp to simulate real-time data flow
    df = pd.DataFrame(all_data)
    df = df.sort_values("timestamp")

    # Verify data distribution
    early_data = df[df["timestamp"] < one_day_mark]
    recent_data = df[df["timestamp"] >= one_day_mark]
    print(
        f"Generated {len(early_data)} early data points and {len(recent_data)} recent data points"
    )

    # Double-check after_one_day flag
    flag_mismatch = len(df[(df["timestamp"] >= one_day_mark) != df["after_one_day"]])
    if flag_mismatch > 0:
        print(f"WARNING: {flag_mismatch} rows have inconsistent after_one_day flags!")

    return df


# %%
def populate_sql_database_incremental(
    df, batch_size=10000, compute_hot_or_not_after_day_one=True
):
    """Populate the SQL database with test data in batches, with hot-or-not calculation

    Args:
        df: DataFrame with engagement data
        batch_size: Number of rows to process in each batch (increased default)
        compute_hot_or_not_after_day_one: Whether to compute hot-or-not status after day 1
    """
    total_rows = len(df)

    # Sort data by timestamp to ensure correct processing order
    df = df.sort_values("timestamp")

    # 1. Get data distribution based on the after_one_day flag (more reliable than timestamp comparison)
    df_older_data = df[~df["after_one_day"]].copy()  # Use the flag directly
    df_recent_day = df[df["after_one_day"]].copy()  # Use the flag directly

    print(f"Older data (should be larger): {len(df_older_data)}")
    if not df_older_data.empty:
        print(
            f"Older data timestamp range: {df_older_data['timestamp'].min()} to {df_older_data['timestamp'].max()}"
        )
        display(df_older_data.head())

    print(f"Recent day data (should be smaller): {len(df_recent_day)}")
    if not df_recent_day.empty:
        print(
            f"Recent data timestamp range: {df_recent_day['timestamp'].min()} to {df_recent_day['timestamp'].max()}"
        )
        display(df_recent_day.head())

    # Connect once and reuse the connection
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            with tqdm(total=total_rows, desc="Populating SQL Database") as pbar:
                # Insert ALL older data in a SINGLE batch
                if not df_older_data.empty:
                    print("Inserting ALL older data in a single batch...")

                    # Prepare all data at once
                    values_list = [
                        "(%s::VARCHAR(255), %s::BOOLEAN, %s::NUMERIC(5,2))"
                    ] * len(df_older_data)

                    # Faster way to extract data in bulk
                    video_ids = df_older_data["video_id"].astype(str).tolist()
                    liked_values = df_older_data["liked"].astype(bool).tolist()
                    watch_percentages = (
                        df_older_data["watch_percentage"]
                        .astype(float)
                        .round(2)
                        .tolist()
                    )

                    args = []
                    # Interleave the values
                    for vid, liked, watch in zip(
                        video_ids, liked_values, watch_percentages
                    ):
                        args.extend([vid, liked, watch])

                    placeholders = ", ".join(values_list)
                    query = f"""
                        WITH batch_data(video_id, liked, watch_percentage) AS (
                            VALUES {placeholders}
                        )
                        SELECT hot_or_not_evaluator.update_counter(
                            video_id, liked, watch_percentage
                        ) FROM batch_data
                    """

                    # Execute the query for ALL older data in ONE batch
                    cur.execute(query, args)
                    conn.commit()
                    pbar.update(len(df_older_data))
                    print(
                        f"Inserted {len(df_older_data)} older data rows in a single batch"
                    )

                # Group the recent day data by timestamp chunking in 5-minute intervals
                if not df_recent_day.empty:
                    print("Now processing recent day data in 5-minute chunks...")

                    # Use .loc to avoid SettingWithCopyWarning
                    df_recent_day.loc[:, "timestamp_chunk"] = df_recent_day[
                        "timestamp"
                    ].dt.floor("5min")
                    time_chunks = df_recent_day.groupby("timestamp_chunk")
                    chunk_count = len(df_recent_day["timestamp_chunk"].unique())
                    print(f"Processing {chunk_count} time chunks for recent day data")

                    # Track the timestamp of the previous batch to detect 5-minute intervals
                    prev_batch_time = None

                    # Process the recent day data in 5-minute time-order batches
                    for timestamp, chunk_df in time_chunks:
                        if chunk_df.empty:
                            continue

                        # Process this entire 5-minute chunk at once
                        batch_time = chunk_df["timestamp"].iloc[0]
                        after_one_day = chunk_df["after_one_day"].iloc[0]

                        # Optimize: Pre-allocate and use faster methods
                        values_list = [
                            "(%s::VARCHAR(255), %s::BOOLEAN, %s::NUMERIC(5,2))"
                        ] * len(chunk_df)

                        # Faster way to extract data in bulk
                        video_ids = chunk_df["video_id"].astype(str).tolist()
                        liked_values = chunk_df["liked"].astype(bool).tolist()
                        watch_percentages = (
                            chunk_df["watch_percentage"].astype(float).round(2).tolist()
                        )

                        args = []
                        # Interleave the values
                        for vid, liked, watch in zip(
                            video_ids, liked_values, watch_percentages
                        ):
                            args.extend([vid, liked, watch])

                        # Build batch query
                        placeholders = ", ".join(values_list)
                        query = f"""
                            WITH batch_data(video_id, liked, watch_percentage) AS (
                                VALUES {placeholders}
                            )
                            SELECT hot_or_not_evaluator.update_counter(
                                video_id, liked, watch_percentage
                            ) FROM batch_data
                        """

                        # Execute batch query for the entire 5-minute chunk
                        cur.execute(query, args)
                        conn.commit()

                        # Update progress bar
                        pbar.update(len(chunk_df))
                        print(
                            f"Inserted {len(chunk_df)} rows for time chunk {timestamp}"
                        )

                        # Calculate hot-or-not after day 1 and every 5 minutes
                        if (
                            compute_hot_or_not_after_day_one
                            and after_one_day
                            and (
                                prev_batch_time is None
                                or (timestamp - prev_batch_time).total_seconds() >= 300
                            )
                        ):  # 5 minutes = 300 seconds
                            cur.execute(
                                "SELECT hot_or_not_evaluator.compute_hot_or_not()"
                            )
                            conn.commit()
                            print(f"Computed hot-or-not status after chunk {timestamp}")

                            # Get hot-or-not results but don't display them
                            _ = get_sql_results(conn)

                            # Update previous batch time
                            prev_batch_time = timestamp


def get_sql_results(conn=None):
    """Get hot-or-not results from SQL database

    Args:
        conn: Optional existing database connection

    Returns:
        DataFrame with hot-or-not results from SQL
    """
    close_conn = False
    if conn is None:
        conn = psycopg.connect(conn_string)
        close_conn = True

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, hot_or_not, current_avg_ds_score, reference_predicted_avg_ds_score
                FROM hot_or_not_evaluator.video_hot_or_not_status
                WHERE video_id LIKE 'sgx-%'
                """
            )
            results = cur.fetchall()

            return pd.DataFrame(
                results,
                columns=[
                    "video_id",
                    "hot_or_not",
                    "current_avg_ds_score",
                    "reference_predicted_avg_ds_score",
                ],
            )
    finally:
        if close_conn:
            conn.close()


# %% [markdown]
## 3. Implement Pandas-based Hot-or-Not Calculation


# %%
def calculate_metrics(df):
    """Calculate all metrics for each video and minute, matching the SQL update_counter function"""
    # Check if input dataframe is empty
    if df.empty:
        print("Warning: Empty dataframe passed to calculate_metrics")
        return pd.DataFrame()

    try:
        # Group by video_id and timestamp (minute truncated)
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df_copy = df.copy()
        df_copy["timestamp_mnt"] = df_copy["timestamp"].dt.floor("min")

        # print(f"Grouping {len(df_copy)} rows by video_id and timestamp_mnt")
        # print(
        #     f"Unique videos: {df_copy['video_id'].nunique()}, Unique timestamps: {df_copy['timestamp_mnt'].nunique()}"
        # )

        grouped = (
            df_copy.groupby(["video_id", "timestamp_mnt"])
            .agg({"liked": ["count", "sum"], "watch_percentage": ["mean"]})
            .reset_index()
        )

        # Check if grouped dataframe is empty after aggregation
        if grouped.empty:
            print("Warning: Grouped dataframe is empty after aggregation")
            return pd.DataFrame()

        # Flatten column names
        grouped.columns = [
            "video_id",
            "timestamp",
            "watch_count_mnt",
            "like_count_mnt",
            "average_percentage_watched_mnt",
        ]

        # print(f"After grouping: {len(grouped)} rows")

        # Calculate like_ctr_mnt (matches SQL CASE WHEN statement)
        grouped["like_ctr_mnt"] = grouped.apply(
            lambda row: (
                (row["like_count_mnt"] * 100.0) / row["watch_count_mnt"]
                if row["watch_count_mnt"] > 0
                else 0
            ),
            axis=1,
        )

        # Sort by video_id and timestamp for cumulative calculations (required for window functions)
        grouped = grouped.sort_values(["video_id", "timestamp"])

        # Calculate cumulative metrics for each video (similar to SQL cumulative calculations)
        video_ids = grouped["video_id"].unique()
        # print(f"Processing cumulative metrics for {len(video_ids)} videos")

        # Create a results dataframe to avoid SettingWithCopyWarning
        result_df = grouped.copy()
        # Initialize the new columns
        result_df["cumulative_like_count"] = 0.0
        result_df["cumulative_watch_count"] = 0.0
        result_df["cumulative_average_percentage_watched"] = 0.0
        result_df["cumulative_like_ctr"] = 0.0

        for video_id in video_ids:
            video_mask = grouped["video_id"] == video_id
            video_df = grouped.loc[video_mask]

            # Skip empty dataframes
            if video_df.empty:
                print(f"Warning: No data for video {video_id}")
                continue

            # Perform cumulative calculations on the entire video dataframe at once
            video_indices = result_df.index[video_mask]
            result_df.loc[video_indices, "cumulative_like_count"] = (
                video_df["like_count_mnt"].cumsum().values
            )
            result_df.loc[video_indices, "cumulative_watch_count"] = (
                video_df["watch_count_mnt"].cumsum().values
            )

            # Calculate weighted cumulative average percentage watched (matches SQL formula)
            watch_counts = video_df["watch_count_mnt"].values
            watch_percentages = video_df["average_percentage_watched_mnt"].values

            # Calculate weighted cumulative average
            cumulative_watch_sum = np.cumsum(watch_counts * watch_percentages)
            cumulative_watch_count = np.cumsum(watch_counts)
            cumulative_avg_perc = np.zeros_like(cumulative_watch_sum, dtype=float)

            # Avoid division by zero
            nonzero_mask = cumulative_watch_count > 0
            cumulative_avg_perc[nonzero_mask] = (
                cumulative_watch_sum[nonzero_mask]
                / cumulative_watch_count[nonzero_mask]
            )

            result_df.loc[video_indices, "cumulative_average_percentage_watched"] = (
                cumulative_avg_perc
            )

            # Calculate cumulative like CTR (matches SQL formula)
            like_counts = video_df["like_count_mnt"].values
            cumulative_like_sum = np.cumsum(like_counts)
            cumulative_like_ctr = np.zeros_like(cumulative_like_sum, dtype=float)

            # Avoid division by zero
            cumulative_like_ctr[nonzero_mask] = (
                cumulative_like_sum[nonzero_mask] * 100.0
            ) / cumulative_watch_count[nonzero_mask]

            result_df.loc[video_indices, "cumulative_like_ctr"] = cumulative_like_ctr

        # Initialize normalized columns and calculated values all at once
        result_df["normalized_cumulative_like_ctr"] = 0.0
        result_df["normalized_cumulative_watch_percentage"] = 0.0
        result_df["harmonic_mean"] = 0.0
        result_df["ds_score"] = 0.0

        # Process each row without using chained assignment
        for i, row in result_df.iterrows():
            # Calculate normalized like CTR
            if LIKE_CTR_RANGE != 0:
                norm_like_ctr = max(
                    0, (row["cumulative_like_ctr"] - LIKE_CTR_CENTER) / LIKE_CTR_RANGE
                )
            else:
                norm_like_ctr = 0
            result_df.loc[i, "normalized_cumulative_like_ctr"] = norm_like_ctr

            # Calculate normalized watch percentage
            if WATCH_PERCENTAGE_RANGE != 0:
                norm_watch_perc = max(
                    0,
                    (
                        row["cumulative_average_percentage_watched"]
                        - WATCH_PERCENTAGE_CENTER
                    )
                    / WATCH_PERCENTAGE_RANGE,
                )
            else:
                norm_watch_perc = 0
            result_df.loc[i, "normalized_cumulative_watch_percentage"] = norm_watch_perc

            # Calculate harmonic mean
            denominator = norm_like_ctr + norm_watch_perc + 2
            if denominator != 0:
                h_mean = ((norm_like_ctr + 1) * (norm_watch_perc + 1)) / denominator
            else:
                h_mean = 1
            result_df.loc[i, "harmonic_mean"] = h_mean
            result_df.loc[i, "ds_score"] = h_mean - 1

        # print(f"Final metrics dataframe has {len(result_df)} rows")
        return result_df

    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        import traceback

        traceback.print_exc()
        return pd.DataFrame()  # Return empty dataframe on error


def determine_hot_or_not(metrics_df, current_window=5, reference_window=1440):
    """Determine hot-or-not status for each video, matching SQL compute_hot_or_not function

    Args:
        metrics_df: DataFrame with calculated metrics
        current_window: Window size for current metrics in minutes (default: 5)
        reference_window: Window size for reference period in minutes (default: 1440 = 1 day)

    Returns:
        DataFrame with hot-or-not results for each video
    """
    # Get the latest timestamp in the data
    v_now = metrics_df["timestamp"].max()

    # Define time windows
    v_5_mins_ago = v_now - timedelta(minutes=current_window)
    v_1_day_ago = v_now - timedelta(minutes=reference_window)

    results = []
    previous_statuses = {}

    # Process each video, similar to SQL loop
    for video_id in metrics_df["video_id"].unique():
        # Initialize values, matching SQL variables
        v_current_avg_ds = None
        v_ref_slope = None
        v_ref_intercept = None
        v_ref_count = 0
        v_ref_predicted_avg_ds = None
        v_is_hot = None

        try:
            # Get the previous hot status to maintain if comparison can't be made
            v_previous_hot_status = previous_statuses.get(video_id, None)

            # Filter data for this video
            video_data = metrics_df[metrics_df["video_id"] == video_id]

            # Calculate current average ds_score (last 5 minutes)
            current_window_mask = (video_data["timestamp"] >= v_5_mins_ago) & (
                video_data["timestamp"] < v_now
            )
            current_window_data = video_data[current_window_mask]

            if not current_window_data.empty:
                v_current_avg_ds = current_window_data["ds_score"].mean()

            # Calculate OLS parameters for the reference period (1 day ago to 5 mins ago)
            reference_window_mask = (video_data["timestamp"] >= v_1_day_ago) & (
                video_data["timestamp"] < v_5_mins_ago
            )
            reference_window_data = video_data[reference_window_mask]

            # Only perform regression if we have at least 2 points
            if len(reference_window_data) >= 2:
                # Convert timestamp to epoch for regression
                reference_window_data_copy = reference_window_data.copy()
                reference_window_data_copy["epoch"] = reference_window_data_copy[
                    "timestamp"
                ].apply(lambda x: int(x.timestamp()))

                # Use sklearn for regression (equivalent to SQL's regr_slope and regr_intercept)
                from sklearn.linear_model import LinearRegression

                X = reference_window_data_copy["epoch"].values.reshape(-1, 1)
                y = reference_window_data_copy["ds_score"].values

                model = LinearRegression()
                model.fit(X, y)

                v_ref_slope = model.coef_[0]
                v_ref_intercept = model.intercept_
                v_ref_count = len(reference_window_data_copy)

                # Calculate predicted value at midpoint, matching SQL calculation
                if (
                    v_ref_count >= 2
                    and v_ref_slope is not None
                    and v_ref_intercept is not None
                ):
                    # Get the midpoint of current window
                    current_midpoint = v_now - timedelta(minutes=current_window / 2)
                    current_epoch = int(current_midpoint.timestamp())

                    # Calculate predicted value
                    v_ref_predicted_avg_ds = (
                        v_ref_slope * current_epoch + v_ref_intercept
                    )

            # Determine hot-or-not status (matching SQL logic)
            if v_current_avg_ds is not None and v_ref_predicted_avg_ds is not None:
                v_is_hot = v_current_avg_ds > v_ref_predicted_avg_ds
            else:
                v_is_hot = v_previous_hot_status

            # Save status for next evaluation
            previous_statuses[video_id] = v_is_hot

            # Add result to list
            results.append(
                {
                    "video_id": video_id,
                    "last_updated_mnt": v_now,
                    "hot_or_not": v_is_hot,
                    "current_avg_ds_score": v_current_avg_ds,
                    "reference_predicted_avg_ds_score": v_ref_predicted_avg_ds,
                }
            )

        except Exception as e:
            print(f"Error processing video {video_id} in determine_hot_or_not: {e}")
            # Add error result with previous status
            results.append(
                {
                    "video_id": video_id,
                    "last_updated_mnt": v_now,
                    "hot_or_not": v_previous_hot_status,
                    "current_avg_ds_score": None,
                    "reference_predicted_avg_ds_score": None,
                }
            )

    return pd.DataFrame(results)


# %% [markdown]
## 4. Implement Incremental Processing for Pandas


# %%
def process_pandas_incremental(df, minute_interval=5):
    """Process data incrementally in pandas, similar to SQL processing

    Args:
        df: DataFrame with engagement data
        minute_interval: Interval in minutes for processing batches

    Returns:
        Tuple of (metrics_df, hot_or_not_results_list)
    """
    print("Starting incremental pandas processing...")

    # Make a copy of the input dataframe to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Group data by timestamp at the specified minute_interval
    df_copy["timestamp_rounded"] = df_copy["timestamp"].dt.floor(
        f"{minute_interval}min"
    )
    time_groups = df_copy.groupby("timestamp_rounded")

    # Create empty metrics dataframe
    metrics_df = pd.DataFrame()

    # Track hot-or-not results over time
    hot_or_not_results_list = []

    # Store previous timestamp for interval detection
    prev_timestamp = None
    one_day_mark = df_copy["timestamp"].min() + timedelta(days=1)

    print(f"One day mark: {one_day_mark}")

    # Process each time group
    for timestamp, batch_df in tqdm(time_groups, desc="Processing Pandas Data"):
        # Skip empty batches
        if batch_df.empty:
            continue

        # Accumulate data - use all data up to this timestamp
        current_df = df_copy[df_copy["timestamp"] <= timestamp].copy()

        # Skip if no data to process
        if current_df.empty:
            print(f"Warning: No data to process for timestamp {timestamp}")
            continue

        # Calculate metrics on all data so far
        current_metrics = calculate_metrics(current_df)
        metrics_df = current_metrics

        # Display summary of current metrics
        # print(f"Metrics calculated, total rows: {len(current_metrics)}")

        # Check if metrics dataframe is empty
        if current_metrics.empty:
            print(
                f"Warning: Metrics calculation produced empty dataframe for timestamp {timestamp}"
            )
            print(
                f"Original data had {len(current_df)} rows with {current_df['video_id'].nunique()} unique videos"
            )
            continue

        # Show sample metrics for one video (with safety check)
        if len(current_metrics) > 0 and "video_id" in current_metrics.columns:
            sample_video = current_metrics["video_id"].iloc[0]
            # print(f"Sample metrics for video {sample_video}:")
            # display(
            #     current_metrics[current_metrics["video_id"] == sample_video].tail(2)
            # )
        else:
            print(f"Warning: No metrics data available to display sample")

        # Check if we're after one day and should calculate hot-or-not
        if timestamp >= one_day_mark:
            # Check if it's been at least 5 minutes since the last calculation
            if (
                prev_timestamp is None
                or (timestamp - prev_timestamp).total_seconds() >= 300
            ):
                # print(f"\nCalculating pandas hot-or-not at {timestamp}...")

                # Calculate hot-or-not using current metrics
                hot_or_not_result = determine_hot_or_not(
                    current_metrics, current_window=5, reference_window=1440
                )

                # Add timestamp to results for tracking
                hot_or_not_result["timestamp"] = timestamp

                # Store results
                hot_or_not_results_list.append(hot_or_not_result)

                # Display current results
                # print("Current Pandas-based Hot-or-Not Results:")
                # display(hot_or_not_result)

                # Update previous timestamp
                prev_timestamp = timestamp

    # Combine all hot-or-not results
    if hot_or_not_results_list:
        combined_hot_or_not = pd.concat(hot_or_not_results_list)
        # Keep only the latest result for each video
        latest_hot_or_not = (
            combined_hot_or_not.sort_values("timestamp")
            .groupby("video_id")
            .last()
            .reset_index()
        )
        latest_hot_or_not = latest_hot_or_not.drop("timestamp", axis=1)
    else:
        latest_hot_or_not = pd.DataFrame()

    return metrics_df, latest_hot_or_not


# %% [markdown]
## 5. Run Tests with Incremental Processing


# %%
def run_assertion_tests(pandas_results, sql_results):
    """Run assertion tests comparing pandas and SQL results"""
    # Merge results
    comparison = pandas_results.merge(
        sql_results, on="video_id", suffixes=("_pandas", "_sql")
    )

    # Test cases
    tests = []
    status_tests = []
    score_tests = []

    # Test 1: All videos are present in both results
    test_videos = set(v.video_id for v in TEST_PATTERNS)
    tests.append(
        {
            "name": "All videos present",
            "passed": test_videos
            == set(pandas_results["video_id"])
            == set(sql_results["video_id"]),
            "message": "Some videos are missing from results",
        }
    )

    # Test 2: Hot-or-not status matches
    for _, row in comparison.iterrows():
        status_test = {
            "name": f"Hot-or-not status for {row['video_id']}",
            "passed": row["hot_or_not_pandas"] == row["hot_or_not_sql"],
            "message": f"Status mismatch for {row['video_id']}: Pandas={row['hot_or_not_pandas']}, SQL={row['hot_or_not_sql']}",
        }
        status_tests.append(status_test)
        tests.append(status_test)

    # Test 3: DS scores are close (within 1% relative difference)
    for _, row in comparison.iterrows():
        pandas_score = row["current_avg_ds_score_pandas"]
        sql_score = row["current_avg_ds_score_sql"]
        if pandas_score is not None and sql_score is not None:
            # Convert Decimal to float if needed
            pandas_score = float(pandas_score)
            sql_score = float(sql_score)

            relative_diff = abs(pandas_score - sql_score) / max(
                abs(pandas_score), abs(sql_score), 1e-10  # Avoid division by zero
            )
            score_test = {
                "name": f"DS score for {row['video_id']}",
                "passed": relative_diff < 0.01,
                "message": f"Score mismatch for {row['video_id']}: Pandas={pandas_score:.5f}, SQL={sql_score:.5f}, Diff={relative_diff*100:.2f}%",
                "diff_percentage": relative_diff * 100,
            }
            score_tests.append(score_test)
            tests.append(score_test)

    # Print test results
    print("\nAssertion Test Results:")
    for test in tests:
        status = "✅" if test["passed"] else "❌"
        print(f"{status} {test['name']}")
        if not test["passed"]:
            print(f"   {test['message']}")

    # Calculate and print error percentages
    status_failures = sum(1 for test in status_tests if not test["passed"])
    status_error_percentage = (
        (status_failures / len(status_tests)) * 100 if status_tests else 0
    )

    score_failures = sum(1 for test in score_tests if not test["passed"])
    score_error_percentage = (
        (score_failures / len(score_tests)) * 100 if score_tests else 0
    )

    avg_score_diff = (
        sum(test.get("diff_percentage", 0) for test in score_tests) / len(score_tests)
        if score_tests
        else 0
    )

    print(f"\nError Summary:")
    print(
        f"Status Mismatches: {status_failures}/{len(status_tests)} ({status_error_percentage:.2f}%)"
    )
    print(
        f"Score Mismatches: {score_failures}/{len(score_tests)} ({score_error_percentage:.2f}%)"
    )
    print(f"Average Score Difference: {avg_score_diff:.2f}%")
    print(f"Support: {len(comparison)} videos tested")

    # Return overall success
    return all(test["passed"] for test in tests)


# %% [markdown]
## 6. Main Test Runner


# %%
def clean_tables():
    """Truncate all test-related tables to start with clean state"""
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                DELETE FROM hot_or_not_evaluator.video_engagement_relation
                WHERE video_id LIKE 'sgx-%';
            """
            )
            cur.execute(
                """
                DELETE FROM hot_or_not_evaluator.video_hot_or_not_status
                WHERE video_id LIKE 'sgx-%';
                """
            )
            conn.commit()
    print("Test tables cleaned up successfully.")


# %%
# Clean up existing data
print("Cleaning up existing test data...")
clean_tables()

# Generate test data spanning 2 days for each pattern
print("Generating test data...")
end_time = datetime.now()
test_data = generate_deterministic_data(
    TEST_PATTERNS,
    end_time=end_time,
    duration_days=1.05,  # Generate 2 days of data
    events_per_minute=10,
    minute_interval=4,  # Generate data every 4 minutes to keep dataset manageable
)

# Verify test data has content
if test_data.empty:
    print("Error: Generated test data is empty. Cannot proceed.")
    raise ValueError("Generated test data is empty")

# Print data statistics
print(
    f"Generated data from {test_data['timestamp'].min()} to {test_data['timestamp'].max()}"
)
print(f"Total rows: {len(test_data)}")
print(f"Unique timestamps: {test_data['timestamp'].nunique()}")
print(f"Unique videos: {test_data['video_id'].nunique()}")

# Display sample of generated data
print("\nSample of generated data:")
display(test_data.groupby("video_id").head(2))

# %%
# Run pandas incremental processing first since SQL part is commented out
print("\n2. Processing data incrementally in pandas...")
metrics_df, pandas_hot_or_not = process_pandas_incremental(test_data)

# Verify pandas processing was successful
if metrics_df.empty:
    print("Warning: Metrics calculation produced empty results")
if pandas_hot_or_not.empty:
    print("Warning: No hot-or-not results were generated by pandas processing")

# Display sample of calculated metrics if available
if not metrics_df.empty:
    print("\nSample of calculated metrics:")
    display(metrics_df.groupby("video_id").tail(1))

# %%
# Get SQL results
print("\n3. Getting final SQL results...")

# Uncomment these lines if you want to run SQL processing
# print("\n1. Populating SQL database with incremental hot-or-not calculation...")
populate_sql_database_incremental(test_data, batch_size=100)

# %%
# Get final SQL results if SQL processing was run
sql_hot_or_not = get_sql_results()

# Print final results
print("\nFinal Pandas-based Hot-or-Not Results:")
display(pandas_hot_or_not)

print("\nFinal SQL-based Hot-or-Not Results:")
display(sql_hot_or_not)

# %%
# Run assertion tests if both pandas and SQL results are available
if not pandas_hot_or_not.empty and not sql_hot_or_not.empty:
    print("\n4. Running assertion tests...")
    success = run_assertion_tests(pandas_hot_or_not, sql_hot_or_not)
    print(f"\nOverall test {'passed' if success else 'failed'}")
else:
    print("Skipping assertion tests due to missing results")
    success = False
