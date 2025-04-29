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
    VideoPattern("sgx-steady_low", steady_low, "Steady Low Engagement"),
    VideoPattern("sgx-rising", rising, "Rising Engagement"),
    VideoPattern("sgx-falling", falling, "Falling Engagement"),
    VideoPattern("sgx-spike", spike_middle, "Spike in Middle"),
    VideoPattern("sgx-dip", dip_middle, "Dip in Middle"),
]

# %% [markdown]
## 2. Generate Deterministic Engagement Data
# We'll generate one hour of data with one-minute intervals for each pattern.


# %%
def generate_deterministic_data(
    patterns, end_time, duration_days=1, events_per_minute=10, minute_interval=1
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

    # Calculate start time (e.g., 1 day ago)
    start_time = end_time - timedelta(days=duration_days)

    # Calculate total minutes to generate
    total_minutes = int(duration_days * 24 * 60)

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

                all_data.append(
                    {
                        "video_id": pattern.video_id,
                        "timestamp": timestamp,
                        "liked": liked,
                        "watch_percentage": engagement["watch_percentage"],
                    }
                )

    # Create DataFrame and sort by timestamp to simulate real-time data flow
    df = pd.DataFrame(all_data)
    df = df.sort_values("timestamp")

    return df


# %% [markdown]
## 3. Implement Pandas-based Hot-or-Not Calculation


# %%
def calculate_metrics(df):
    """Calculate all metrics for each video and minute"""
    # Group by video_id and timestamp
    grouped = (
        df.groupby(["video_id", "timestamp"])
        .agg({"liked": ["count", "sum"], "watch_percentage": ["mean"]})
        .reset_index()
    )

    # Flatten column names
    grouped.columns = [
        "video_id",
        "timestamp",
        "watch_count_mnt",
        "like_count_mnt",
        "average_percentage_watched_mnt",
    ]

    # Calculate like_ctr_mnt
    grouped["like_ctr_mnt"] = grouped.apply(
        lambda row: (
            (row["like_count_mnt"] * 100.0) / row["watch_count_mnt"]
            if row["watch_count_mnt"] > 0
            else 0
        ),
        axis=1,
    )

    # Sort by video_id and timestamp for cumulative calculations
    grouped = grouped.sort_values(["video_id", "timestamp"])

    # Calculate cumulative metrics
    grouped["cumulative_like_count"] = grouped.groupby("video_id")[
        "like_count_mnt"
    ].cumsum()
    grouped["cumulative_watch_count"] = grouped.groupby("video_id")[
        "watch_count_mnt"
    ].cumsum()

    # Calculate cumulative averages with proper weight handling
    # First, safely calculate cumulative like CTR
    grouped["cumulative_like_ctr"] = grouped.apply(
        lambda row: (
            (row["cumulative_like_count"] * 100.0) / row["cumulative_watch_count"]
            if row["cumulative_watch_count"] > 0
            else 0
        ),
        axis=1,
    )

    # Calculate cumulative average percentage watched using expanding window
    # This is the pandas equivalent of the weighted average calculation in SQL
    grouped["cumulative_average_percentage_watched"] = (
        grouped.groupby("video_id")["average_percentage_watched_mnt"]
        .expanding()
        .mean()
        .values
    )

    # Normalize cumulative metrics with proper zero handling
    grouped["normalized_cumulative_like_ctr"] = grouped.apply(
        lambda row: (
            max(0, (row["cumulative_like_ctr"] - LIKE_CTR_CENTER) / LIKE_CTR_RANGE)
            if LIKE_CTR_RANGE != 0
            else 0
        ),
        axis=1,
    )

    grouped["normalized_cumulative_watch_percentage"] = grouped.apply(
        lambda row: (
            max(
                0,
                (row["cumulative_average_percentage_watched"] - WATCH_PERCENTAGE_CENTER)
                / WATCH_PERCENTAGE_RANGE,
            )
            if WATCH_PERCENTAGE_RANGE != 0
            else 0
        ),
        axis=1,
    )

    # Calculate harmonic mean and ds_score with proper denominator check
    grouped["harmonic_mean"] = grouped.apply(
        lambda row: (
            (
                (row["normalized_cumulative_like_ctr"] + 1)
                * (row["normalized_cumulative_watch_percentage"] + 1)
            )
            / (
                row["normalized_cumulative_like_ctr"]
                + row["normalized_cumulative_watch_percentage"]
                + 2
            )
            if (
                row["normalized_cumulative_like_ctr"]
                + row["normalized_cumulative_watch_percentage"]
                + 2
            )
            != 0
            else 1
        ),
        axis=1,
    )

    grouped["ds_score"] = grouped["harmonic_mean"] - 1

    return grouped


def determine_hot_or_not(metrics_df, current_window=5, reference_window=60):
    """Determine hot-or-not status for each video"""
    results = []

    # Keep track of previous hot status for each video
    previous_statuses = {}

    # Sort the dataframe by timestamp to ensure we have the correct order
    metrics_df = metrics_df.sort_values(["video_id", "timestamp"])

    for video_id in metrics_df["video_id"].unique():
        video_data = metrics_df[metrics_df["video_id"] == video_id].copy()

        # Initialize values to NULL
        current_avg_ds = None
        predicted_ds = None
        is_hot = None

        try:
            # Get the latest timestamp - avoid using partially aggregated data
            latest_time = video_data["timestamp"].max()

            # Create time window boundaries for filtering
            current_window_start = latest_time - timedelta(minutes=current_window)
            reference_window_start = latest_time - timedelta(minutes=reference_window)
            current_window_end = latest_time

            # Calculate current window average (last 5 minutes)
            # Use query method instead of boolean indexing to avoid type issues
            current_window_mask = (video_data["timestamp"] >= current_window_start) & (
                video_data["timestamp"] < current_window_end
            )
            current_window_data = video_data[current_window_mask]

            # Only calculate if we have data
            if not current_window_data.empty:
                current_avg_ds = current_window_data["ds_score"].mean()

            # Calculate reference window (previous period excluding current window)
            reference_window_mask = (video_data["timestamp"] < current_window_start) & (
                video_data["timestamp"] >= reference_window_start
            )
            reference_window_data = video_data[
                reference_window_mask
            ].copy()  # Create a proper copy to prevent SettingWithCopyWarning

            # Only perform regression if we have at least 2 points (required for linear regression)
            if len(reference_window_data) >= 2:
                # Convert timestamp to epoch for regression - safe now with a proper copy
                reference_window_data["epoch"] = reference_window_data[
                    "timestamp"
                ].apply(lambda x: int(x.timestamp()))

                # Perform OLS regression
                X = reference_window_data["epoch"].values.reshape(-1, 1)
                y = reference_window_data["ds_score"].values

                from sklearn.linear_model import LinearRegression

                model = LinearRegression()
                model.fit(X, y)

                # Get slope and intercept
                ref_slope = model.coef_[0]
                ref_intercept = model.intercept_

                # Make sure regression was successful
                if ref_slope is not None and ref_intercept is not None:
                    # Get the midpoint of the current window for prediction
                    # This mimics the SQL's approach using (v_now + (v_now - v_5_mins_ago)/2)
                    current_midpoint = latest_time - timedelta(
                        minutes=current_window / 2
                    )
                    current_epoch = int(current_midpoint.timestamp())

                    # Calculate the predicted value at the midpoint
                    predicted_ds = ref_slope * current_epoch + ref_intercept

            # Determine hot-or-not status
            if current_avg_ds is not None and predicted_ds is not None:
                is_hot = current_avg_ds > predicted_ds
            else:
                # Use previous status if we can't make a comparison
                is_hot = previous_statuses.get(video_id, None)

            # Save status for next time
            previous_statuses[video_id] = is_hot

            # Add result to list
            results.append(
                {
                    "video_id": video_id,
                    "current_avg_ds_score": current_avg_ds,
                    "reference_predicted_avg_ds_score": predicted_ds,
                    "hot_or_not": is_hot,
                }
            )

        except Exception as e:
            # Handle any errors and continue processing other videos
            print(f"Error processing video {video_id}: {e}")
            results.append(
                {
                    "video_id": video_id,
                    "current_avg_ds_score": None,
                    "reference_predicted_avg_ds_score": None,
                    "hot_or_not": previous_statuses.get(video_id, None),
                }
            )

    return pd.DataFrame(results)


# %% [markdown]
## 4. Generate Test Data and Calculate Results

# %%
# Generate test data spanning 1 day plus 5 minutes for each pattern
end_time = datetime.now()
test_data = generate_deterministic_data(
    TEST_PATTERNS,
    end_time=end_time,
    duration_days=2,
    events_per_minute=10,
    minute_interval=30,  # Generate data every 5 minutes to keep dataset manageable
)

# Calculate metrics
metrics = calculate_metrics(test_data)

# Print data statistics
print(
    f"Generated data from {test_data['timestamp'].min()} to {test_data['timestamp'].max()}"
)
print(f"Total rows: {len(test_data)}")
print(f"Unique timestamps: {test_data['timestamp'].nunique()}")
print(f"Unique videos: {test_data['video_id'].nunique()}")

# Calculate hot-or-not results
hot_or_not_results = determine_hot_or_not(
    metrics, current_window=5, reference_window=1440
)  # 1440 minutes = 1 day
print("Pandas-based Hot-or-Not Results:")
print(hot_or_not_results)

# %% [markdown]
## 5. Populate SQL Database and Compare Results


# %%


def populate_sql_database(df):
    """Populate the SQL database with test data in batches"""
    batch_size = 1000
    total_rows = len(df)

    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            with tqdm(total=total_rows, desc="Populating SQL Database") as pbar:
                for start_idx in range(0, total_rows, batch_size):
                    batch = df.iloc[start_idx : min(start_idx + batch_size, total_rows)]

                    # Create batch parameters and SQL
                    values_list = []
                    args = []

                    for _, row in batch.iterrows():
                        values_list.append(
                            "(%s::VARCHAR(255), %s::BOOLEAN, %s::NUMERIC(5,2))"
                        )
                        args.extend(
                            [
                                str(row["video_id"]),
                                bool(row["liked"]),
                                round(float(row["watch_percentage"]), 2),
                            ]
                        )

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

                    # Execute batch query
                    cur.execute(query, args)
                    pbar.update(len(batch))

            conn.commit()


def get_sql_results():
    """Get hot-or-not results from SQL database"""
    with psycopg.connect(conn_string) as conn:
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


# Populate SQL database
print("Populating SQL database...")
populate_sql_database(test_data)
# %%
test_data.dtypes
# %%

# Trigger hot-or-not computation
with psycopg.connect(conn_string) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT hot_or_not_evaluator.compute_hot_or_not()")
    conn.commit()

# Get SQL results
sql_results = get_sql_results()
print("\nSQL-based Hot-or-Not Results:")
print(sql_results)

# %% [markdown]
## 6. Assertion Tests


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
                abs(pandas_score), abs(sql_score)
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


# Run assertion tests
success = run_assertion_tests(hot_or_not_results, sql_results)
print(f"\nOverall test {'passed' if success else 'failed'}")
# %%
hot_or_not_results
