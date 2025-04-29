# %%
from tqdm import tqdm
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg
from dotenv import load_dotenv
import os
import json
import time
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns

# Load environment variables
load_dotenv("/Users/sagar/work/yral/hot-or-not-game-evaluator/test/.env")

# Database connection parameters
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} sslmode=require"

# Constants for normalization (from metric_const table)
LIKE_CTR_CENTER = 0
LIKE_CTR_RANGE = 0.2
WATCH_PERCENTAGE_CENTER = 0
WATCH_PERCENTAGE_RANGE = 0.7


# %%
def get_latest_timestamp(video_id="sgx-test_video_simple", conn_string=conn_string):
    """
    Get the latest timestamp in the video_engagement_relation table for a specific video.
    This will be our starting point for new data.

    Args:
        video_id (str): The ID of the video
        conn_string (str): PostgreSQL connection string

    Returns:
        datetime: The latest timestamp for the video
    """
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT MAX(timestamp_mnt)
                FROM hot_or_not_evaluator.video_engagement_relation
                WHERE video_id = %s
                """,
                (video_id,),
            )
            latest_timestamp = cur.fetchone()[0]

    if latest_timestamp is None:
        # If no data exists, start from provided default time or current time
        latest_timestamp = datetime(2025, 4, 29, 14, 0, 0)

    # If timestamp from DB is timezone-aware (has tzinfo), use it as is
    # Otherwise, create a timezone-naive timestamp
    if hasattr(latest_timestamp, "tzinfo") and latest_timestamp.tzinfo is not None:
        # Make sure we're returning the timestamp with its timezone info preserved
        return latest_timestamp
    else:
        # Return a timezone-naive timestamp with the same value
        return (
            latest_timestamp.replace(tzinfo=None)
            if hasattr(latest_timestamp, "replace")
            else latest_timestamp
        )


# %%
def generate_activity_data(
    video_id="sgx-test_video_simple",
    start_timestamp=None,
    duration_hours=24,
    interval_minutes=4,
    events_per_interval=10,
    growth_rate=1.05,  # 5% growth per interval
):
    """
    Generate steadily increasing activity data for a video.

    Args:
        video_id (str): The ID of the video
        start_timestamp (datetime): Starting timestamp, if None gets latest from DB
        duration_hours (int): Duration of data to generate in hours
        interval_minutes (int): Interval between data points in minutes
        events_per_interval (int): Base number of events per interval
        growth_rate (float): Growth rate of events per interval

    Returns:
        pandas.DataFrame: DataFrame with activity data
    """
    if start_timestamp is None:
        start_timestamp = get_latest_timestamp(video_id)

    print(f"Generating activity data from: {start_timestamp}")

    # Check if start_timestamp is timezone-aware
    is_tz_aware = (
        hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
    )

    # Create time intervals
    end_timestamp = start_timestamp + timedelta(hours=duration_hours)

    # Create date_range with appropriate timezone awareness
    if is_tz_aware:
        # If timezone-aware, create timezone-aware intervals
        intervals = pd.date_range(
            start=start_timestamp + timedelta(minutes=interval_minutes),
            end=end_timestamp,
            freq=f"{interval_minutes}min",
            tz=start_timestamp.tzinfo,
        )
    else:
        # If timezone-naive, create timezone-naive intervals
        intervals = pd.date_range(
            start=start_timestamp + timedelta(minutes=interval_minutes),
            end=end_timestamp,
            freq=f"{interval_minutes}min",
        )

    data = []
    current_events = events_per_interval

    # Generate data with steadily increasing activity
    for i, timestamp in enumerate(intervals):
        # Calculate the number of events for this interval (growing over time)
        num_events = int(current_events)
        current_events *= growth_rate

        # For each event, generate watch and like data
        for event in range(num_events):
            # Increasing likelihood of likes over time to show improvement
            base_like_probability = 0.1 + (i / len(intervals)) * 0.2
            like_probability = min(0.5, base_like_probability)

            # Generate better watch percentages over time
            base_watch_pct = 40 + (i / len(intervals)) * 30
            watch_pct = min(95, base_watch_pct + np.random.normal(0, 10))

            # Add some randomness to watch percentage
            watch_pct = max(10, min(100, watch_pct))

            # Determine if this event resulted in a like
            liked = np.random.random() < like_probability

            # Add event to data
            data.append(
                {
                    "video_id": video_id,
                    "timestamp_mnt": timestamp,
                    "liked": liked,
                    "watch_percentage": watch_pct,
                }
            )

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df


# %%
def process_with_pandas(events_df, conn_string=conn_string):
    """
    Process the events data using pandas to mimic the stored procedures.
    This simulates what happens in update_counter and compute_hot_or_not.

    Args:
        events_df (pandas.DataFrame): DataFrame with events

    Returns:
        tuple: (minute_metrics_df, hot_or_not_status)
    """
    # First, query historical data from the database
    video_id = events_df["video_id"].iloc[0]
    start_timestamp = events_df["timestamp_mnt"].min() - timedelta(days=1)
    end_timestamp = events_df["timestamp_mnt"].max()

    print(f"Querying historical data from {start_timestamp} to {end_timestamp}")

    # Get historical data from the database
    historical_data = pd.DataFrame()
    try:
        # Connect to database
        from sqlalchemy import create_engine

        # Convert psycopg conn_string to SQLAlchemy format
        db_params = {
            param.split("=")[0]: param.split("=")[1]
            for param in conn_string.split()
            if "=" in param
        }
        sqlalchemy_uri = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
        engine = create_engine(sqlalchemy_uri)

        # Query existing data
        query = """
        SELECT video_id, timestamp_mnt, like_count_mnt, watch_count_mnt,
               average_percentage_watched_mnt, cumulative_like_count,
               cumulative_watch_count, cumulative_like_ctr,
               cumulative_average_percentage_watched, ds_score
        FROM hot_or_not_evaluator.video_engagement_relation
        WHERE video_id = %s AND timestamp_mnt < %s
        ORDER BY timestamp_mnt
        """
        historical_data = pd.read_sql_query(
            query, engine, params=(video_id, events_df["timestamp_mnt"].min())
        )
        print(f"Retrieved {len(historical_data)} historical records")

        # Check timezone awareness of historical data
        if len(historical_data) > 0:
            historical_tz_aware = (
                historical_data["timestamp_mnt"].iloc[0].tzinfo is not None
            )
            print(
                f"Historical data timestamps are {'timezone-aware' if historical_tz_aware else 'timezone-naive'}"
            )

            # Check timezone awareness of new events
            events_tz_aware = events_df["timestamp_mnt"].iloc[0].tzinfo is not None
            print(
                f"New events timestamps are {'timezone-aware' if events_tz_aware else 'timezone-naive'}"
            )

            # Make timestamps consistent
            if historical_tz_aware != events_tz_aware:
                print("Converting timestamps to consistent timezone awareness")
                if historical_tz_aware and not events_tz_aware:
                    # Convert events to timezone-aware
                    import pytz

                    events_df = events_df.copy()
                    events_df["timestamp_mnt"] = events_df["timestamp_mnt"].apply(
                        lambda x: x.replace(tzinfo=pytz.UTC)
                    )
                elif not historical_tz_aware and events_tz_aware:
                    # Convert historical to timezone-aware
                    import pytz

                    historical_data = historical_data.copy()
                    historical_data["timestamp_mnt"] = historical_data[
                        "timestamp_mnt"
                    ].apply(lambda x: x.replace(tzinfo=pytz.UTC))

    except Exception as e:
        print(f"Error retrieving historical data: {e}")

    # 1. Aggregate data by minute for new events
    minute_agg = (
        events_df.groupby(["video_id", "timestamp_mnt"])
        .agg(
            like_count_mnt=("liked", lambda x: sum(x)),
            watch_count_mnt=("liked", "count"),
            average_percentage_watched_mnt=("watch_percentage", "mean"),
        )
        .reset_index()
    )

    # 2. Calculate minute metrics
    minute_agg["like_ctr_mnt"] = minute_agg.apply(
        lambda row: (
            (row["like_count_mnt"] * 100 / row["watch_count_mnt"])
            if row["watch_count_mnt"] > 0
            else 0
        ),
        axis=1,
    )

    # 3. Create a copy for cumulative calculations to avoid SettingWithCopyWarning
    minute_metrics = minute_agg.copy()

    # Sort by timestamp to ensure proper cumulative calculations
    minute_metrics = minute_metrics.sort_values("timestamp_mnt")

    # 4. Calculate cumulative metrics
    video_id = minute_metrics["video_id"].iloc[
        0
    ]  # Assuming all rows have the same video_id

    # Get previous cumulative values from historical data
    previous_cumulative = {}
    if not historical_data.empty:
        last_row = historical_data.iloc[-1]
        previous_cumulative = {
            "cumulative_like_count": last_row["cumulative_like_count"],
            "cumulative_watch_count": last_row["cumulative_watch_count"],
            "cumulative_watched_sum": last_row["cumulative_average_percentage_watched"]
            * last_row["cumulative_watch_count"],
        }
    else:
        # If no historical data, get from database function
        previous_cumulative = get_previous_cumulative_metrics(video_id)

    # Initialize cumulative counters with previous values or zeros
    cumulative_like_count = previous_cumulative.get("cumulative_like_count", 0)
    cumulative_watch_count = previous_cumulative.get("cumulative_watch_count", 0)
    cumulative_watched_sum = previous_cumulative.get("cumulative_watched_sum", 0)

    # Arrays to store calculated metrics
    cumulative_like_counts = []
    cumulative_watch_counts = []
    cumulative_like_ctrs = []
    cumulative_avg_watch_pcts = []
    norm_like_ctrs = []
    norm_watch_pcts = []
    harmonic_means = []
    ds_scores = []

    # Calculate cumulative metrics for each minute
    for idx, row in minute_metrics.iterrows():
        # Update cumulative counters
        cumulative_like_count += row["like_count_mnt"]
        cumulative_watch_count += row["watch_count_mnt"]
        cumulative_watched_sum += (
            row["average_percentage_watched_mnt"] * row["watch_count_mnt"]
        )

        # Calculate derived metrics
        cumulative_avg_watch_pct = (
            cumulative_watched_sum / cumulative_watch_count
            if cumulative_watch_count > 0
            else 0
        )
        cumulative_like_ctr = (
            (cumulative_like_count * 100 / cumulative_watch_count)
            if cumulative_watch_count > 0
            else 0
        )

        # Normalize metrics
        norm_like_ctr = max(0, (cumulative_like_ctr - LIKE_CTR_CENTER) / LIKE_CTR_RANGE)
        norm_watch_pct = max(
            0,
            (cumulative_avg_watch_pct - WATCH_PERCENTAGE_CENTER)
            / WATCH_PERCENTAGE_RANGE,
        )

        # Calculate harmonic mean and ds_score
        harmonic_mean = ((norm_like_ctr + 1) * (norm_watch_pct + 1)) / (
            norm_like_ctr + norm_watch_pct + 2
        )
        ds_score = harmonic_mean - 1

        # Store metrics for this minute
        cumulative_like_counts.append(cumulative_like_count)
        cumulative_watch_counts.append(cumulative_watch_count)
        cumulative_like_ctrs.append(cumulative_like_ctr)
        cumulative_avg_watch_pcts.append(cumulative_avg_watch_pct)
        norm_like_ctrs.append(norm_like_ctr)
        norm_watch_pcts.append(norm_watch_pct)
        harmonic_means.append(harmonic_mean)
        ds_scores.append(ds_score)

    # Assign computed values to the dataframe
    minute_metrics["cumulative_like_count"] = cumulative_like_counts
    minute_metrics["cumulative_watch_count"] = cumulative_watch_counts
    minute_metrics["cumulative_like_ctr"] = cumulative_like_ctrs
    minute_metrics["cumulative_average_percentage_watched"] = cumulative_avg_watch_pcts
    minute_metrics["normalized_cumulative_like_ctr"] = norm_like_ctrs
    minute_metrics["normalized_cumulative_watch_percentage"] = norm_watch_pcts
    minute_metrics["harmonic_mean_of_like_count_and_watch_count"] = harmonic_means
    minute_metrics["ds_score"] = ds_scores

    # 5. Combine historical data with new minute metrics for hot_or_not computation
    combined_metrics = pd.DataFrame()

    # Check if historical data has any rows
    if not historical_data.empty:
        # Ensure consistent timezone awareness before combining
        if len(historical_data) > 0 and len(minute_metrics) > 0:
            historical_tz_aware = (
                historical_data["timestamp_mnt"].iloc[0].tzinfo is not None
            )
            minute_metrics_tz_aware = (
                minute_metrics["timestamp_mnt"].iloc[0].tzinfo is not None
            )

            # Convert timestamps to ensure consistency
            if historical_tz_aware != minute_metrics_tz_aware:
                print(
                    f"Making timestamps consistent for combination: historical={historical_tz_aware}, metrics={minute_metrics_tz_aware}"
                )
                if historical_tz_aware:
                    # Convert minute_metrics to match historical
                    import pytz

                    minute_metrics = minute_metrics.copy()
                    minute_metrics["timestamp_mnt"] = minute_metrics[
                        "timestamp_mnt"
                    ].apply(
                        lambda x: x.replace(tzinfo=pytz.UTC) if x.tzinfo is None else x
                    )
                else:
                    # Convert historical to match minute_metrics
                    historical_data = historical_data.copy()
                    historical_data["timestamp_mnt"] = historical_data[
                        "timestamp_mnt"
                    ].apply(
                        lambda x: x.replace(tzinfo=None) if x.tzinfo is not None else x
                    )

        # Select only needed columns from historical data
        historical_subset = historical_data[["video_id", "timestamp_mnt", "ds_score"]]
        # Combine with new metrics
        combined_metrics = pd.concat(
            [
                historical_subset,
                minute_metrics[["video_id", "timestamp_mnt", "ds_score"]],
            ]
        )
    else:
        combined_metrics = minute_metrics[["video_id", "timestamp_mnt", "ds_score"]]

    # Ensure timestamps are sorted - this is where the error was occurring
    combined_metrics = combined_metrics.sort_values("timestamp_mnt")

    # 6. Simulate compute_hot_or_not procedure with combined data
    hot_or_not_status = compute_hot_or_not_pandas(combined_metrics, video_id)

    return minute_metrics, hot_or_not_status


# %%
def get_previous_cumulative_metrics(video_id, conn_string=conn_string):
    """
    Get the previous cumulative metrics for a video from the database.

    Args:
        video_id (str): The ID of the video
        conn_string (str): PostgreSQL connection string

    Returns:
        dict: Dictionary with previous cumulative metrics
    """
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        cumulative_like_count,
                        cumulative_watch_count,
                        cumulative_average_percentage_watched * cumulative_watch_count AS cumulative_watched_sum
                    FROM hot_or_not_evaluator.video_engagement_relation
                    WHERE video_id = %s
                    ORDER BY timestamp_mnt DESC
                    LIMIT 1
                    """,
                    (video_id,),
                )
                row = cur.fetchone()

                if row is not None:
                    return {
                        "cumulative_like_count": row[0],
                        "cumulative_watch_count": row[1],
                        "cumulative_watched_sum": row[2],
                    }
    except Exception as e:
        print(f"Error getting previous cumulative metrics: {e}")

    # Default empty values if no previous data or error
    return {
        "cumulative_like_count": 0,
        "cumulative_watch_count": 0,
        "cumulative_watched_sum": 0,
    }


# %%
def compute_hot_or_not_pandas(minute_metrics, video_id):
    """
    Compute the hot-or-not status for a video using pandas.
    This mimics the compute_hot_or_not stored procedure.

    Args:
        minute_metrics (pandas.DataFrame): DataFrame with minute metrics
        video_id (str): The ID of the video

    Returns:
        dict: Hot or not status information
    """
    # Ensure timestamps in minute_metrics are timezone-aware if they aren't already
    if minute_metrics.empty:
        return {
            "video_id": video_id,
            "last_updated_mnt": datetime.now(),
            "hot_or_not": None,
            "current_avg_ds_score": None,
            "reference_predicted_avg_ds_score": None,
        }

    # Check if timestamps are timezone-aware
    has_tzinfo = minute_metrics["timestamp_mnt"].iloc[0].tzinfo is not None

    # Create timezone-aware now, five_mins_ago, and one_day_ago that match the data's timezone awareness
    if has_tzinfo:
        # Timestamps in data are timezone-aware, so use timezone-aware now
        import pytz

        now = datetime.now(pytz.UTC)
        five_mins_ago = now - timedelta(minutes=5)
        one_day_ago = now - timedelta(days=1)
    else:
        # Timestamps in data are timezone-naive, so use timezone-naive now
        now = datetime.now()
        five_mins_ago = now - timedelta(minutes=5)
        one_day_ago = now - timedelta(days=1)

    # Get previous hot status (if available)
    previous_hot_status = get_previous_hot_status(video_id)

    # Calculate current average ds_score (last 5 minutes)
    current_window = minute_metrics[
        (minute_metrics["timestamp_mnt"] >= five_mins_ago)
        & (minute_metrics["timestamp_mnt"] < now)
    ]

    if len(current_window) > 0:
        current_avg_ds = current_window["ds_score"].mean()
        print(
            f"Current window has {len(current_window)} records, avg DS score: {current_avg_ds}"
        )
    else:
        current_avg_ds = None
        print("No records in current window")

    # Calculate reference period metrics (1 day ago to 5 mins ago)
    reference_period = minute_metrics[
        (minute_metrics["timestamp_mnt"] >= one_day_ago)
        & (minute_metrics["timestamp_mnt"] < five_mins_ago)
    ]

    print(f"Reference period has {len(reference_period)} records")

    # Following the SQL stored procedure approach for regression
    ref_predicted_avg_ds = None

    # Perform linear regression if we have enough data points
    if len(reference_period) >= 2:
        try:
            # Convert timestamps to numeric (seconds since epoch) for regression
            reference_period = reference_period.copy()
            reference_period["timestamp_seconds"] = reference_period[
                "timestamp_mnt"
            ].apply(lambda x: x.timestamp())

            # Use the SQL regr_slope and regr_intercept approach
            X = reference_period["timestamp_seconds"].values
            y = reference_period["ds_score"].values

            # Calculate regression parameters similar to SQL's regr_slope and regr_intercept
            n = len(X)
            if n > 0:
                mean_x = X.mean()
                mean_y = y.mean()

                # Calculate covariance and variance (denominator)
                covariance = sum((X - mean_x) * (y - mean_y))
                variance = sum((X - mean_x) ** 2)

                if variance != 0:
                    # Regression slope
                    slope = covariance / variance
                    # Regression intercept
                    intercept = mean_y - slope * mean_x

                    # Calculate the timestamp for the midpoint of current window (like in SQL procedure)
                    current_midpoint = (now.timestamp() + five_mins_ago.timestamp()) / 2

                    # Calculate predicted value at midpoint of current window
                    ref_predicted_avg_ds = slope * current_midpoint + intercept

                    print(f"OLS regression: slope={slope}, intercept={intercept}")
                    print(
                        f"Midpoint timestamp: {current_midpoint}, predicted DS score: {ref_predicted_avg_ds}"
                    )
                else:
                    print(
                        "Zero variance in timestamp values, cannot perform regression"
                    )
            else:
                print("No reference data points available")
        except Exception as e:
            print(f"Error in regression calculation: {e}")

    # Determine if video is hot
    if current_avg_ds is not None and ref_predicted_avg_ds is not None:
        is_hot = current_avg_ds > ref_predicted_avg_ds
        print(
            f"Video is {'hot' if is_hot else 'not hot'}: {current_avg_ds} {'>' if is_hot else '<='} {ref_predicted_avg_ds}"
        )
    else:
        is_hot = previous_hot_status
        print(f"Using previous hot status: {is_hot}")

    # Return the result as a dictionary
    return {
        "video_id": video_id,
        "last_updated_mnt": now,
        "hot_or_not": is_hot,
        "current_avg_ds_score": current_avg_ds,
        "reference_predicted_avg_ds_score": ref_predicted_avg_ds,
    }


# %%
def get_previous_hot_status(video_id, conn_string=conn_string):
    """
    Get the previous hot status for a video from the database.

    Args:
        video_id (str): The ID of the video
        conn_string (str): PostgreSQL connection string

    Returns:
        bool or None: Previous hot status, or None if not available
    """
    try:
        with psycopg.connect(conn_string) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT hot_or_not
                    FROM hot_or_not_evaluator.video_hot_or_not_status
                    WHERE video_id = %s
                    """,
                    (video_id,),
                )
                row = cur.fetchone()

                if row is not None:
                    return row[0]
    except Exception as e:
        print(f"Error getting previous hot status: {e}")

    return None  # Default if no previous status or error


# %%
def process_with_postgres(events_df, conn_string=conn_string):
    """
    Process the events data using PostgreSQL stored procedures.
    This calls update_counter for each event and then compute_hot_or_not.

    Args:
        events_df (pandas.DataFrame): DataFrame with events
        conn_string (str): PostgreSQL connection string

    Returns:
        dict: Hot or not status from database
    """
    print(f"Processing {len(events_df)} events with PostgreSQL...")

    # Connect to the database
    with psycopg.connect(conn_string) as conn:
        # First, let's check if the schema and function exist
        with conn.cursor() as cur:
            # Check if the schema exists
            cur.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = 'hot_or_not_evaluator');"
            )
            schema_exists = cur.fetchone()[0]
            if not schema_exists:
                print("ERROR: Schema 'hot_or_not_evaluator' does not exist!")
                return None

            # Check if the function exists
            cur.execute(
                """
                SELECT proname, pg_get_function_arguments(pg_proc.oid)
                FROM pg_proc
                JOIN pg_namespace ON pg_namespace.oid = pg_proc.pronamespace
                WHERE proname = 'update_counter'
                AND pg_namespace.nspname = 'hot_or_not_evaluator';
            """
            )
            func_info = cur.fetchone()
            if not func_info:
                print(
                    "ERROR: Function 'hot_or_not_evaluator.update_counter' does not exist!"
                )
                return None

            print(f"Function signature: {func_info[0]}({func_info[1]})")

        # Process events in batches to avoid large transactions
        batch_size = 100
        event_batches = [
            events_df.iloc[i : i + batch_size]
            for i in range(0, len(events_df), batch_size)
        ]

        for batch_idx, batch_df in enumerate(event_batches):
            with conn.cursor() as cur:
                # Set a long timeout for the transaction
                cur.execute("SET statement_timeout = 300000;")  # 5 minutes

                # Process each event in the batch
                for _, event in batch_df.iterrows():
                    try:
                        # Use direct SQL with explicit casts to ensure proper type handling
                        sql = f"""
                        SELECT hot_or_not_evaluator.update_counter(
                            '{event['video_id']}'::VARCHAR,
                            {str(event['liked']).lower()}::BOOLEAN,
                            {event['watch_percentage']}::NUMERIC
                        );
                        """
                        cur.execute(sql)
                    except Exception as e:
                        print(f"Error executing update_counter: {e}")
                        # Print parameter values for debugging
                        print(
                            f"Parameters: video_id='{event['video_id']}', "
                            f"liked={event['liked']}, "
                            f"watch_percentage={event['watch_percentage']}"
                        )

            # Commit after each batch
            conn.commit()
            print(f"Processed batch {batch_idx+1}/{len(event_batches)} with PostgreSQL")

    # After all events are processed, run compute_hot_or_not
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            print("Computing hot-or-not status with PostgreSQL...")
            try:
                cur.execute("SELECT hot_or_not_evaluator.compute_hot_or_not();")
                print("Compute hot-or-not completed successfully")
            except Exception as e:
                print(f"Error executing compute_hot_or_not: {e}")
        conn.commit()

    # Get the computed hot-or-not status
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT video_id, last_updated_mnt, hot_or_not,
                       current_avg_ds_score, reference_predicted_avg_ds_score
                FROM hot_or_not_evaluator.video_hot_or_not_status
                WHERE video_id = %s
                """,
                (events_df["video_id"].iloc[0],),
            )
            result = cur.fetchone()

            if result:
                status = {
                    "video_id": result[0],
                    "last_updated_mnt": result[1],
                    "hot_or_not": result[2],
                    "current_avg_ds_score": result[3],
                    "reference_predicted_avg_ds_score": result[4],
                }
                return status
            else:
                print("No hot-or-not status found after processing")
                return None


# %%
def compare_results(pandas_metrics, pandas_status, postgres_status):
    """
    Compare the results from pandas and PostgreSQL processing.

    Args:
        pandas_metrics (pandas.DataFrame): Minute metrics from pandas processing
        pandas_status (dict): Hot or not status from pandas processing
        postgres_status (dict): Hot or not status from PostgreSQL processing

    Returns:
        dict: Comparison results
    """
    # Check if both statuses are available
    if pandas_status is None or postgres_status is None:
        return {"error": "Missing status information for comparison"}

    # Compare hot or not statuses
    hot_status_match = pandas_status["hot_or_not"] == postgres_status["hot_or_not"]

    # Convert Decimal to float for comparison if needed
    postgres_current_ds = postgres_status["current_avg_ds_score"]
    if postgres_current_ds is not None and hasattr(
        postgres_current_ds, "to_eng_string"
    ):
        postgres_current_ds = float(postgres_current_ds)

    postgres_ref_ds = postgres_status["reference_predicted_avg_ds_score"]
    if postgres_ref_ds is not None and hasattr(postgres_ref_ds, "to_eng_string"):
        postgres_ref_ds = float(postgres_ref_ds)

    # Compare DS scores (with tolerance for floating-point differences)
    current_ds_diff = (
        abs(pandas_status["current_avg_ds_score"] - postgres_current_ds)
        if (
            pandas_status["current_avg_ds_score"] is not None
            and postgres_current_ds is not None
        )
        else None
    )

    ref_ds_diff = (
        abs(pandas_status["reference_predicted_avg_ds_score"] - postgres_ref_ds)
        if (
            pandas_status["reference_predicted_avg_ds_score"] is not None
            and postgres_ref_ds is not None
        )
        else None
    )

    # Prepare comparison results
    comparison = {
        "hot_status_match": hot_status_match,
        "pandas_hot": pandas_status["hot_or_not"],
        "postgres_hot": postgres_status["hot_or_not"],
        "current_ds_score_diff": current_ds_diff,
        "reference_ds_score_diff": ref_ds_diff,
        "pandas_current_ds": pandas_status["current_avg_ds_score"],
        "postgres_current_ds": postgres_current_ds,
        "pandas_reference_ds": pandas_status["reference_predicted_avg_ds_score"],
        "postgres_reference_ds": postgres_ref_ds,
    }

    return comparison


# %%
def retrieve_data_for_comparison(
    video_id, start_timestamp, end_timestamp, conn_string=conn_string
):
    """
    Retrieve data from the database for comparison with pandas results.

    Args:
        video_id (str): The ID of the video
        start_timestamp (datetime): Start timestamp for data retrieval
        end_timestamp (datetime): End timestamp for data retrieval
        conn_string (str): PostgreSQL connection string

    Returns:
        pandas.DataFrame: DataFrame with data from PostgreSQL
    """
    query = """
    SELECT
        video_id, timestamp_mnt, like_count_mnt, watch_count_mnt,
        average_percentage_watched_mnt, like_ctr_mnt,
        cumulative_like_count, cumulative_watch_count,
        cumulative_like_ctr, cumulative_average_percentage_watched,
        normalized_cumulative_like_ctr, normalized_cumulative_watch_percentage,
        harmonic_mean_of_like_count_and_watch_count, ds_score
    FROM hot_or_not_evaluator.video_engagement_relation
    WHERE video_id = %s
      AND timestamp_mnt >= %s AND timestamp_mnt <= %s
    ORDER BY timestamp_mnt
    """

    # Create SQLAlchemy engine from connection string
    # Convert psycopg conn_string to SQLAlchemy format
    db_params = {
        param.split("=")[0]: param.split("=")[1]
        for param in conn_string.split()
        if "=" in param
    }
    sqlalchemy_uri = f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    engine = create_engine(sqlalchemy_uri)

    # Use SQLAlchemy engine with pandas
    postgres_data = pd.read_sql_query(
        query, engine, params=(video_id, start_timestamp, end_timestamp)
    )

    # Also retrieve hot_or_not status for comparison
    status_query = """
    SELECT * FROM hot_or_not_evaluator.video_hot_or_not_status
    WHERE video_id = %s
    """
    status_data = pd.read_sql_query(status_query, engine, params=(video_id,))

    if not status_data.empty:
        print(
            f"Hot status from DB: {'Hot' if status_data['hot_or_not'].iloc[0] else 'Not Hot'}"
        )
        print(f"DB current_avg_ds_score: {status_data['current_avg_ds_score'].iloc[0]}")
        print(
            f"DB reference_predicted_avg_ds_score: {status_data['reference_predicted_avg_ds_score'].iloc[0]}"
        )

    return postgres_data


# %%
def visualize_score_comparisons(
    postgres_data, pandas_metrics, pandas_status, postgres_status, save_path=""
):
    # Set seaborn style
    sns.set_style("whitegrid")

    # Plot 1: DS scores over time
    plt.figure(figsize=(12, 6))

    # Plot PostgreSQL data
    sns.lineplot(
        x=postgres_data["timestamp_mnt"],
        y=postgres_data["ds_score"],
        label="PostgreSQL DS Score",
        color="navy",
    )

    # Plot pandas data
    if not pandas_metrics.empty:
        sns.lineplot(
            x=pandas_metrics["timestamp_mnt"],
            y=pandas_metrics["ds_score"],
            label="Pandas DS Score",
            color="crimson",
            linestyle="--",
        )

    plt.title("DS Score Comparison", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("DS Score", fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Plot 2: Bar comparison if we have current and reference scores
    if (
        pandas_status["current_avg_ds_score"] is not None
        and postgres_status["current_avg_ds_score"] is not None
    ):

        # Create DataFrame for better seaborn integration
        import pandas as pd

        comp_data = pd.DataFrame(
            {
                "Metric": [
                    "Current DS Score",
                    "Current DS Score",
                    "Reference DS Score",
                    "Reference DS Score",
                ],
                "Implementation": ["Pandas", "PostgreSQL", "Pandas", "PostgreSQL"],
                "Value": [
                    pandas_status["current_avg_ds_score"],
                    postgres_status["current_avg_ds_score"],
                    pandas_status["reference_predicted_avg_ds_score"],
                    postgres_status["reference_predicted_avg_ds_score"],
                ],
            }
        )
        plt.figure(figsize=(10, 6))

        # Create grouped bar chart
        ax = sns.barplot(
            x="Metric",
            y="Value",
            hue="Implementation",
            data=comp_data,
            palette=["crimson", "navy"],
        )

        # Add values on top of the bars
        for bar in ax.containers:
            ax.bar_label(bar, fmt="%.2f")

        plt.title("Current vs Reference DS Score Comparison", fontsize=14)
        plt.ylabel("DS Score", fontsize=12)
        plt.legend(title="Implementation", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

    return


# %%
def main(
    video_id="sgx-test_video_simple",
    start_timestamp=None,  # Default to None to use the defined default or database timestamp
    duration_hours=1,
    interval_minutes=4,
    events_per_interval=10,
    growth_rate=1.05,
):
    """
    Main function to run the activity simulation.

    Args:
        video_id (str): The ID of the video
        start_timestamp (datetime): Starting timestamp for the activity data
        duration_hours (int): Duration of simulation in hours
        interval_minutes (int): Interval between data points in minutes
        events_per_interval (int): Base number of events per interval
        growth_rate (float): Growth rate of events per interval

    Returns:
        None
    """
    # Use default timestamp if none provided

    print(f"Starting activity simulation for video {video_id} from {start_timestamp}")

    # 1. Get the latest timestamp from the database (if needed)
    if start_timestamp is None:
        start_timestamp = get_latest_timestamp(video_id)

    print(f"Using timestamp: {start_timestamp}")

    # 2. Generate activity data
    events_df = generate_activity_data(
        video_id=video_id,
        start_timestamp=start_timestamp,
        duration_hours=duration_hours,
        interval_minutes=interval_minutes,
        events_per_interval=events_per_interval,
        growth_rate=growth_rate,
    )
    print(
        f"Generated {len(events_df)} events across {events_df['timestamp_mnt'].nunique()} time intervals"
    )

    # Print the min and max timestamps
    print(
        f"Time range: {events_df['timestamp_mnt'].min()} to {events_df['timestamp_mnt'].max()}"
    )

    # 3. Process with PostgreSQL first
    print("Processing with PostgreSQL...")
    postgres_status = process_with_postgres(events_df)
    print("PostgreSQL processing complete")
    if postgres_status:
        print(
            f"PostgreSQL hot status: {'Hot' if postgres_status['hot_or_not'] else 'Not Hot'}"
        )
    else:
        print("Failed to get PostgreSQL status")

    # 4. Retrieve data from PostgreSQL for comparison
    start_timestamp_for_query = start_timestamp - timedelta(
        days=1
    )  # Include historical data
    end_timestamp = events_df["timestamp_mnt"].max()
    postgres_data = retrieve_data_for_comparison(
        video_id, start_timestamp_for_query, end_timestamp
    )
    print(f"Retrieved {len(postgres_data)} rows from PostgreSQL for comparison")

    # 5. Process with pandas using the events data
    print("Processing with pandas...")
    pandas_metrics, pandas_status = process_with_pandas(events_df)
    print("Pandas processing complete")
    print(f"Pandas hot status: {'Hot' if pandas_status['hot_or_not'] else 'Not Hot'}")

    # 6. Compare results
    if postgres_status:
        comparison = compare_results(pandas_metrics, pandas_status, postgres_status)
        print("\nComparison Results:")
        for key, value in comparison.items():
            print(f"{key}: {value}")

        # 7. Visualize data if enough data points
        if len(postgres_data) >= 2:
            visualize_score_comparisons(
                postgres_data, pandas_metrics, pandas_status, postgres_status
            )
    else:
        print("Cannot compare results: PostgreSQL processing failed")

    print("Activity simulation complete!")


# %%
if __name__ == "__main__":
    from one_day_history import (
        generate_data_to_populate_database,
        clean_database_post_data_population,
    )

    if True:
        clean_database_post_data_population(
            test_video_id_prefix="sgx-",
            end_time=datetime(2025, 4, 29, 16, 30, 0),
        )
        generate_data_to_populate_database()

    main(
        video_id="sgx-test_video_simple",
        start_timestamp=datetime(2025, 4, 29, 17, 00, 0),
        duration_hours=30 / 60,  # Just 25 minutes for testing
        interval_minutes=4,  # Generate data every 4 minutes
        events_per_interval=15,  # Start with 10 events per interval
        growth_rate=1.05,  # 5% growth rate per interval
    )

# %%
