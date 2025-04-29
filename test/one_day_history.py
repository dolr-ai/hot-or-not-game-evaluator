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

# %%
# Constants for normalization (from metric_const table)
LIKE_CTR_CENTER = 0
LIKE_CTR_RANGE = 0.2
WATCH_PERCENTAGE_CENTER = 0
WATCH_PERCENTAGE_RANGE = 0.7


# %%
def generate_one_day_data(
    end_time=datetime(2025, 4, 29, 14, 0, 0),
    period=timedelta(days=1),
    video_id="sgx-test_video_simple",
):
    """
    Generate one day of synthetic video engagement data for a specific video ID, ending at end_time.

    Args:
        end_time (datetime): The end timestamp for the data generation.
        video_id (str): The ID of the video to generate data for.

    Returns:
        pandas.DataFrame: DataFrame containing the generated engagement data.
    """
    # Set up time range - 1 day of data in minute increments, going backwards from end_time
    start_time = end_time - period

    # Create a dataframe with one row per minute for the past day
    minutes = pd.date_range(start=start_time, end=end_time, freq="1min")
    df = pd.DataFrame({"timestamp_mnt": minutes})

    # Add video_id column
    df["video_id"] = video_id

    # Generate synthetic watch count data with a daily pattern and some randomness
    # More views during daytime hours (9am-11pm)
    df["hour"] = df["timestamp_mnt"].dt.hour
    base_watch_count = np.random.randint(2, 6)  # Base number of views per minute

    # Create time-based patterns
    df["watch_count_mnt"] = base_watch_count + np.sin(np.pi * df["hour"] / 12) * 10
    # Add some random noise
    df["watch_count_mnt"] = df["watch_count_mnt"] + np.random.normal(0, 2, size=len(df))
    # Make sure counts are integers and at least 1
    df["watch_count_mnt"] = np.maximum(1, df["watch_count_mnt"].astype(int))

    # Generate like count with a varying CTR over time
    base_ctr = 0.05  # 5% baseline like rate
    # CTR improves over the day as video gets recommended to more targeted viewers
    df["like_ctr"] = (
        base_ctr + (df.index / len(df)) * 0.1
    )  # Gradually increases from 5% to 15%
    # Add some random variation
    df["like_ctr"] = df["like_ctr"] + np.random.normal(0, 0.02, size=len(df))
    df["like_ctr"] = np.clip(
        df["like_ctr"], 0.01, 0.30
    )  # Keep within reasonable bounds

    # Calculate like count based on watch count and CTR
    df["like_count_mnt"] = (df["watch_count_mnt"] * df["like_ctr"]).astype(int)

    # Generate average watch percentage data - typically between 30% and 90%
    # Assume watch percentage improves as video gets shown to more relevant audience
    df["average_percentage_watched_mnt"] = 40 + (df.index / len(df)) * 20
    # Add random noise
    df["average_percentage_watched_mnt"] += np.random.normal(0, 10, size=len(df))
    # Clip to reasonable values
    df["average_percentage_watched_mnt"] = np.clip(
        df["average_percentage_watched_mnt"], 20, 95
    )

    # Create a spike in engagement for a "hot" period (4 hours of better than usual performance)
    # For this example, we'll make the video "hot" for 4 hours starting 6 hours ago
    hot_start = end_time - timedelta(hours=6)
    hot_end = hot_start + timedelta(hours=4)
    hot_mask = (df["timestamp_mnt"] >= hot_start) & (df["timestamp_mnt"] <= hot_end)

    # During "hot" period: double the views, improve CTR, and improve watch percentage
    df.loc[hot_mask, "watch_count_mnt"] = df.loc[hot_mask, "watch_count_mnt"] * 2
    df.loc[hot_mask, "like_ctr"] = df.loc[hot_mask, "like_ctr"] * 1.5
    df.loc[hot_mask, "like_count_mnt"] = (
        df.loc[hot_mask, "watch_count_mnt"] * df.loc[hot_mask, "like_ctr"]
    ).astype(int)
    df.loc[hot_mask, "average_percentage_watched_mnt"] = np.minimum(
        95, df.loc[hot_mask, "average_percentage_watched_mnt"] * 1.3
    )

    # Drop helper columns
    df = df.drop(columns=["hour", "like_ctr"])

    # Ensure data types are correct
    df["like_count_mnt"] = df["like_count_mnt"].astype(int)
    df["watch_count_mnt"] = df["watch_count_mnt"].astype(int)

    return df


# %%
def populate_database_with_data(df, conn_string):
    """
    Populate the database with the generated data.

    This function takes the synthetic data DataFrame and inserts it directly into the
    video_engagement_relation table in batches, bypassing the update_counter function.

    Args:
        df (pandas.DataFrame): DataFrame containing the generated engagement data
        conn_string (str): PostgreSQL connection string
    """
    print(f"Populating database with {len(df)} minutes of data...")

    # Connect to the database
    with psycopg.connect(conn_string) as conn:
        # Insert directly into the video_engagement_relation table in batches
        batch_size = 1000
        total_batches = (len(df) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            print(
                f"Processing batch {batch_idx + 1}/{total_batches} ({start_idx} to {end_idx-1})..."
            )

            # Prepare data for bulk insert
            values = []
            for _, row in batch_df.iterrows():
                # Calculate like_ctr_mnt
                like_ctr = (
                    (row["like_count_mnt"] * 100 / row["watch_count_mnt"])
                    if row["watch_count_mnt"] > 0
                    else 0
                )

                # For direct insertion, we're setting all cumulative metrics equal to the minute metrics
                # This is a simplification for backfilling; in reality, they would accumulate over time
                values.append(
                    (
                        row["video_id"],  # video_id
                        row["timestamp_mnt"],  # timestamp_mnt
                        row["like_count_mnt"],  # like_count_mnt
                        row[
                            "average_percentage_watched_mnt"
                        ],  # average_percentage_watched_mnt
                        row["watch_count_mnt"],  # watch_count_mnt
                        like_ctr,  # like_ctr_mnt
                        row["like_count_mnt"],  # cumulative_like_count
                        row["watch_count_mnt"],  # cumulative_watch_count
                        like_ctr,  # cumulative_like_ctr
                        row[
                            "average_percentage_watched_mnt"
                        ],  # cumulative_average_percentage_watched
                        max(
                            0, (like_ctr - LIKE_CTR_CENTER) / LIKE_CTR_RANGE
                        ),  # normalized_cumulative_like_ctr
                        max(
                            0,
                            (
                                row["average_percentage_watched_mnt"]
                                - WATCH_PERCENTAGE_CENTER
                            )
                            / WATCH_PERCENTAGE_RANGE,
                        ),  # normalized_cumulative_watch_percentage
                    )
                )

            # Bulk insert data
            with conn.cursor() as cur:
                # We'll use a prepared statement for the insert
                insert_stmt = """
                INSERT INTO hot_or_not_evaluator.video_engagement_relation (
                    video_id, timestamp_mnt, like_count_mnt, average_percentage_watched_mnt,
                    watch_count_mnt, like_ctr_mnt, cumulative_like_count, cumulative_watch_count,
                    cumulative_like_ctr, cumulative_average_percentage_watched,
                    normalized_cumulative_like_ctr, normalized_cumulative_watch_percentage
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (video_id, timestamp_mnt) DO UPDATE
                SET like_count_mnt = EXCLUDED.like_count_mnt,
                    average_percentage_watched_mnt = EXCLUDED.average_percentage_watched_mnt,
                    watch_count_mnt = EXCLUDED.watch_count_mnt,
                    like_ctr_mnt = EXCLUDED.like_ctr_mnt,
                    cumulative_like_count = EXCLUDED.cumulative_like_count,
                    cumulative_watch_count = EXCLUDED.cumulative_watch_count,
                    cumulative_like_ctr = EXCLUDED.cumulative_like_ctr,
                    cumulative_average_percentage_watched = EXCLUDED.cumulative_average_percentage_watched,
                    normalized_cumulative_like_ctr = EXCLUDED.normalized_cumulative_like_ctr,
                    normalized_cumulative_watch_percentage = EXCLUDED.normalized_cumulative_watch_percentage
                """

                # Execute batch insert
                cur.executemany(insert_stmt, values)

            # Commit after each batch
            conn.commit()
            print(f"Batch {batch_idx + 1} committed.")

    # Calculate harmonic mean and ds_score in a separate update
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            print("Calculating harmonic mean and ds_score...")
            update_stmt = """
            UPDATE hot_or_not_evaluator.video_engagement_relation
            SET
                harmonic_mean_of_like_count_and_watch_count =
                    ((normalized_cumulative_like_ctr + 1) * (normalized_cumulative_watch_percentage + 1)) /
                    (normalized_cumulative_like_ctr + normalized_cumulative_watch_percentage + 2),
                ds_score =
                    ((normalized_cumulative_like_ctr + 1) * (normalized_cumulative_watch_percentage + 1)) /
                    (normalized_cumulative_like_ctr + normalized_cumulative_watch_percentage + 2) - 1
            WHERE video_id = %s
            """
            cur.execute(update_stmt, (df["video_id"].iloc[0],))
            print(f"Updated {cur.rowcount} rows with harmonic mean and ds_score.")
        conn.commit()

    # Fix the cumulative metrics to be truly cumulative over time
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            print("Fixing cumulative metrics...")

            # First, get all data sorted by timestamp
            cur.execute(
                """
                SELECT video_id, timestamp_mnt, like_count_mnt, watch_count_mnt, average_percentage_watched_mnt
                FROM hot_or_not_evaluator.video_engagement_relation
                WHERE video_id = %s
                ORDER BY timestamp_mnt
            """,
                (df["video_id"].iloc[0],),
            )

            rows = cur.fetchall()

            # Calculate running totals
            cumulative_like_count = 0
            cumulative_watch_count = 0
            cumulative_watch_percentage_sum = 0

            # Prepare batch updates
            update_stmt = """
                UPDATE hot_or_not_evaluator.video_engagement_relation
                SET
                    cumulative_like_count = %s,
                    cumulative_watch_count = %s,
                    cumulative_like_ctr = %s,
                    cumulative_average_percentage_watched = %s,
                    normalized_cumulative_like_ctr = %s,
                    normalized_cumulative_watch_percentage = %s,
                    harmonic_mean_of_like_count_and_watch_count = %s,
                    ds_score = %s
                WHERE video_id = %s AND timestamp_mnt = %s
            """

            update_data = []
            for row in rows:
                video_id, timestamp, like_count, watch_count, avg_watch_pct = row

                # Update running totals
                cumulative_like_count += like_count
                cumulative_watch_count += watch_count
                cumulative_watch_percentage_sum += avg_watch_pct * watch_count

                # Calculate derived metrics
                cumulative_avg_watch_pct = (
                    cumulative_watch_percentage_sum / cumulative_watch_count
                    if cumulative_watch_count > 0
                    else 0
                )
                cumulative_like_ctr = (
                    (cumulative_like_count * 100 / cumulative_watch_count)
                    if cumulative_watch_count > 0
                    else 0
                )

                # Normalize metrics
                norm_like_ctr = max(
                    0, (cumulative_like_ctr - LIKE_CTR_CENTER) / LIKE_CTR_RANGE
                )
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

                # Add to update batch
                update_data.append(
                    (
                        cumulative_like_count,
                        cumulative_watch_count,
                        cumulative_like_ctr,
                        cumulative_avg_watch_pct,
                        norm_like_ctr,
                        norm_watch_pct,
                        harmonic_mean,
                        ds_score,
                        video_id,
                        timestamp,
                    )
                )

            # Execute updates in batches
            batch_size = 1000
            for i in range(0, len(update_data), batch_size):
                batch = update_data[i : i + batch_size]
                cur.executemany(update_stmt, batch)
                conn.commit()
                print(f"Updated cumulative metrics for {len(batch)} rows.")

    # After all data is loaded, run the hot-or-not computation
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            print("Computing hot-or-not status...")
            cur.execute("SELECT hot_or_not_evaluator.compute_hot_or_not();")
        conn.commit()

    print("Data population complete!")


def generate_data_to_populate_database(
    end_time=datetime(2025, 4, 29, 16, 30, 0),
    period=timedelta(days=1),
    video_id="sgx-test_video_simple",
):
    data = generate_one_day_data(end_time, period, video_id)
    populate_database_with_data(data, conn_string)


def clean_database_post_data_population(
    test_video_id_prefix="sgx-",
    end_time=datetime(2025, 4, 29, 16, 30, 0),
):
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM hot_or_not_evaluator.video_hot_or_not_status WHERE video_id LIKE %s",
                (f"{test_video_id_prefix}%",),
            )
            conn.commit()

            cur.execute(
                "DELETE FROM hot_or_not_evaluator.video_engagement_relation WHERE video_id LIKE %s AND timestamp_mnt <= %s",
                (f"{test_video_id_prefix}%", end_time),
            )
            conn.commit()


# %%
# Example usage:
if __name__ == "__main__":
    # Generate the data
    data = generate_one_day_data(
        end_time=datetime(2025, 4, 29, 16, 30, 0),
        period=timedelta(days=1),
        video_id="sgx-test_video_simple",
    )

    # Display a sample of the data
    print(f"Generated {len(data)} minutes of data")
    display(data.head())

    # Print min and max timestamp
    print(f"Min timestamp: {data['timestamp_mnt'].min()}")
    print(f"Max timestamp: {data['timestamp_mnt'].max()}")

    # Populate the database
    populate_database_with_data(data, conn_string)

    # Check hot status after populating
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM hot_or_not_evaluator.video_hot_or_not_status WHERE video_id = %s",
                ("sgx-test_video_simple",),
            )
            result = cur.fetchone()
            if result:
                print(f"Hot status: {'Hot' if result[2] else 'Not Hot'}")
                print(f"Current avg DS score: {result[5]}")
                print(f"Reference predicted DS score: {result[6]}")
            else:
                print("No hot status found")
# # %%
# result
# # %%
# data["watch_count_mnt"].describe()
