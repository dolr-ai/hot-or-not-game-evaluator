import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import psycopg
import os
from tqdm import tqdm
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt


# %%
class ActivityGenerator:
    """
    A class to generate different patterns of activity data for video engagement.

    This class supports various patterns like steady increase, spike, decrease,
    fluctuating, and plateau patterns for simulating different user engagement scenarios.
    """

    def __init__(self, pattern_type="increase"):
        """
        Initialize the ActivityGenerator with a specific pattern type.

        Args:
            pattern_type (str): The type of activity pattern to generate.
                Options include: "increase", "spike", "decrease",
                "fluctuate", "plateau", "drop"
        """
        self.pattern_type = pattern_type
        self.supported_patterns = {
            "increase": self._generate_steady_increase,
            "spike": self._generate_spike,
            "decrease": self._generate_decrease,
            "fluctuate": self._generate_fluctuate,
            "plateau": self._generate_plateau,
            "drop": self._generate_drop,
            "random": self._generate_random,
        }

        if pattern_type not in self.supported_patterns:
            raise ValueError(
                f"Pattern type '{pattern_type}' is not supported. "
                f"Available patterns: {list(self.supported_patterns.keys())}"
            )

    def generate_data(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        **kwargs,
    ):
        """
        Generate activity data based on the selected pattern.

        Args:
            video_id (str): The ID of the video
            start_timestamp (datetime): Starting timestamp
            duration_hours (int): Duration of data to generate in hours
            interval_minutes (int): Interval between data points in minutes
            events_per_interval (int): Base number of events per interval
            **kwargs: Additional pattern-specific parameters

        Returns:
            pandas.DataFrame: DataFrame with activity data
        """
        # Call the appropriate pattern generation method
        return self.supported_patterns[self.pattern_type](
            video_id,
            start_timestamp,
            duration_hours,
            interval_minutes,
            events_per_interval,
            **kwargs,
        )

    def _generate_steady_increase(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        growth_rate=1.05,
        max_events_per_interval=1000,
        linear_increase=True,
        linear_increment=2,
        **kwargs,
    ):
        """
        Generate steadily increasing activity data.

        Args:
            growth_rate (float): Rate of growth per interval if using exponential growth (default 1.05 = 5% growth)
            max_events_per_interval (int): Maximum number of events per interval to prevent excessive growth
            linear_increase (bool): Whether to use linear increase instead of exponential growth
            linear_increment (int): Number of events to add per interval when using linear increase
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        current_events = events_per_interval

        # Generate data with steadily increasing activity
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=len(intervals),
            desc="Generating activity",
        ):
            # Calculate the number of events for this interval
            if linear_increase:
                # Linear increase: Add a fixed number of events each interval
                current_events = events_per_interval + (i * linear_increment)
                num_events = min(int(current_events), max_events_per_interval)
            else:
                # Original exponential growth
                num_events = min(int(current_events), max_events_per_interval)
                current_events *= growth_rate

            # Add events with increasing watch time and like probability
            self._add_events(data, video_id, timestamp, num_events, i, len(intervals))

        return pd.DataFrame(data)

    def _generate_spike(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        spike_position=0.5,
        spike_magnitude=3.0,
        max_events_per_interval=1000,
        **kwargs,
    ):
        """
        Generate activity data with a spike at a specific position.

        Args:
            spike_position (float): Position of spike (0-1), default 0.5 (middle)
            spike_magnitude (float): Magnitude of spike, default 3.0 (3x normal)
            max_events_per_interval (int): Maximum number of events per interval to prevent excessive spikes
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        num_intervals = len(intervals)
        spike_center = int(num_intervals * spike_position)

        # Generate data with a spike
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=len(intervals),
            desc="Generating activity",
        ):
            # Calculate distance from spike center (normalized to 0-1)
            distance = abs(i - spike_center) / (num_intervals / 2)

            # Apply gaussian-like spike effect
            spike_effect = spike_magnitude * np.exp(-5 * distance**2)

            # Calculate events with spike effect (with maximum cap)
            num_events = min(
                int(events_per_interval * (1 + spike_effect)), max_events_per_interval
            )

            # Add events
            self._add_events(data, video_id, timestamp, num_events, i, num_intervals)

        return pd.DataFrame(data)

    def _generate_decrease(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        decay_rate=0.95,
        min_events_per_interval=1,
        **kwargs,
    ):
        """
        Generate steadily decreasing activity data.

        Args:
            decay_rate (float): Rate of decay per interval (default 0.95 = 5% decrease)
            min_events_per_interval (int): Minimum number of events per interval to prevent excessive reduction
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        current_events = events_per_interval

        # Generate data with steadily decreasing activity
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=len(intervals),
            desc="Generating activity",
        ):
            # Calculate the number of events for this interval (declining over time)
            num_events = max(min_events_per_interval, int(current_events))
            current_events *= decay_rate

            # Add events with decreasing quality metrics over time
            self._add_events(
                data,
                video_id,
                timestamp,
                num_events,
                i,
                len(intervals),
                decreasing=True,
            )

        return pd.DataFrame(data)

    def _generate_fluctuate(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        fluctuation_amplitude=0.3,
        min_events_per_interval=1,
        max_events_per_interval=1000,
        **kwargs,
    ):
        """
        Generate fluctuating activity data with a sinusoidal pattern.

        Args:
            fluctuation_amplitude (float): Amplitude of fluctuation (0-1)
            min_events_per_interval (int): Minimum number of events per interval
            max_events_per_interval (int): Maximum number of events per interval
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        num_intervals = len(intervals)

        # Generate data with fluctuating activity
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=len(intervals),
            desc="Generating activity",
        ):
            # Calculate sinusoidal fluctuation effect
            fluctuation = 1 + fluctuation_amplitude * np.sin(
                i * 2 * np.pi / (num_intervals / 3)
            )

            # Calculate events with fluctuation, applying both minimum and maximum caps
            num_events = min(
                max_events_per_interval,
                max(min_events_per_interval, int(events_per_interval * fluctuation)),
            )

            # Add events
            self._add_events(data, video_id, timestamp, num_events, i, num_intervals)

        return pd.DataFrame(data)

    def _generate_plateau(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        growth_phase=0.3,
        plateau_level=2.0,
        max_events_per_interval=1000,
        **kwargs,
    ):
        """
        Generate activity that grows and then plateaus.

        Args:
            growth_phase (float): Proportion of time spent growing (0-1)
            plateau_level (float): Level at which activity plateaus (multiplier)
            max_events_per_interval (int): Maximum number of events per interval to prevent excessive growth
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        num_intervals = len(intervals)
        growth_end = int(num_intervals * growth_phase)
        current_events = events_per_interval

        # Growth rate to reach plateau level by growth_end
        growth_rate = plateau_level ** (1 / growth_end) if growth_end > 0 else 1

        # Generate data with growth followed by plateau
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=len(intervals),
            desc="Generating activity",
        ):
            if i < growth_end:
                # Growth phase (with cap)
                num_events = min(int(current_events), max_events_per_interval)
                current_events *= growth_rate
                # Flag to indicate we're in growth phase
                is_plateau_phase = False
            else:
                # Plateau phase (with cap)
                num_events = min(
                    int(events_per_interval * plateau_level), max_events_per_interval
                )
                # Flag to indicate we're in plateau phase
                is_plateau_phase = True

            # Add events - pass the plateau information
            self._add_events(
                data,
                video_id,
                timestamp,
                num_events,
                i,
                num_intervals,
                is_plateau_phase=is_plateau_phase,
                growth_end=growth_end,
                plateau_level=plateau_level,
            )

        return pd.DataFrame(data)

    def _generate_drop(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        drop_ratio=0.3,
        min_events_per_interval=1,
        **kwargs,
    ):
        """
        Generate activity data that starts high but then suddenly drops.
        This should result in a "not hot" prediction as current DS score
        will be lower than the predicted reference score.

        Args:
            drop_ratio (float): Ratio to which engagement drops (0-1)
            min_events_per_interval (int): Minimum events per interval
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        num_intervals = len(intervals)

        # The drop occurs at 70% of the way through the intervals,
        # which ensures that the current window (last 5 minutes) has the low engagement
        drop_point = int(num_intervals * 0.7)

        # Generate data with a drop pattern
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=len(intervals),
            desc="Generating activity",
        ):
            # Before the drop point, maintain high engagement
            if i < drop_point:
                num_events = events_per_interval
            # After the drop point, reduce engagement by the drop ratio
            else:
                num_events = max(
                    int(events_per_interval * drop_ratio), min_events_per_interval
                )

            # Add events with decreasing quality after the drop
            self._add_events(
                data,
                video_id,
                timestamp,
                num_events,
                i,
                num_intervals,
                decreasing=(i >= drop_point),  # Lower quality after drop
            )

        return pd.DataFrame(data)

    def _generate_random(
        self,
        video_id,
        start_timestamp,
        duration_hours,
        interval_minutes,
        events_per_interval,
        min_events=1,
        max_events=30,
        **kwargs,
    ):
        """
        Generate completely random activity data with no discernible pattern.

        Args:
            min_events (int): Minimum number of events per interval
            max_events (int): Maximum number of events per interval
        """
        # Create time intervals
        end_timestamp = start_timestamp + timedelta(hours=duration_hours)
        is_tz_aware = (
            hasattr(start_timestamp, "tzinfo") and start_timestamp.tzinfo is not None
        )

        if is_tz_aware:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
                tz=start_timestamp.tzinfo,
            )
        else:
            intervals = pd.date_range(
                start=start_timestamp + timedelta(minutes=interval_minutes),
                end=end_timestamp,
                freq=f"{interval_minutes}min",
            )

        data = []
        num_intervals = len(intervals)

        # Generate completely random data
        for i, timestamp in tqdm(
            enumerate(intervals),
            total=num_intervals,
            desc="Generating random activity",
        ):
            # Generate a random number of events between min and max
            num_events = np.random.randint(min_events, max_events + 1)

            # Add events with random metrics
            for _ in range(num_events):
                # Random watch percentage between 0 and 100
                watch_pct = np.random.uniform(0, 100)

                # Random like probability between 0 and 0.5
                like_probability = np.random.uniform(0, 0.5)

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

        return pd.DataFrame(data)

    def _add_events(
        self,
        data,
        video_id,
        timestamp,
        num_events,
        interval_index,
        total_intervals,
        decreasing=False,
        is_plateau_phase=False,
        growth_end=None,
        plateau_level=None,
    ):
        """
        Helper method to add events to the data list.

        Args:
            data (list): List to append events to
            video_id (str): Video ID
            timestamp (datetime): Timestamp for the events
            num_events (int): Number of events to add
            interval_index (int): Current interval index
            total_intervals (int): Total number of intervals
            decreasing (bool): Whether metrics should decrease over time
            is_plateau_phase (bool): Whether we're in the plateau phase for plateau pattern
            growth_end (int): Index when growth phase ends for plateau pattern
            plateau_level (float): Level at which metrics should plateau
        """
        # Calculate progress through the time series (0 to 1)
        progress = interval_index / total_intervals

        # For decreasing pattern, invert the progress
        if decreasing:
            quality_factor = 1 - progress
        elif (
            self.pattern_type == "plateau"
            and is_plateau_phase
            and growth_end is not None
        ):
            # For plateau pattern in plateau phase, use a constant quality factor
            # equal to the progress at the growth_end point
            plateau_progress = growth_end / total_intervals
            quality_factor = plateau_progress
        else:
            quality_factor = progress

        # Base metrics improve with progress (or decline if decreasing)
        base_like_probability = 0.1 + quality_factor * 0.2
        like_probability = min(0.5, base_like_probability)

        # Add the specified number of events
        for _ in range(num_events):
            # Keep watch percentage constant between 60-80%
            watch_pct = 60 + np.random.uniform(0, 20)

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


class BackFill:
    """
    A class to generate and backfill historical engagement data into the database.

    This class handles generating synthetic video engagement data and populating
    the database with that data for testing and development purposes. It uses
    ActivityGenerator to create different patterns of engagement data.
    """

    # Constants for normalization (from metric_const table)
    LIKE_CTR_CENTER = 0
    LIKE_CTR_RANGE = 0.05
    WATCH_PERCENTAGE_CENTER = 0
    WATCH_PERCENTAGE_RANGE = 0.9

    def __init__(self, conn_string=None, pattern_type="increase"):
        """
        Initialize the BackFill class.

        Args:
            conn_string (str, optional): PostgreSQL connection string. If not provided,
                                         it will attempt to load from environment variables.
            pattern_type (str): Type of activity pattern to generate. Options include:
                              "increase", "spike", "decrease", "fluctuate", "plateau"
        """
        # Initialize database connection
        if conn_string is None:
            # Load environment variables for DB connection
            env_path = "/Users/sagar/work/yral/hot-or-not-game-evaluator/test/.env"
            load_dotenv(env_path)

            # Database connection parameters
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")
            db_user = os.getenv("DB_USER")
            db_password = os.getenv("DB_PASSWORD")
            self.conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} sslmode=require"
        else:
            self.conn_string = conn_string

        # Create ActivityGenerator instance with the specified pattern
        self.activity_generator = ActivityGenerator(pattern_type=pattern_type)

    def generate_activity_data(
        self,
        video_id,
        end_time=datetime(2025, 4, 29, 14, 0, 0),
        interval_minutes=5,
        events_per_interval=10,
        period=timedelta(days=1),
        pattern_kwargs=None,
    ):
        """
        Generate synthetic video engagement data for a specific video ID, ending at end_time.
        Uses the ActivityGenerator to create data with the specified pattern.

        Args:
            video_id (str): The ID of the video to generate data for.
            end_time (datetime): The end timestamp for the data generation.
            period (timedelta): The time period to generate data for, default is 1 day. Meaning end_time-1d period's data will be generated.
            pattern_kwargs (dict): Additional keyword arguments for the specific pattern generator.

        Returns:
            pandas.DataFrame: DataFrame containing the generated engagement data.
        """
        if pattern_kwargs is None:
            pattern_kwargs = {}

        # Calculate start time and duration
        start_time = end_time - period
        duration_hours = period.total_seconds() / 3600

        print(
            f"Generating {duration_hours} hours of data for {video_id} with pattern {pattern_kwargs['pattern_type']}"
        )
        print(f"Start time: {start_time}, End time: {end_time}")
        # Generate activity data using ActivityGenerator
        raw_activity = self.activity_generator.generate_data(
            video_id=video_id,
            start_timestamp=start_time,
            duration_hours=duration_hours,
            interval_minutes=interval_minutes,
            events_per_interval=events_per_interval,
            **pattern_kwargs,
        )

        # Process raw activity data into the format expected by the database
        processed_df = self._process_activity_data(raw_activity, video_id)

        return processed_df

    def _process_activity_data(self, activity_df, video_id):
        """
        Process the raw activity data from ActivityGenerator into the format needed for database insertion.

        Args:
            activity_df (pandas.DataFrame): Raw activity data from ActivityGenerator
            video_id (str): Video ID to use

        Returns:
            pandas.DataFrame: Processed data ready for database insertion
        """
        # Group by timestamp and calculate aggregates
        grouped = activity_df.groupby(["video_id", "timestamp_mnt"])

        # Create a new DataFrame with the aggregated metrics
        result = pd.DataFrame()
        result["video_id"] = grouped.video_id.first()
        result["timestamp_mnt"] = grouped.timestamp_mnt.first().index.get_level_values(
            "timestamp_mnt"
        )
        result["watch_count_mnt"] = grouped.size()
        result["like_count_mnt"] = grouped.liked.sum()

        # Calculate average watch percentage per minute
        result["average_percentage_watched_mnt"] = grouped.watch_percentage.mean()

        # Reset index to get a flat DataFrame
        result = result.reset_index(drop=True)

        return result

    def populate_database_with_data(self, df):
        """
        Populate the database with the generated data.

        This function takes the synthetic data DataFrame and inserts it directly into the
        video_engagement_relation table in batches, bypassing the update_counter function.

        Args:
            df (pandas.DataFrame): DataFrame containing the generated engagement data
        """
        print(f"Populating database with {len(df)} minutes of data...")

        # Connect to the database
        with psycopg.connect(self.conn_string) as conn:
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
                                0,
                                (like_ctr - self.LIKE_CTR_CENTER) / self.LIKE_CTR_RANGE,
                            ),  # normalized_cumulative_like_ctr
                            max(
                                0,
                                (
                                    row["average_percentage_watched_mnt"]
                                    - self.WATCH_PERCENTAGE_CENTER
                                )
                                / self.WATCH_PERCENTAGE_RANGE,
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
        with psycopg.connect(self.conn_string) as conn:
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
        with psycopg.connect(self.conn_string) as conn:
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
                        0,
                        (cumulative_like_ctr - self.LIKE_CTR_CENTER)
                        / self.LIKE_CTR_RANGE,
                    )
                    norm_watch_pct = max(
                        0,
                        (cumulative_avg_watch_pct - self.WATCH_PERCENTAGE_CENTER)
                        / self.WATCH_PERCENTAGE_RANGE,
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
        with psycopg.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                print("Computing hot-or-not status...")
                cur.execute("SELECT hot_or_not_evaluator.compute_hot_or_not();")
            conn.commit()

        print("Data population complete!")

    def backfill_data(
        self,
        video_id,
        end_time=datetime(2025, 4, 29, 16, 30, 0),
        period=timedelta(days=1),
        interval_minutes=5,
        events_per_interval=10,
        pattern_kwargs=None,
        output_dir=None,
    ):
        """
        Generate activity data and populate the database in one step.

        Args:
            video_id (str): The ID of the video to generate data for.
            end_time (datetime): The end timestamp for the data generation.
            period (timedelta): The time period to generate data for, default is 1 day.
            pattern_kwargs (dict): Additional keyword arguments for the pattern generator.
            output_dir (str): Directory to save visualization outputs, defaults to "activity_data/activity_data_backfill"
        """
        data = self.generate_activity_data(
            video_id=video_id,
            end_time=end_time,
            period=period,
            interval_minutes=interval_minutes,
            events_per_interval=events_per_interval,
            pattern_kwargs=pattern_kwargs,
        )
        # visualize this data
        print("#" * 100)
        print(
            {
                "pattern_kwargs": pattern_kwargs,
                "len(data)": len(data),
                "sample_data": data.head().to_dict(),
                "data['timestamp_mnt'].min()": data["timestamp_mnt"].min(),
                "data['timestamp_mnt'].max()": data["timestamp_mnt"].max(),
            }
        )

        # Set default output directory if none provided
        if output_dir is None:
            # Create default directory structure
            root_dir = "activity_data"
            output_dir = os.path.join(root_dir, "activity_data_backfill")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        sns.set(style="whitegrid")  # Use seaborn's whitegrid style

        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot average percentage watched on the first axis
        color = "tab:blue"
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Average Percentage Watched", color=color)
        ax1.plot(
            data["timestamp_mnt"], data["average_percentage_watched_mnt"], color=color
        )
        ax1.tick_params(axis="y", labelcolor=color)

        # Create a second y-axis for watch count
        ax2 = ax1.twinx()
        color = "tab:red"
        ax2.set_ylabel("Watch Count", color=color)
        ax2.plot(data["timestamp_mnt"], data["watch_count_mnt"], color=color)
        ax2.tick_params(axis="y", labelcolor=color)

        plt.title(f"Activity Data for {video_id} ({pattern_kwargs['pattern_type']})")
        plt.xticks(rotation=45, ha="right")  # Rotate timestamps for visibility

        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, ["Avg % Watched", "Watch Count"], loc="upper right")

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping

        # Use a more concise filename that's still unique
        output_filename = f"{video_id}_{pattern_kwargs['pattern_type']}_activity.png"
        plt.savefig(os.path.join(output_dir, output_filename))
        plt.close()  # Close the plot to free memory

        print("#" * 100)
        self.populate_database_with_data(data)
        return data


# Static method for cleaning database outside of BackFill class
def clean_database_static(
    test_video_id_prefix="sgx-",
    end_time=datetime(2025, 4, 29, 16, 30, 0),
    conn_string=None,
):
    """
    Clean up test data from the database (static method version).

    Args:
        test_video_id_prefix (str): Prefix identifying test video IDs to clean up.
        end_time (datetime): End time boundary for cleanup. If None, all videos with the prefix are removed.
        conn_string (str): Database connection string. If None, will attempt to load from env.
    """
    # If conn_string is not provided, load from environment variables
    if conn_string is None:
        env_path = "/Users/sagar/work/yral/hot-or-not-game-evaluator/test/.env"
        load_dotenv(env_path)

        # Database connection parameters
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} sslmode=require"

    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM hot_or_not_evaluator.video_hot_or_not_status WHERE video_id LIKE %s",
                (f"{test_video_id_prefix}%",),
            )
            hot_status_rows_deleted = cur.rowcount
            conn.commit()

            if end_time is None:
                cur.execute(
                    "DELETE FROM hot_or_not_evaluator.video_engagement_relation WHERE video_id LIKE %s",
                    (f"{test_video_id_prefix}%",),
                )
            else:
                cur.execute(
                    "DELETE FROM hot_or_not_evaluator.video_engagement_relation WHERE video_id LIKE %s AND timestamp_mnt <= %s",
                    (f"{test_video_id_prefix}%", end_time),
                )
            engagement_rows_deleted = cur.rowcount
            conn.commit()

    print(
        f"Cleaned up test data with video_id prefix '{test_video_id_prefix}'. "
        f"Removed {hot_status_rows_deleted} rows from video_hot_or_not_status and {engagement_rows_deleted} rows from video_engagement_relation."
    )
