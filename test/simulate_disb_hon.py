import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
from sqlalchemy import create_engine
import random
from collections import defaultdict
import asyncio
import concurrent.futures

# Import ActivityGenerator and BackFill from activity_generator.py
from activity_generator import ActivityGenerator, BackFill, clean_database_static

# Load environment variables
load_dotenv("/Users/sagar/work/yral/hot-or-not-game-evaluator/prod.env")

# Database connection parameters
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} sslmode=require"

# Constants for normalization (from metric_const table)
LIKE_CTR_CENTER = 0
LIKE_CTR_RANGE = 0.05
WATCH_PERCENTAGE_CENTER = 0
WATCH_PERCENTAGE_RANGE = 0.9

# Directory structure for output
ROOT_OUTPUT_DIR = "simulation_results_disb_hon_random"
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)


class HotOrNotGameSimulator:
    """
    Class to simulate multiple hot-or-not prediction games and analyze results
    """

    def __init__(self, conn_string=conn_string):
        """Initialize the simulator with database connection"""
        self.conn_string = conn_string
        self.pattern_results = defaultdict(
            lambda: {
                "hot_count": 0,
                "not_hot_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "failed_count": 0,
                "game_results": [],
            }
        )

    def get_latest_timestamp(self, video_id, override_timestamp=None):
        """Get the latest timestamp for a video from the database or use override timestamp"""
        if override_timestamp:
            return override_timestamp

        with psycopg.connect(self.conn_string) as conn:
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
            latest_timestamp = datetime(2025, 4, 29, 14, 0, 0)

        return latest_timestamp

    def generate_activity_data(
        self,
        video_id,
        pattern_type,
        start_timestamp=None,
        duration_hours=0.5,
        interval_minutes=4,
        events_per_interval=10,
        **kwargs,
    ):
        """Generate activity data using the ActivityGenerator class"""
        if start_timestamp is None:
            start_timestamp = self.get_latest_timestamp(video_id)

        # Minimal logging
        if kwargs.get("verbose", False):
            tqdm.write(
                f"Generating {pattern_type} activity data from: {start_timestamp}"
            )

        # Create generator with specified pattern
        generator = ActivityGenerator(pattern_type=pattern_type)
        return generator.generate_data(
            video_id=video_id,
            start_timestamp=start_timestamp,
            duration_hours=duration_hours,
            interval_minutes=interval_minutes,
            events_per_interval=events_per_interval,
            **kwargs,
        )

    def process_with_pandas(self, events_df):
        """
        Process the events data using pandas to mimic the stored procedures.
        This simulates what happens in update_counter and compute_hot_or_not.

        Args:
            events_df (pandas.DataFrame): DataFrame with events

        Returns:
            dict: Hot or not status computed in memory
        """
        video_id = events_df["video_id"].iloc[0]

        # Get previous cumulative metrics (this will be empty since we're in-memory only)
        previous_cumulative = {
            "cumulative_like_count": 0,
            "cumulative_watch_count": 0,
            "cumulative_watched_sum": 0,
        }

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

        # 3. Create a copy for cumulative calculations
        minute_metrics = minute_agg.copy()

        # Sort by timestamp to ensure proper cumulative calculations
        minute_metrics = minute_metrics.sort_values("timestamp_mnt")

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

        # Constants for normalization (from metric_const table)
        LIKE_CTR_CENTER = 0
        LIKE_CTR_RANGE = 0.05
        WATCH_PERCENTAGE_CENTER = 0
        WATCH_PERCENTAGE_RANGE = 0.9

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
        minute_metrics["cumulative_average_percentage_watched"] = (
            cumulative_avg_watch_pcts
        )
        minute_metrics["normalized_cumulative_like_ctr"] = norm_like_ctrs
        minute_metrics["normalized_cumulative_watch_percentage"] = norm_watch_pcts
        minute_metrics["harmonic_mean_of_like_count_and_watch_count"] = harmonic_means
        minute_metrics["ds_score"] = ds_scores

        # Compute hot-or-not status
        now = datetime.now()
        five_mins_ago = now - timedelta(minutes=5)
        one_day_ago = now - timedelta(days=1)

        # Calculate current average ds_score (last 5 minutes)
        current_window = minute_metrics[
            (minute_metrics["timestamp_mnt"] >= five_mins_ago)
            & (minute_metrics["timestamp_mnt"] < now)
        ]

        if len(current_window) > 0:
            current_avg_ds = current_window["ds_score"].mean()
        else:
            current_avg_ds = None

        # Calculate reference period metrics (1 day ago to 5 mins ago)
        reference_period = minute_metrics[
            (minute_metrics["timestamp_mnt"] >= one_day_ago)
            & (minute_metrics["timestamp_mnt"] < five_mins_ago)
        ]

        # Perform linear regression if we have enough data points
        ref_predicted_avg_ds = None
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

                # Calculate regression parameters
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

                        # Calculate the timestamp for the midpoint of current window
                        current_midpoint = (
                            now.timestamp() + five_mins_ago.timestamp()
                        ) / 2

                        # Calculate predicted value at midpoint of current window
                        ref_predicted_avg_ds = slope * current_midpoint + intercept
            except Exception as e:
                print(f"Error in regression calculation: {e}")

        # Determine if video is hot
        if current_avg_ds is not None and ref_predicted_avg_ds is not None:
            is_hot = current_avg_ds > ref_predicted_avg_ds
        else:
            is_hot = None

        # Return the result as a dictionary
        return {
            "video_id": video_id,
            "last_updated_mnt": now,
            "hot_or_not": is_hot,
            "current_avg_ds_score": current_avg_ds,
            "reference_predicted_avg_ds_score": ref_predicted_avg_ds,
        }

    def is_prediction_correct(self, status, pattern_type):
        """
        Determine if the hot-or-not prediction matched the expected trend

        For increasing pattern: Should be hot
        For decreasing pattern: Should be not hot
        For spike/plateau patterns: Depends on where in the pattern we are
        For fluctuating pattern: Can go either way
        """
        if status is None or status["hot_or_not"] is None:
            return False

        # Simple trend-based expected outcomes
        if pattern_type == "increase":
            # For increasing pattern, we expect hot
            return status["hot_or_not"] == True
        elif pattern_type == "decrease":
            # For decreasing pattern, we expect not hot
            return status["hot_or_not"] == False
        else:
            # For other patterns, check if current_avg_ds_score > reference_predicted_avg_ds_score
            # This is considered a win because the prediction is based on actual calculation
            if (
                status["current_avg_ds_score"] is not None
                and status["reference_predicted_avg_ds_score"] is not None
            ):
                is_hot = (
                    status["current_avg_ds_score"]
                    > status["reference_predicted_avg_ds_score"]
                )
                return status["hot_or_not"] == is_hot
            return False

    async def run_game_async(
        self, pattern_type, video_id, game_number, start_timestamp=None, **kwargs
    ):
        """Run a single hot-or-not prediction game asynchronously"""
        # Use tqdm.write to avoid messing up the progress bar
        if game_number % 5 == 0:  # Only log every 5th game to reduce verbosity
            tqdm.write(
                f"Starting game #{game_number} for {pattern_type} pattern on {video_id}"
            )

        # Get the latest timestamp or use provided start timestamp
        if start_timestamp is None:
            timestamp = self.get_latest_timestamp(video_id)
        else:
            timestamp = start_timestamp

        # Add variation to pattern parameters for each run
        varied_kwargs = self.add_pattern_variation(pattern_type, kwargs, game_number)

        # Generate activity data - reduce verbosity
        events_df = self.generate_activity_data(
            video_id=video_id,
            pattern_type=pattern_type,
            start_timestamp=timestamp,
            **varied_kwargs,
        )

        # Process with pandas instead of PostgreSQL
        status = self.process_with_pandas(events_df)

        # Determine if prediction was correct
        prediction_correct = self.is_prediction_correct(status, pattern_type)

        # Log result concisely
        if status and status["hot_or_not"] is not None:
            result_str = "✅" if prediction_correct else "❌"
            hot_status = "🔥" if status["hot_or_not"] else "❄️"
            if game_number % 5 == 0:  # Only log every 5th game
                tqdm.write(
                    f"Game #{game_number} {pattern_type}: {hot_status} {result_str}"
                )

        # Store the result
        result = {
            "game_number": game_number,
            "pattern_type": pattern_type,
            "video_id": video_id,
            "hot_or_not": status["hot_or_not"] if status else None,
            "current_avg_ds_score": (
                float(status["current_avg_ds_score"])
                if status and status["current_avg_ds_score"] is not None
                else None
            ),
            "reference_predicted_avg_ds_score": (
                float(status["reference_predicted_avg_ds_score"])
                if status and status["reference_predicted_avg_ds_score"] is not None
                else None
            ),
            "prediction_correct": prediction_correct,
            "timestamp": datetime.now().isoformat(),
            "parameters": varied_kwargs,
        }

        # Update pattern results
        if status and status["hot_or_not"] is not None:
            if status["hot_or_not"]:
                self.pattern_results[pattern_type]["hot_count"] += 1
            else:
                self.pattern_results[pattern_type]["not_hot_count"] += 1

            if prediction_correct:
                self.pattern_results[pattern_type]["win_count"] += 1
            else:
                self.pattern_results[pattern_type]["loss_count"] += 1

        self.pattern_results[pattern_type]["game_results"].append(result)

        return result

    def add_pattern_variation(self, pattern_type, kwargs, game_number):
        """Add variation to pattern parameters based on game number"""
        varied_kwargs = kwargs.copy()

        # Use game number to create variation
        # Variation factor is between 0.8 and 1.2
        variation_factor = 0.8 + (game_number % 10) * 0.04

        if pattern_type == "increase":
            if "growth_rate" in varied_kwargs:
                varied_kwargs["growth_rate"] *= variation_factor
            if "linear_increment" in varied_kwargs:
                varied_kwargs["linear_increment"] = max(
                    1, int(varied_kwargs["linear_increment"] * variation_factor)
                )

        elif pattern_type == "decrease":
            if "decay_rate" in varied_kwargs:
                # For decay, we want to stay below 1.0, so use different variation
                varied_decay = (
                    varied_kwargs["decay_rate"] + (variation_factor - 1.0) * 0.05
                )
                varied_kwargs["decay_rate"] = min(0.99, max(0.75, varied_decay))

        elif pattern_type == "spike":
            if "spike_magnitude" in varied_kwargs:
                varied_kwargs["spike_magnitude"] *= variation_factor
            if "spike_position" in varied_kwargs:
                # Vary spike position between 0.3 and 0.8
                varied_kwargs["spike_position"] = 0.3 + (game_number % 10) * 0.05

        elif pattern_type == "fluctuate":
            if "fluctuation_amplitude" in varied_kwargs:
                varied_kwargs["fluctuation_amplitude"] *= variation_factor

        elif pattern_type == "plateau":
            if "plateau_level" in varied_kwargs:
                varied_kwargs["plateau_level"] *= variation_factor
            if "growth_phase" in varied_kwargs:
                # Vary growth phase between 0.2 and 0.7
                varied_kwargs["growth_phase"] = 0.2 + (game_number % 10) * 0.05

        # Vary common parameters
        if "events_per_interval" in varied_kwargs:
            varied_kwargs["events_per_interval"] = max(
                5, int(varied_kwargs["events_per_interval"] * variation_factor)
            )

        # Only log detailed parameter variations occasionally to reduce verbosity
        if game_number % 10 == 0:
            tqdm.write(f"Game #{game_number} variation factor: {variation_factor:.2f}")

        return varied_kwargs

    async def run_multiple_games_async(
        self, pattern_configs, num_games=50, concurrency=3, start_timestamp=None
    ):
        """
        Run multiple hot-or-not prediction games for each pattern asynchronously

        Args:
            pattern_configs: List of pattern configurations
            num_games: Number of games to run for each pattern
            concurrency: Maximum number of concurrent games
            start_timestamp: Optional fixed start timestamp for all games

        Returns:
            Dictionary with pattern results
        """
        self.pattern_results = defaultdict(
            lambda: {
                "hot_count": 0,
                "not_hot_count": 0,
                "win_count": 0,
                "loss_count": 0,
                "failed_count": 0,  # Track failed games
                "game_results": [],
            }
        )

        # Calculate total number of games for progress tracking
        total_games = len(pattern_configs) * num_games
        progress_bar = tqdm(total=total_games, desc="Total games progress")
        completed_games = 0

        # Process each pattern type
        for pattern_config in pattern_configs:
            pattern_type = pattern_config.pop("pattern_type")
            video_id = pattern_config.pop("video_id")

            tqdm.write(
                f"\n{'='*80}\nRunning {num_games} games for {pattern_type} pattern on {video_id}\n{'='*80}"
            )

            # Run games with limited concurrency
            tasks = []
            semaphore = asyncio.Semaphore(concurrency)

            async def run_game_with_semaphore(game_number):
                nonlocal completed_games
                async with semaphore:
                    # Small delay to avoid overwhelming the system
                    await asyncio.sleep(0.1)

                    result = await self.run_game_async(
                        pattern_type=pattern_type,
                        video_id=video_id,
                        game_number=game_number,
                        start_timestamp=start_timestamp,  # Pass the start timestamp
                        **pattern_config,
                    )
                    # Update progress bar after each game completes
                    completed_games += 1
                    progress_bar.update(1)

                    if result is None:
                        # Track failed games
                        self.pattern_results[pattern_type]["failed_count"] += 1
                        progress_bar.set_postfix(
                            pattern=pattern_type,
                            completed=f"{completed_games}/{total_games}",
                            failed=self.pattern_results[pattern_type]["failed_count"],
                        )
                    else:
                        progress_bar.set_postfix(
                            pattern=pattern_type,
                            completed=f"{completed_games}/{total_games}",
                        )

                    return result

            # Create tasks for all games
            for game_number in range(1, num_games + 1):
                tasks.append(run_game_with_semaphore(game_number))

            # Wait for all games to complete
            results = await asyncio.gather(*tasks)

            # Count successful games
            successful_games = sum(1 for r in results if r is not None)
            failed_games = sum(1 for r in results if r is None)

            # Display pattern results summary
            pattern_results = self.pattern_results[pattern_type]
            total_pattern_games = (
                pattern_results["hot_count"] + pattern_results["not_hot_count"]
            )
            if total_pattern_games > 0:
                win_rate = pattern_results["win_count"] / total_pattern_games * 100
                hot_rate = pattern_results["hot_count"] / total_pattern_games * 100

                tqdm.write(f"\n{'-'*40}")
                tqdm.write(f"Summary for {pattern_type} pattern:")
                tqdm.write(
                    f"Total games: {successful_games}/{num_games} ({failed_games} failed)"
                )
                tqdm.write(
                    f"Hot: {pattern_results['hot_count']} ({hot_rate:.1f}%) | Not Hot: {pattern_results['not_hot_count']} ({100-hot_rate:.1f}%)"
                )
                tqdm.write(
                    f"Correct predictions: {pattern_results['win_count']} ({win_rate:.1f}%) | Incorrect: {pattern_results['loss_count']} ({100-win_rate:.1f}%)"
                )
                tqdm.write(f"{'-'*40}\n")

            # Save intermediate results after each pattern
            self.save_results(
                f"{pattern_type}_results.json", self.pattern_results[pattern_type]
            )

            # Short delay between pattern types
            await asyncio.sleep(1)

        # Close the progress bar
        progress_bar.close()

        return dict(self.pattern_results)

    def save_results(self, filename, results):
        """Save results to a JSON file"""
        file_path = os.path.join(ROOT_OUTPUT_DIR, filename)

        # Create a custom encoder to handle non-serializable types
        class CustomJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                # Handle datetime objects
                if isinstance(obj, datetime):
                    return obj.isoformat()
                # Handle numpy types
                elif hasattr(obj, "item"):
                    return obj.item()
                # Handle other non-serializable types
                elif hasattr(obj, "__dict__"):
                    return obj.__dict__
                else:
                    try:
                        # Try converting to a standard type
                        return (
                            float(obj)
                            if isinstance(obj, (int, float, bool))
                            else str(obj)
                        )
                    except:
                        return str(obj)

        # Convert to regular dict to make it JSON serializable
        if isinstance(results, defaultdict):
            results = dict(results)

        # Preprocess the results to ensure all nested structures are serializable
        def clean_for_json(item):
            if isinstance(item, dict):
                return {k: clean_for_json(v) for k, v in item.items()}
            elif isinstance(item, list):
                return [clean_for_json(i) for i in item]
            elif isinstance(item, (datetime, np.generic)):
                return str(item)
            elif isinstance(item, (int, float, str, bool, type(None))):
                return item
            else:
                return str(item)

        serializable_results = clean_for_json(results)

        with open(file_path, "w") as f:
            json.dump(serializable_results, f, indent=2, cls=CustomJSONEncoder)

        print(f"Results saved to {file_path}")

    def generate_reports(self):
        """Generate summary reports and visualizations of game results"""
        if not self.pattern_results:
            print("No results to report. Run some games first.")
            return

        # Create summary report
        report_file = os.path.join(ROOT_OUTPUT_DIR, "summary_report.txt")
        with open(report_file, "w") as f:
            f.write("Hot or Not Game Simulation Summary Report\n")
            f.write("========================================\n\n")
            f.write(f"Generated on: {datetime.now()}\n\n")

            for pattern_type, results in self.pattern_results.items():
                total_games = results["hot_count"] + results["not_hot_count"]
                if total_games == 0:
                    continue

                win_rate = (
                    results["win_count"] / total_games * 100 if total_games > 0 else 0
                )

                f.write(f"Pattern: {pattern_type.upper()}\n")
                f.write(f"  Total games: {total_games}\n")
                f.write(
                    f"  Hot predictions: {results['hot_count']} ({results['hot_count']/total_games*100:.1f}%)\n"
                )
                f.write(
                    f"  Not hot predictions: {results['not_hot_count']} ({results['not_hot_count']/total_games*100:.1f}%)\n"
                )
                f.write(
                    f"  Correct predictions: {results['win_count']} ({win_rate:.1f}%)\n"
                )
                f.write(
                    f"  Incorrect predictions: {results['loss_count']} ({100-win_rate:.1f}%)\n\n"
                )

        print(f"Summary report generated at: {report_file}")

        # Generate visualizations
        self.generate_visualizations()

    def generate_visualizations(self):
        """Generate visualizations of the game results"""
        if not self.pattern_results:
            return

        # Set the style for plots
        sns.set(style="whitegrid")

        # 1. Combined pattern analysis dashboard (combines hot/not hot and win/loss)
        self._create_pattern_comparison_dashboard()

        # 2. DS Score Distributions in a combined view
        self._create_combined_ds_score_distributions()

    def _create_pattern_comparison_dashboard(self):
        """Create a comprehensive dashboard with all pattern comparisons in one figure"""
        # Prepare data for all three types of plots we want to combine

        # 1. Data for Hot vs Not Hot plot
        hot_not_data = []

        # 2. Data for Win vs Loss plot
        win_loss_data = []

        # 3. Data for combined categories plot
        combined_data = []

        for pattern, results in self.pattern_results.items():
            total = results["hot_count"] + results["not_hot_count"]
            if total == 0:
                continue

            # Hot vs Not Hot data
            hot_pct = (results["hot_count"] / total) * 100
            not_hot_pct = (results["not_hot_count"] / total) * 100

            hot_not_data.append(
                {
                    "Pattern": pattern,
                    "Hot": hot_pct,
                    "Not Hot": not_hot_pct,
                    "Hot_count": results["hot_count"],
                    "Not_Hot_count": results["not_hot_count"],
                    "Total": total,
                }
            )

            # Win vs Loss data
            total_outcomes = results["win_count"] + results["loss_count"]
            if total_outcomes > 0:
                win_pct = (results["win_count"] / total_outcomes) * 100
                loss_pct = (results["loss_count"] / total_outcomes) * 100

                win_loss_data.append(
                    {
                        "Pattern": pattern,
                        "Win": win_pct,
                        "Loss": loss_pct,
                        "Win_count": results["win_count"],
                        "Loss_count": results["loss_count"],
                        "Total": total_outcomes,
                    }
                )

            # Combined category data
            hot_win = sum(
                1
                for r in results["game_results"]
                if r["hot_or_not"] and r["prediction_correct"]
            )
            hot_loss = sum(
                1
                for r in results["game_results"]
                if r["hot_or_not"] and not r["prediction_correct"]
            )
            not_hot_win = sum(
                1
                for r in results["game_results"]
                if not r["hot_or_not"] and r["prediction_correct"]
            )
            not_hot_loss = sum(
                1
                for r in results["game_results"]
                if not r["hot_or_not"] and not r["prediction_correct"]
            )

            # Calculate percentages
            hot_win_pct = (hot_win / total) * 100 if total > 0 else 0
            hot_loss_pct = (hot_loss / total) * 100 if total > 0 else 0
            not_hot_win_pct = (not_hot_win / total) * 100 if total > 0 else 0
            not_hot_loss_pct = (not_hot_loss / total) * 100 if total > 0 else 0

            combined_data.append(
                {
                    "Pattern": pattern,
                    "Hot & Correct": hot_win_pct,
                    "Hot & Incorrect": hot_loss_pct,
                    "Not Hot & Correct": not_hot_win_pct,
                    "Not Hot & Incorrect": not_hot_loss_pct,
                    "Hot_Correct_count": hot_win,
                    "Hot_Incorrect_count": hot_loss,
                    "NotHot_Correct_count": not_hot_win,
                    "NotHot_Incorrect_count": not_hot_loss,
                    "Total": total,
                }
            )

        # Create DataFrames
        hot_not_df = pd.DataFrame(hot_not_data)
        win_loss_df = pd.DataFrame(win_loss_data)
        combined_df = pd.DataFrame(combined_data)

        # Melt DataFrames for plotting
        hot_not_melted = pd.melt(
            hot_not_df,
            id_vars=["Pattern", "Hot_count", "Not_Hot_count", "Total"],
            value_vars=["Hot", "Not Hot"],
            var_name="Prediction",
            value_name="Percentage",
        )

        win_loss_melted = pd.melt(
            win_loss_df,
            id_vars=["Pattern", "Win_count", "Loss_count", "Total"],
            value_vars=["Win", "Loss"],
            var_name="Outcome",
            value_name="Percentage",
        )

        # For combined plot
        category_vars = [
            "Hot & Correct",
            "Hot & Incorrect",
            "Not Hot & Correct",
            "Not Hot & Incorrect",
        ]
        count_cols = [
            "Hot_Correct_count",
            "Hot_Incorrect_count",
            "NotHot_Correct_count",
            "NotHot_Incorrect_count",
        ]
        count_mapping = dict(zip(category_vars, count_cols))

        combined_melted = pd.melt(
            combined_df,
            id_vars=["Pattern", "Total"] + count_cols,
            value_vars=category_vars,
            var_name="Category",
            value_name="Percentage",
        )

        # Create figure with 3 subplots (2 rows, 2 columns)
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # 1. Hot vs Not Hot plot (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = sns.barplot(
            x="Pattern",
            y="Percentage",
            hue="Prediction",
            data=hot_not_melted,
            ax=ax1,
            palette=["#3498db", "#e74c3c"],  # Blue, Red
        )

        # Add percentage and count labels
        for i, container in enumerate(ax1.containers):
            count_col = "Hot_count" if i == 0 else "Not_Hot_count"
            ax1.bar_label(
                container,
                labels=[
                    f"{p:.1f}%\n(n={c})"
                    for p, c in zip(
                        container.datavalues,
                        hot_not_melted[
                            hot_not_melted["Prediction"]
                            == ("Hot" if i == 0 else "Not Hot")
                        ][count_col],
                    )
                ],
                fontsize=10,
            )

        ax1.set_title("Hot vs Not Hot Predictions by Pattern", fontsize=16)
        ax1.set_xlabel("Pattern Type", fontsize=14)
        ax1.set_ylabel("Percentage of Predictions", fontsize=14)
        ax1.set_ylim(0, 100)
        ax1.legend(title="Prediction", loc="best")

        # 2. Win vs Loss plot (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = sns.barplot(
            x="Pattern",
            y="Percentage",
            hue="Outcome",
            data=win_loss_melted,
            ax=ax2,
            palette=["#2ecc71", "#e67e22"],  # Green, Orange
        )

        # Add percentage and count labels
        for i, container in enumerate(ax2.containers):
            count_col = "Win_count" if i == 0 else "Loss_count"
            ax2.bar_label(
                container,
                labels=[
                    f"{p:.1f}%\n(n={c})"
                    for p, c in zip(
                        container.datavalues,
                        win_loss_melted[
                            win_loss_melted["Outcome"] == ("Win" if i == 0 else "Loss")
                        ][count_col],
                    )
                ],
                fontsize=10,
            )

        ax2.set_title("Prediction Accuracy by Pattern", fontsize=16)
        ax2.set_xlabel("Pattern Type", fontsize=14)
        ax2.set_ylabel("Percentage of Games", fontsize=14)
        ax2.set_ylim(0, 100)
        ax2.legend(title="Outcome", loc="best")

        # 3. Combined categories plot (spans bottom row)
        ax3 = fig.add_subplot(gs[1, :])
        bars3 = sns.barplot(
            x="Pattern",
            y="Percentage",
            hue="Category",
            data=combined_melted,
            ax=ax3,
            palette=[
                "#3498db",
                "#9b59b6",
                "#2ecc71",
                "#e74c3c",
            ],  # Blue, Purple, Green, Red
        )

        # Add percentage and count labels
        for i, container in enumerate(ax3.containers):
            category = category_vars[i]
            count_col = count_mapping[category]

            ax3.bar_label(
                container,
                labels=[
                    f"{p:.1f}%\n(n={combined_df.loc[idx, count_col]})"
                    for idx, p in enumerate(container.datavalues)
                ],
                fontsize=10,
            )

        # Fix the x-axis tick labels with proper tick locator
        current_ticks = ax3.get_xticks()
        ax3.set_xticks(current_ticks[: len(combined_df)])
        ax3.set_xticklabels(
            [
                f"{p}\n(n={t})"
                for p, t in zip(combined_df["Pattern"], combined_df["Total"])
            ]
        )

        ax3.set_title(
            "Combined Hot/Not Hot and Prediction Accuracy by Pattern", fontsize=16
        )
        ax3.set_xlabel("Pattern Type", fontsize=14)
        ax3.set_ylabel("Percentage of Games", fontsize=14)
        ax3.set_ylim(0, 100)
        ax3.legend(title="Category", loc="upper right")

        # Add a title for the entire figure
        fig.suptitle("Pattern Analysis Dashboard", fontsize=24, y=0.98)

        # Save the combined figure
        plt.savefig(
            os.path.join(ROOT_OUTPUT_DIR, "pattern_analysis_dashboard.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_combined_ds_score_distributions(self):
        """Create a combined visualization of all DS score distributions in one figure with improved clarity"""
        # Get all patterns that have data
        valid_patterns = []
        for pattern, results in self.pattern_results.items():
            if results["game_results"]:
                valid_patterns.append(pattern)

        if not valid_patterns:
            return

        # Calculate grid dimensions based on number of patterns
        n_patterns = len(valid_patterns)
        n_cols = min(2, n_patterns)  # Max 2 columns
        n_rows = (n_patterns + n_cols - 1) // n_cols  # Ceiling division

        # Create figure with subplots
        fig = plt.figure(figsize=(15 * n_cols, 10 * n_rows))

        # Title for the entire figure
        fig.suptitle("Decision Score Distributions by Pattern", fontsize=24, y=0.98)

        # Process each pattern
        for idx, pattern in enumerate(valid_patterns):
            results = self.pattern_results[pattern]

            # Create subplot
            ax = fig.add_subplot(n_rows, n_cols, idx + 1)

            # Extract data points
            data_points = []
            for r in results["game_results"]:
                current = r["current_avg_ds_score"]
                reference = r["reference_predicted_avg_ds_score"]
                if current is not None and reference is not None:
                    # Determine if it's actually hot (current > reference)
                    actually_hot = current > reference
                    data_points.append(
                        {
                            "Current DS Score": current,
                            "Reference DS Score": reference,
                            "Prediction": "Hot" if r["hot_or_not"] else "Not Hot",
                            "Prediction Correct": (
                                "Correct" if r["prediction_correct"] else "Incorrect"
                            ),
                            "Actually Hot": actually_hot,
                            "Match Type": (
                                "True Positive"
                                if r["hot_or_not"] and actually_hot
                                else (
                                    "False Positive"
                                    if r["hot_or_not"] and not actually_hot
                                    else (
                                        "True Negative"
                                        if not r["hot_or_not"] and not actually_hot
                                        else "False Negative"
                                    )
                                )
                            ),
                        }
                    )

            if not data_points:
                ax.text(
                    0.5, 0.5, "No data available", ha="center", va="center", fontsize=14
                )
                ax.set_title(f"{pattern.capitalize()} Pattern", fontsize=16)
                continue

            # Create DataFrame
            df = pd.DataFrame(data_points)

            # Calculate important statistics
            total_points = len(df)

            # Actual hot/not hot counts (based on Current vs Reference)
            actually_hot_count = sum(df["Actually Hot"])
            actually_not_hot_count = total_points - actually_hot_count

            # Predicted hot/not hot counts
            predicted_hot_count = sum(df["Prediction"] == "Hot")
            predicted_not_hot_count = sum(df["Prediction"] == "Not Hot")

            # Accuracy metrics
            true_positives = sum((df["Prediction"] == "Hot") & df["Actually Hot"])
            false_positives = sum((df["Prediction"] == "Hot") & ~df["Actually Hot"])
            true_negatives = sum((df["Prediction"] == "Not Hot") & ~df["Actually Hot"])
            false_negatives = sum((df["Prediction"] == "Not Hot") & df["Actually Hot"])

            # Calculate percentages
            actually_hot_pct = (
                (actually_hot_count / total_points) * 100 if total_points > 0 else 0
            )
            actually_not_hot_pct = (
                (actually_not_hot_count / total_points) * 100 if total_points > 0 else 0
            )

            overall_accuracy = (
                ((true_positives + true_negatives) / total_points) * 100
                if total_points > 0
                else 0
            )

            # Create scatter plot with improved colors and markers
            scatter = sns.scatterplot(
                x="Reference DS Score",
                y="Current DS Score",
                hue="Match Type",
                style="Match Type",
                s=100,
                palette={
                    "True Positive": "#2ecc71",  # Green
                    "False Positive": "#e74c3c",  # Red
                    "True Negative": "#3498db",  # Blue
                    "False Negative": "#f39c12",  # Orange
                },
                markers={
                    "True Positive": "o",
                    "False Positive": "X",
                    "True Negative": "o",
                    "False Negative": "X",
                },
                data=df,
                ax=ax,
            )

            # Add diagonal line (the decision boundary)
            min_val = (
                min(df["Reference DS Score"].min(), df["Current DS Score"].min()) - 0.1
            )
            max_val = (
                max(df["Reference DS Score"].max(), df["Current DS Score"].max()) + 0.1
            )

            ax.plot(
                [min_val, max_val],
                [min_val, max_val],
                "k--",
                linewidth=2,
                label="Decision Boundary (Current = Reference)",
            )

            # Color zones
            ax.fill_between(
                [min_val, max_val],
                [min_val, max_val],
                [max_val + 1, max_val + 1],
                color="lightgreen",
                alpha=0.1,
                label="Hot Zone (Current > Reference)",
            )
            ax.fill_between(
                [min_val, max_val],
                [min_val, max_val],
                [min_val - 1, min_val - 1],
                color="lightpink",
                alpha=0.1,
                label="Not Hot Zone (Current < Reference)",
            )

            # Add detailed stats annotations
            stats_text = (
                f"ACTUAL HOT: {actually_hot_count} ({actually_hot_pct:.1f}%)\n"
                f"• True Positives: {true_positives}\n"
                f"• False Negatives: {false_negatives}\n\n"
                f"ACTUAL NOT HOT: {actually_not_hot_count} ({actually_not_hot_pct:.1f}%)\n"
                f"• True Negatives: {true_negatives}\n"
                f"• False Positives: {false_positives}\n\n"
                f"Overall Accuracy: {overall_accuracy:.1f}%"
            )

            # Positioning the stats box in a better location based on data distribution
            text_x = min_val + (max_val - min_val) * 0.05  # Left side
            text_y = min_val + (max_val - min_val) * 0.75  # Upper portion

            ax.text(
                text_x,
                text_y,
                stats_text,
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.8),
                verticalalignment="top",
            )

            # Add HOT ZONE and NOT HOT ZONE labels directly on the plot
            ax.annotate(
                "HOT ZONE\n(Current > Reference)",
                xy=(max_val * 0.8, max_val * 0.85),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.5),
                horizontalalignment="center",
            )

            ax.annotate(
                "NOT HOT ZONE\n(Current < Reference)",
                xy=(max_val * 0.8, min_val * 0.5),
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightpink", alpha=0.5),
                horizontalalignment="center",
            )

            # Add title and labels
            ax.set_title(
                f"{pattern.capitalize()} Pattern (n={total_points})", fontsize=16
            )
            ax.set_xlabel(
                "Reference DS Score (Predicted from historical trend)", fontsize=12
            )
            ax.set_ylabel("Current DS Score (Actual recent performance)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)

            # Remove original legend and create a cleaner custom legend
            ax.get_legend().remove()

            # Create a custom legend
            from matplotlib.lines import Line2D

            legend_elements = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#2ecc71",
                    markersize=10,
                    label="True Positive (Correctly predicted Hot)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    markerfacecolor="#e74c3c",
                    markersize=10,
                    label="False Positive (Incorrectly predicted Hot)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="#3498db",
                    markersize=10,
                    label="True Negative (Correctly predicted Not Hot)",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    markerfacecolor="#f39c12",
                    markersize=10,
                    label="False Negative (Incorrectly predicted Not Hot)",
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="k",
                    label="Decision Boundary (Current = Reference)",
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        # Add an explanatory note at the bottom with clearer explanation
        explanation_text = (
            "How to read these charts:\n"
            "• Each point represents one game's score comparison.\n"
            "• The diagonal line is the decision boundary: points above it are actually Hot, below are Not Hot.\n"
            "• Points are colored by whether the prediction matched reality:\n"
            "   - Green circles (○): True Positives - Correctly predicted as Hot\n"
            "   - Red X marks: False Positives - Incorrectly predicted as Hot\n"
            "   - Blue circles (○): True Negatives - Correctly predicted as Not Hot\n"
            "   - Orange X marks: False Negatives - Incorrectly predicted as Not Hot\n"
            "• A video's true Hot/Not Hot status is determined by whether its Current DS Score > Reference DS Score."
        )

        fig.text(
            0.5,
            0.01,
            explanation_text,
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Save the figure
        plt.savefig(
            os.path.join(ROOT_OUTPUT_DIR, "combined_ds_score_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


async def run_simulation_async(num_games=50, concurrency=3):
    """Run the main simulation asynchronously"""
    # Create the simulator
    simulator = HotOrNotGameSimulator()

    # Define pattern configurations for the games
    # Run only on the last 30 minutes of data
    pattern_configs = [
        {
            "pattern_type": "random",
            "video_id": "sim-random",
            "duration_hours": 0.5,
            "interval_minutes": 4,
            "events_per_interval": 10,
        }
    ]
    """
    pattern_configs = [
        {
            "pattern_type": "increase",
            "video_id": "sim-increase",
            "duration_hours": 0.5,  # Run for 30 minutes
            "interval_minutes": 5,
            "events_per_interval": 10,
            "growth_rate": 1.05,
            "linear_increase": True,
            "linear_increment": 1,
        },
        {
            "pattern_type": "decrease",
            "video_id": "sim-decrease",
            "duration_hours": 0.5,
            "interval_minutes": 5,
            "events_per_interval": 10,
            "decay_rate": 0.9,
        },
        {
            "pattern_type": "spike",
            "video_id": "sim-spike",
            "duration_hours": 0.5,
            "interval_minutes": 5,
            "events_per_interval": 10,
            "spike_position": 0.5,
            "spike_magnitude": 3.0,
        },
        {
            "pattern_type": "fluctuate",
            "video_id": "sim-fluctuate",
            "duration_hours": 0.5,
            "interval_minutes": 5,
            "events_per_interval": 10,
            "fluctuation_amplitude": 0.4,
        },
    ]
    """

    # Get the timestamp to start simulation (30 minutes before now)
    now = datetime.now()
    start_time = now - timedelta(minutes=30)

    print(f"Starting simulation at timestamp: {start_time}")
    print(f"Running {num_games} games with {concurrency} concurrent tasks")

    # Run multiple games for each pattern, passing the start timestamp
    results = await simulator.run_multiple_games_async(
        pattern_configs,
        num_games=num_games,
        concurrency=concurrency,
        start_timestamp=start_time,
    )

    # Generate reports and visualizations
    simulator.generate_reports()

    return results


if __name__ == "__main__":

    # Run the simulation asynchronously with more games and higher concurrency
    # since we're using in-memory processing
    asyncio.run(run_simulation_async(num_games=10_000, concurrency=100))
