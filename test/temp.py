# %%
from activity_generator import ActivityGenerator, BackFill, clean_database_static
from datetime import datetime, timedelta

# %%
base_timestamp = datetime(2025, 4, 29, 14, 0, 0)

clean_database_static(test_video_id_prefix="sgx-", end_time=base_timestamp)
backfill_increase = BackFill(pattern_type="increase")
# %%
data_increase = backfill_increase.backfill_data(
    video_id="sgx-test_video_increase",
    end_time=base_timestamp,
    period=timedelta(days=1),
    interval_minutes=5,
    events_per_interval=10,
    pattern_kwargs={
        "pattern_type": "increase",
        "growth_rate": 1.08,
        "max_events_per_interval": 300,  # Add cap to prevent excessive growth
    },
)
