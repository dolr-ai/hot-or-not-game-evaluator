-- Schema definition for the Hot or Not Video Evaluator
-- Target Database: AlloyDB (PostgreSQL-compatible)

-- Phase 1: Task 1.4 - Define metric_const Table Schema & Populate

-- Creating schema hot_or_not_evaluator.*
CREATE SCHEMA IF NOT EXISTS hot_or_not_evaluator;


-- Dropping the previous metric_const table if it exists

-- Creating the metric_const table in the hot_or_not_evaluator schema
CREATE TABLE IF NOT EXISTS hot_or_not_evaluator.metric_const (
    id INT PRIMARY KEY DEFAULT 1, -- Ensure only one row
    like_ctr_center NUMERIC NOT NULL,
    like_ctr_range NUMERIC NOT NULL,
    watch_percentage_center NUMERIC NOT NULL,
    watch_percentage_range NUMERIC NOT NULL,
    CONSTRAINT single_row CHECK (id = 1)
);

COMMENT ON TABLE hot_or_not_evaluator.metric_const IS 'Stores global normalization constants for metrics.';
COMMENT ON COLUMN hot_or_not_evaluator.metric_const.like_ctr_center IS 'Center value for normalizing cumulative like CTR.';
COMMENT ON COLUMN hot_or_not_evaluator.metric_const.like_ctr_range IS 'Range value for normalizing cumulative like CTR.';
COMMENT ON COLUMN hot_or_not_evaluator.metric_const.watch_percentage_center IS 'Center value for normalizing cumulative average watch percentage.';
COMMENT ON COLUMN hot_or_not_evaluator.metric_const.watch_percentage_range IS 'Range value for normalizing cumulative average watch percentage.';

INSERT INTO hot_or_not_evaluator.metric_const (like_ctr_center, like_ctr_range, watch_percentage_center, watch_percentage_range)
VALUES (0, 0.05, 0, 0.9); -- to update this later in order to upsert instead of insert

CREATE TABLE IF NOT EXISTS hot_or_not_evaluator.video_engagement_relation (
    video_id VARCHAR(255) NOT NULL, -- Or appropriate type based on actual video IDs
    timestamp_mnt TIMESTAMPTZ NOT NULL,
    like_count_mnt BIGINT DEFAULT 0 NOT NULL,
    average_percentage_watched_mnt FLOAT DEFAULT 0.00 NOT NULL, -- Assuming percentage 0.00 to 100.00
    watch_count_mnt BIGINT DEFAULT 0 NOT NULL,
    like_ctr_mnt FLOAT DEFAULT 0.00 NOT NULL,
    cumulative_like_count BIGINT DEFAULT 0 NOT NULL,
    cumulative_watch_count BIGINT DEFAULT 0 NOT NULL,
    cumulative_like_ctr FLOAT DEFAULT 0.00 NOT NULL,
    cumulative_average_percentage_watched FLOAT DEFAULT 0.00 NOT NULL,
    normalized_cumulative_like_ctr FLOAT,
    normalized_cumulative_watch_percentage FLOAT,
    harmonic_mean_of_like_count_and_watch_count FLOAT,
    ds_score FLOAT,
    PRIMARY KEY (video_id, timestamp_mnt)
);

COMMENT ON TABLE hot_or_not_evaluator.video_engagement_relation IS 'Stores raw and calculated engagement metrics per video per minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.video_id IS 'Identifier for the video.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.timestamp_mnt IS 'Timestamp truncated to the minute for aggregation.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.like_count_mnt IS 'Total likes received within this specific minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.average_percentage_watched_mnt IS 'Average watch percentage of views within this specific minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.watch_count_mnt IS 'Total views (watches) within this specific minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.like_ctr_mnt IS 'Like CTR calculated for this specific minute (like_count_mnt * 100 / watch_count_mnt).';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.cumulative_like_count IS 'Total likes for the video up to and including this minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.cumulative_watch_count IS 'Total views for the video up to and including this minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.cumulative_like_ctr IS 'Cumulative Like CTR for the video up to this minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.cumulative_average_percentage_watched IS 'Cumulative average watch percentage for the video up to this minute.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.normalized_cumulative_like_ctr IS 'Cumulative like CTR normalized using metric_const.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.normalized_cumulative_watch_percentage IS 'Cumulative average watch percentage normalized using metric_const.';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.harmonic_mean_of_like_count_and_watch_count IS 'Harmonic mean calculated from normalized values (+1 adjustment).';
COMMENT ON COLUMN hot_or_not_evaluator.video_engagement_relation.ds_score IS 'Final score derived from harmonic mean (-1 adjustment).';

CREATE INDEX idx_ver_video_id ON hot_or_not_evaluator.video_engagement_relation(video_id);
CREATE INDEX idx_ver_timestamp_mnt ON hot_or_not_evaluator.video_engagement_relation(timestamp_mnt);

-- Phase 1: Task 1.2 - Define video_hot_or_not_status Table Schema (Latest Status Version)
CREATE TABLE hot_or_not_evaluator.video_hot_or_not_status (
    video_id VARCHAR(255) PRIMARY KEY, -- Assuming only latest status needed per video
    last_updated_mnt TIMESTAMPTZ NOT NULL,
    hot_or_not BOOLEAN,
    reference_range INTERVAL NOT NULL DEFAULT '1 day',
    current_window_range INTERVAL NOT NULL DEFAULT '5 minutes',
    current_avg_ds_score NUMERIC(10, 5), -- Store scores for context/debugging
    reference_predicted_avg_ds_score NUMERIC(10, 5) -- Store scores for context/debugging
);

COMMENT ON TABLE hot_or_not_evaluator.video_hot_or_not_status IS 'Stores the latest calculated Hot or Not status for each video.';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.video_id IS 'Identifier for the video.';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.last_updated_mnt IS 'Timestamp when the status was last calculated.';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.hot_or_not IS 'The classification result: TRUE for Hot, FALSE for Not.';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.reference_range IS 'The reference time window used for the calculation (e.g., 1 day).';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.current_window_range IS 'The current time window used for the calculation (e.g., 5 minutes).';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.current_avg_ds_score IS 'The average ds_score calculated for the current window.';
COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.reference_predicted_avg_ds_score IS 'The predicted average ds_score based on the reference window OLS extrapolation.';

CREATE INDEX idx_vhons_last_updated ON hot_or_not_evaluator.video_hot_or_not_status(last_updated_mnt);


-- End of schema definition