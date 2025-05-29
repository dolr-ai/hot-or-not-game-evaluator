-- Migration: Add Hot or Not Windows Support
-- Date: 2025-05-28
-- Description: Adds support for multiple 5-minute window hot_or_not calculations
-- This migration adds 3 new boolean columns and 3 new numeric columns to track
-- hot_or_not status and average ds_scores for historical 5-minute windows.
-- this migration adds a new function to get the hot-or-not status for a video
-- Also adds avg_reference_predicted_ds_score column to metric_const table


BEGIN;

ALTER TABLE hot_or_not_evaluator.video_hot_or_not_status 
ADD COLUMN hot_or_not_5_to_10_mins_ago BOOLEAN DEFAULT NULL;

ALTER TABLE hot_or_not_evaluator.video_hot_or_not_status 
ADD COLUMN hot_or_not_10_to_15_mins_ago BOOLEAN DEFAULT NULL;

ALTER TABLE hot_or_not_evaluator.video_hot_or_not_status 
ADD COLUMN hot_or_not_15_to_20_mins_ago BOOLEAN DEFAULT NULL;

ALTER TABLE hot_or_not_evaluator.video_hot_or_not_status 
ADD COLUMN avg_ds_score_5_to_10_mins_ago NUMERIC(10, 5) DEFAULT NULL;

ALTER TABLE hot_or_not_evaluator.video_hot_or_not_status 
ADD COLUMN avg_ds_score_10_to_15_mins_ago NUMERIC(10, 5) DEFAULT NULL;

ALTER TABLE hot_or_not_evaluator.video_hot_or_not_status 
ADD COLUMN avg_ds_score_15_to_20_mins_ago NUMERIC(10, 5) DEFAULT NULL;

ALTER TABLE hot_or_not_evaluator.metric_const 
ADD COLUMN avg_reference_predicted_ds_score NUMERIC DEFAULT NULL;

-- Add columns for efficient incremental updates
ALTER TABLE hot_or_not_evaluator.metric_const 
ADD COLUMN avg_reference_predicted_ds_score_count BIGINT DEFAULT 0;

ALTER TABLE hot_or_not_evaluator.metric_const 
ADD COLUMN avg_reference_predicted_ds_score_last_updated TIMESTAMPTZ DEFAULT NULL;

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.hot_or_not IS 
'The classification result for current window (0 to -5 mins): TRUE for Hot, FALSE for Not.';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.current_avg_ds_score IS 
'The average ds_score calculated for the current window (0 to -5 minutes).';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.hot_or_not_5_to_10_mins_ago IS 
'The classification result for -5 to -10 minutes window: TRUE for Hot, FALSE for Not.';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.hot_or_not_10_to_15_mins_ago IS 
'The classification result for -10 to -15 minutes window: TRUE for Hot, FALSE for Not.';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.hot_or_not_15_to_20_mins_ago IS 
'The classification result for -15 to -20 minutes window: TRUE for Hot, FALSE for Not.';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.avg_ds_score_5_to_10_mins_ago IS 
'The average ds_score for the -5 to -10 minutes window.';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.avg_ds_score_10_to_15_mins_ago IS 
'The average ds_score for the -10 to -15 minutes window.';

COMMENT ON COLUMN hot_or_not_evaluator.video_hot_or_not_status.avg_ds_score_15_to_20_mins_ago IS 
'The average ds_score for the -15 to -20 minutes window.';

COMMENT ON COLUMN hot_or_not_evaluator.metric_const.avg_reference_predicted_ds_score IS 
'Average of reference_predicted_avg_ds_score across all videos, updated periodically.';

COMMENT ON COLUMN hot_or_not_evaluator.metric_const.avg_reference_predicted_ds_score_count IS 
'Count of videos used in the average calculation for incremental updates.';

COMMENT ON COLUMN hot_or_not_evaluator.metric_const.avg_reference_predicted_ds_score_last_updated IS 
'Timestamp of the last update to the average reference predicted ds score.';

COMMENT ON TABLE hot_or_not_evaluator.video_hot_or_not_status IS 
'Stores the latest calculated Hot or Not status for each video with historical 5-minute window scores.';

COMMIT;
