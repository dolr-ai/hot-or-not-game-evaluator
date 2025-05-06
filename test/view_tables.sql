-- check the metric_const table
select * from hot_or_not_evaluator.metric_const;

-- check the video_engagement_relation table
select * from hot_or_not_evaluator.video_engagement_relation;

-- check the video_hot_or_not_status table
select * from hot_or_not_evaluator.video_hot_or_not_status;

-- compute_hot_or_not calculates the "Hot or Not" status for all recently active videos.
-- It compares the average ds_score of the last 5 minutes against a predicted score
-- derived from an OLS linear regression of the ds_score over the preceding period (1 day - 5 mins).
-- It then updates the video_hot_or_not_status table.
-- It is intended to be run periodically (e.g., every 5 minutes).
SELECT hot_or_not_evaluator.compute_hot_or_not();

-- test_linear_decrease_like_and_watch
-- test_linear_increase_like_and_watch
SELECT hot_or_not_evaluator.get_hot_or_not('test_linear_decrease_like_and_watch');