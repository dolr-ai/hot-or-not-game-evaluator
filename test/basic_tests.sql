/*
TEST SUMMARY:

1. Watch percentage clipping: To be checked.
    - Extremely high watch percentage: Partially Failed. Should clip to 100.
    - Negative watch percentage: Partially failed. Should not accept negative values.

2. Null values:
    - Minimum entries for hot or not computation: To be checked.
    - handling Null video_id input: Passed.
    - handling Null LIKE input: Passed.
    - handling Null watch_percentage input: Passed.
    - Missing reference data: Returns null. Confirm expected behavior.
*/

/* delete rows with sgx- prefix from all video ids */
DELETE FROM hot_or_not_evaluator.video_engagement_relation
WHERE video_id LIKE 'sgx-%';


/* Basic Test Cases */

-- 1. Insert a new video engagement event (passed)
SELECT hot_or_not_evaluator.update_counter('sgx-video_1', TRUE, 80);

-- Check if the record was created (passed)
SELECT * FROM hot_or_not_evaluator.video_engagement_relation
WHERE video_id = 'sgx-video_1'
ORDER BY timestamp_mnt DESC
LIMIT 1;

-- 2. Record multiple engagements for the same video (passed)
SELECT hot_or_not_evaluator.update_counter('sgx-video_2', TRUE, 90);
SELECT hot_or_not_evaluator.update_counter('sgx-video_2', TRUE, 95);
SELECT hot_or_not_evaluator.update_counter('sgx-video_2', FALSE, 30);

-- Verify aggregation (passed)
SELECT * FROM hot_or_not_evaluator.video_engagement_relation
WHERE video_id = 'sgx-video_2'
ORDER BY timestamp_mnt DESC
LIMIT 1;

-- 3. Compute hot or not status (passed)
SELECT hot_or_not_evaluator.compute_hot_or_not();

-- Check results (passed)
SELECT * FROM hot_or_not_evaluator.video_hot_or_not_status;

-- 4. Check a specific video's status (failed)
-- returned null values not enough data to compare
-- need to gracefully handle this case
SELECT hot_or_not_evaluator.get_hot_or_not('sgx-video_1');

/* Edge Cases */

-- 1. Zero Watch Count (passed)
SELECT hot_or_not_evaluator.update_counter('sgx-edge_video_1', TRUE, 0);

-- Check division by zero handling (passed)
SELECT * FROM hot_or_not_evaluator.video_engagement_relation
WHERE video_id = 'sgx-edge_video_1';

-- 2. Extremely High Values (failed)
-- should not accept values greater than 100 or should clip them to 100
SELECT hot_or_not_evaluator.update_counter('sgx-edge_video_2', TRUE, 999);

-- negative values (partially failed)
-- should not accept negative values
SELECT hot_or_not_evaluator.update_counter('sgx-edge_video_2', TRUE, -100);

-- Normal values after extreme values (passed)
SELECT hot_or_not_evaluator.update_counter('sgx-edge_video_2', TRUE, 50);

-- null values after extreme values (passed)
-- does not allow null values
SELECT hot_or_not_evaluator.update_counter('sgx-edge_video_2', TRUE, NULL);

-- Check normalization (partially failed)
-- extreme values skew the result which are out of range 0-100
    SELECT * FROM hot_or_not_evaluator.video_engagement_relation
WHERE video_id = 'sgx-edge_video_2'
ORDER BY timestamp_mnt;

-- 3. Time-Based Edge Cases (passed)
INSERT INTO hot_or_not_evaluator.video_engagement_relation
(video_id, timestamp_mnt, like_count_mnt, average_percentage_watched_mnt, watch_count_mnt, ds_score)
VALUES
('sgx-time_video', NOW() - INTERVAL '1 day', 10, 50, 100, 0.5),
('sgx-time_video', NOW() - INTERVAL '12 hours', 20, 60, 200, 0.6),
('sgx-time_video', NOW() - INTERVAL '6 hours', 30, 70, 300, 0.7);

-- Add current data (passed)
SELECT hot_or_not_evaluator.update_counter('sgx-time_video', TRUE, 90);

-- Run computation and check (passed)
SELECT hot_or_not_evaluator.compute_hot_or_not();
SELECT * FROM hot_or_not_evaluator.video_hot_or_not_status
WHERE video_id = 'sgx-time_video';

-- 4. Missing Reference Data (doubt)
-- returns null value when reference data is missing
-- is this the expected behavior?
-- i am assuming that it is because it is neither hot nor not hot status of this video would be unknown hence null
-- check my assumption
SELECT hot_or_not_evaluator.update_counter('sgx-new_video', TRUE, 95);

-- Check handling of missing history
SELECT hot_or_not_evaluator.compute_hot_or_not();
SELECT * FROM hot_or_not_evaluator.video_hot_or_not_status
WHERE video_id = 'sgx-new_video';


-- 5. Invalid Inputs
-- video_id should not accept null values (passed)
SELECT hot_or_not_evaluator.update_counter(NULL, TRUE, 80);

-- LIKE should not accept null values should be either true or false (passed)
-- gracefully handles null values and updates counts to 0
SELECT hot_or_not_evaluator.update_counter('sgx-null_test', NULL, 80);

-- watch percentage should not accept null values (passed)
SELECT hot_or_not_evaluator.update_counter('sgx-null_test', TRUE, NULL);

--

-- TEST_PATTERNS = [
--     VideoPattern("test_steady_high", steady_high, "Steady High Engagement"),
--     VideoPattern("test_steady_low", steady_low, "Steady Low Engagement"),
--     VideoPattern("test_rising", rising, "Rising Engagement"),
--     VideoPattern("test_falling", falling, "Falling Engagement"),
--     VideoPattern("test_spike", spike_middle, "Spike in Middle"),
--     VideoPattern("test_dip", dip_middle, "Dip in Middle"),
-- ]
DELETE FROM hot_or_not_evaluator.video_hot_or_not_status
WHERE video_id LIKE 'sgx-%';

DELETE FROM hot_or_not_evaluator.video_engagement_relation
WHERE video_id LIKE 'sgx-%'
-- AND timestamp_mnt <= '2025-04-29 16:30:00';



select * from hot_or_not_evaluator.video_engagement_relation
where video_id LIKE 'sgx-%'
-- order by timestamp_mnt desc
-- order by video_id, timestamp_mnt desc
-- limit 10;

select * from hot_or_not_evaluator.video_hot_or_not_status
where video_id LIKE 'sgx-%'
-- limit 10;



select * from hot_or_not_evaluator.video_hot_or_not_status
where video_id LIKE 'sgx-%'

select count(*) from hot_or_not_evaluator.video_engagement_relation
where video_id LIKE 'sgx-%'