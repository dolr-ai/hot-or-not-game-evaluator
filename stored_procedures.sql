CREATE OR REPLACE FUNCTION hot_or_not_evaluator.update_counter(
    p_video_id VARCHAR,
    p_liked BOOLEAN,
    p_watch_percentage NUMERIC -- Expecting 0-100
)
RETURNS VOID 
SECURITY DEFINER
AS $$
DECLARE
    -- Time and Input Handling
    v_current_minute TIMESTAMPTZ := date_trunc('minute', NOW());
    v_like_increment INT := CASE WHEN p_liked THEN 1 ELSE 0 END; -- 1 if liked, 0 if not
    v_watch_increment INT := 1; -- Assuming one call per view event

    -- Normalization Constants
    v_like_ctr_center NUMERIC;
    v_like_ctr_range NUMERIC;
    v_watch_perc_center NUMERIC;
    v_watch_perc_range NUMERIC;

    -- Previous Cumulative Values (for the video, before current minute)
    v_prev_cumulative_like_count BIGINT;
    v_prev_cumulative_watch_count BIGINT;
    v_prev_cumulative_like_ctr NUMERIC;
    v_prev_cumulative_avg_watch_perc NUMERIC;

    -- Current Minute Aggregated Values (after UPSERT)
    v_current_like_count_mnt BIGINT;
    v_current_watch_count_mnt BIGINT;
    v_current_avg_perc_watched_mnt NUMERIC;

    -- Calculated Derived Metrics for Current Minute Row
    v_like_ctr_mnt NUMERIC;
    v_cumulative_like_count BIGINT;
    v_cumulative_watch_count BIGINT;
    v_cumulative_like_ctr NUMERIC;
    v_cumulative_avg_watch_perc NUMERIC;
    v_norm_like_ctr NUMERIC;
    v_norm_watch_perc NUMERIC;
    v_harmonic_mean NUMERIC;
    v_ds_score NUMERIC;
BEGIN
    -- Step 0: Get normalization constants
    -- Consider caching these if they rarely change and performance is critical
    SELECT like_ctr_center, like_ctr_range, watch_percentage_center, watch_percentage_range
    INTO v_like_ctr_center, v_like_ctr_range, v_watch_perc_center, v_watch_perc_range
    FROM hot_or_not_evaluator.metric_const WHERE id = 1;

    -- Step 1: UPSERT base metrics for the current minute
    -- This handles concurrent updates to the same video/minute atomically.
    INSERT INTO hot_or_not_evaluator.video_engagement_relation (
        video_id, timestamp_mnt, like_count_mnt,
        average_percentage_watched_mnt, watch_count_mnt
    )
    VALUES (
        p_video_id, v_current_minute, v_like_increment,
        p_watch_percentage, v_watch_increment
    )
    ON CONFLICT (video_id, timestamp_mnt) DO UPDATE SET
        like_count_mnt = hot_or_not_evaluator.video_engagement_relation.like_count_mnt + EXCLUDED.like_count_mnt,
        -- Calculate running average watch percentage for the minute safely
        average_percentage_watched_mnt =
            CASE
                WHEN (hot_or_not_evaluator.video_engagement_relation.watch_count_mnt + EXCLUDED.watch_count_mnt) > 0 THEN
                    ((hot_or_not_evaluator.video_engagement_relation.average_percentage_watched_mnt * hot_or_not_evaluator.video_engagement_relation.watch_count_mnt) + (EXCLUDED.average_percentage_watched_mnt * EXCLUDED.watch_count_mnt))
                    / (hot_or_not_evaluator.video_engagement_relation.watch_count_mnt + EXCLUDED.watch_count_mnt)
                ELSE 0 -- Avoid division by zero if counts somehow become zero during update; shouldn't happen ever
            END,
        watch_count_mnt = hot_or_not_evaluator.video_engagement_relation.watch_count_mnt + EXCLUDED.watch_count_mnt;

    -- Step 2: Get the latest cumulative values for this video *before* the current minute
    SELECT
        cumulative_like_count, cumulative_watch_count,
        cumulative_like_ctr, cumulative_average_percentage_watched
    INTO
        v_prev_cumulative_like_count, v_prev_cumulative_watch_count,
        v_prev_cumulative_like_ctr, v_prev_cumulative_avg_watch_perc
    FROM hot_or_not_evaluator.video_engagement_relation
    WHERE video_id = p_video_id AND timestamp_mnt < v_current_minute
    ORDER BY timestamp_mnt DESC
    LIMIT 1;

    -- Handle the case where this is the first record ever for the video
    v_prev_cumulative_like_count := COALESCE(v_prev_cumulative_like_count, 0);
    v_prev_cumulative_watch_count := COALESCE(v_prev_cumulative_watch_count, 0);
    v_prev_cumulative_like_ctr := COALESCE(v_prev_cumulative_like_ctr, 0);
    v_prev_cumulative_avg_watch_perc := COALESCE(v_prev_cumulative_avg_watch_perc, 0);

    -- Step 3: Get the fully updated aggregate values for the current minute (after the UPSERT)
    -- Re-select the row to ensure we have the latest aggregates for calculations.
    SELECT
        like_count_mnt, watch_count_mnt, average_percentage_watched_mnt
    INTO
        v_current_like_count_mnt, v_current_watch_count_mnt, v_current_avg_perc_watched_mnt
    FROM hot_or_not_evaluator.video_engagement_relation
    WHERE video_id = p_video_id AND timestamp_mnt = v_current_minute
    FOR UPDATE; -- Lock the row to prevent concurrent updates between read and subsequent UPDATE

    -- Step 4: Calculate all derived metrics based on current minute and previous cumulative values

    -- like_ctr_mnt: Likes per watch for this specific minute
    v_like_ctr_mnt := CASE
                          WHEN v_current_watch_count_mnt > 0 THEN
                              (v_current_like_count_mnt * 100.0) / v_current_watch_count_mnt
                          ELSE 0
                      END;

    -- ctr has a range of 0-100 in percentages 
    -- cumulative counts: Add current minute's totals to previous cumulative totals
    -- Note: Assumes the UPSERT correctly aggregated the *minute's* counts.
    v_cumulative_like_count := v_prev_cumulative_like_count + v_current_like_count_mnt;
    v_cumulative_watch_count := v_prev_cumulative_watch_count + v_current_watch_count_mnt;

    -- cumulative averages (weighted by watch counts)
    v_cumulative_like_ctr := CASE
                                 WHEN v_cumulative_watch_count > 0 THEN
                                     ((v_prev_cumulative_like_ctr * v_prev_cumulative_watch_count) + (v_like_ctr_mnt * v_current_watch_count_mnt)) / v_cumulative_watch_count
                                 ELSE 0
                             END;
    v_cumulative_avg_watch_perc := CASE
                                       WHEN v_cumulative_watch_count > 0 THEN
                                           ((v_prev_cumulative_avg_watch_perc * v_prev_cumulative_watch_count) + (v_current_avg_perc_watched_mnt * v_current_watch_count_mnt)) / v_cumulative_watch_count
                                       ELSE 0
                                   END;

    -- normalized cumulative values (handle division by zero if range is 0)
    v_norm_like_ctr := CASE
                           WHEN v_like_ctr_range != 0 THEN
                               GREATEST(0, (v_cumulative_like_ctr - v_like_ctr_center) / v_like_ctr_range)
                           ELSE 0 -- Or NULL, depending on desired behavior
                       END;
    v_norm_watch_perc := CASE
                             WHEN v_watch_perc_range != 0 THEN
                                 GREATEST(0, (v_cumulative_avg_watch_perc - v_watch_perc_center) / v_watch_perc_range)
                             ELSE 0 -- Or NULL
                         END;

    -- harmonic mean (using the +1/+2 adjustment from the formula)
    v_harmonic_mean := CASE
                           WHEN (v_norm_like_ctr + v_norm_watch_perc + 2) != 0 THEN
                               ((v_norm_like_ctr + 1) * (v_norm_watch_perc + 1)) / (v_norm_like_ctr + v_norm_watch_perc + 2)
                           ELSE 1 -- Or NULL, if denominator is zero
                       END;

    -- ds_score (harmonic mean - 1)
    v_ds_score := v_harmonic_mean - 1;

    -- Step 5: Update the current minute's row with all the calculated derived metrics
    UPDATE hot_or_not_evaluator.video_engagement_relation
    SET like_ctr_mnt = v_like_ctr_mnt,
        cumulative_like_count = v_cumulative_like_count,
        cumulative_watch_count = v_cumulative_watch_count,
        cumulative_like_ctr = v_cumulative_like_ctr,
        cumulative_average_percentage_watched = v_cumulative_avg_watch_perc,
        normalized_cumulative_like_ctr = v_norm_like_ctr,
        normalized_cumulative_watch_percentage = v_norm_watch_perc,
        harmonic_mean_of_like_count_and_watch_count = v_harmonic_mean,
        ds_score = v_ds_score
    WHERE video_id = p_video_id AND timestamp_mnt = v_current_minute;

END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hot_or_not_evaluator.update_counter(VARCHAR, BOOLEAN, NUMERIC) IS
'Updates video engagement metrics for a given video based on a single engagement event (like/watch).
It aggregates metrics per minute and calculates cumulative and derived scores including the ds_score.
Parameters: p_video_id, p_liked (TRUE if liked), p_watch_percentage (0-100).';

-- End of update_counter function
-- Phase 2: Task 2.2 - Implement compute_hot_or_not Stored Procedure/Function

CREATE OR REPLACE FUNCTION hot_or_not_evaluator.compute_hot_or_not()
RETURNS VOID AS $$
DECLARE
    -- Time boundaries
    v_now TIMESTAMPTZ := NOW();
    v_5_mins_ago TIMESTAMPTZ := v_now - INTERVAL '5 minutes';
    v_1_day_ago TIMESTAMPTZ := v_now - INTERVAL '1 day';

    -- Video iter item
    v_video_record RECORD;

    -- Calculated values per video
    v_current_avg_ds NUMERIC;
    v_ref_slope NUMERIC;
    v_ref_intercept NUMERIC;
    v_ref_count BIGINT; -- To check if enough data for regression
    v_ref_predicted_avg_ds NUMERIC;
    v_is_hot BOOLEAN;
    v_previous_hot_status BOOLEAN;
BEGIN
    RAISE NOTICE 'Starting compute_hot_or_not at %', v_now;

    -- This avoids processing inactive videos unnecessarily.
    FOR v_video_record IN
        SELECT DISTINCT video_id
        FROM hot_or_not_evaluator.video_engagement_relation
        WHERE timestamp_mnt >= v_1_day_ago AND timestamp_mnt < v_now
    LOOP
        v_current_avg_ds := NULL;
        v_ref_slope := NULL;
        v_ref_intercept := NULL;
        v_ref_count := 0;
        v_ref_predicted_avg_ds := NULL;
        v_is_hot := NULL; -- we don't know if the video is hot or not yet

        BEGIN
            -- Get the previous hot status to maintain if comparison can't be made
            SELECT hot_or_not
            INTO v_previous_hot_status
            FROM hot_or_not_evaluator.video_hot_or_not_status
            WHERE video_id = v_video_record.video_id;
            
            -- Handle the case where this is the first record for the video
            -- Default to a random boolean value (TRUE/FALSE) for new videos instead of NULL
            v_previous_hot_status := COALESCE(v_previous_hot_status, (random() > 0.5));

            -- Calculate current average ds_score (last 5 minutes)
            SELECT AVG(ds_score)
            INTO v_current_avg_ds
            FROM hot_or_not_evaluator.video_engagement_relation
            WHERE video_id = v_video_record.video_id
            -- we do < v_now and not <= v_now because we want to avoid computing the average for the partially aggregated minute data
              AND timestamp_mnt >= v_5_mins_ago AND timestamp_mnt < v_now;


            -- Calculate OLS parameters for the reference period (1 day ago to 5 mins ago)
            SELECT
                regr_slope(ds_score, EXTRACT(EPOCH FROM timestamp_mnt)),
                regr_intercept(ds_score, EXTRACT(EPOCH FROM timestamp_mnt)),
                regr_count(ds_score, EXTRACT(EPOCH FROM timestamp_mnt))
            INTO v_ref_slope, v_ref_intercept, v_ref_count
            FROM hot_or_not_evaluator.video_engagement_relation
            WHERE video_id = v_video_record.video_id
              AND timestamp_mnt >= v_1_day_ago AND timestamp_mnt < v_5_mins_ago;

            -- Calculate predicted value
            IF v_ref_count >= 2 AND v_ref_slope IS NOT NULL AND v_ref_intercept IS NOT NULL THEN
                -- using the OLS line derived from the reference period.
                -- Instead of averaging over the range, just use the midpoint (as for a linear function, the average over a symmetric interval equals the value at the midpoint)
                SELECT v_ref_slope * EXTRACT(EPOCH FROM date_trunc('minute', v_now + (v_now - v_5_mins_ago)/2)) + v_ref_intercept
                INTO v_ref_predicted_avg_ds;
            END IF;

            -- Determine if video is hot
            IF v_current_avg_ds IS NOT NULL AND v_ref_predicted_avg_ds IS NOT NULL THEN
                v_is_hot := v_current_avg_ds > v_ref_predicted_avg_ds;
            ELSE
                v_is_hot := v_previous_hot_status; -- Maintain previous status if comparison cannot be made
            END IF;

            -- Update status table (using UPSERT for the latest status)
            INSERT INTO hot_or_not_evaluator.video_hot_or_not_status (
                video_id, last_updated_mnt, hot_or_not,
                current_avg_ds_score, reference_predicted_avg_ds_score
            )
            VALUES (
                v_video_record.video_id, v_now, v_is_hot,
                v_current_avg_ds, v_ref_predicted_avg_ds
            )
            ON CONFLICT (video_id) DO UPDATE SET
                last_updated_mnt = EXCLUDED.last_updated_mnt,
                hot_or_not = EXCLUDED.hot_or_not,
                current_avg_ds_score = EXCLUDED.current_avg_ds_score,
                reference_predicted_avg_ds_score = EXCLUDED.reference_predicted_avg_ds_score;

        EXCEPTION
            WHEN OTHERS THEN
                RAISE WARNING 'Error processing video % in compute_hot_or_not: %', v_video_record.video_id, SQLERRM;
                CONTINUE; -- Move to the next video
        END;

    END LOOP;

    RAISE NOTICE 'Finished compute_hot_or_not at %', clock_timestamp();

END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hot_or_not_evaluator.compute_hot_or_not() IS
'Calculates the "Hot or Not" status for all recently active videos.
It compares the average ds_score of the last 5 minutes against a predicted score
derived from an OLS linear regression of the ds_score over the preceding period (1 day - 5 mins).
Updates the video_hot_or_not_status table. Intended to be run periodically (e.g., every 5 minutes).';



CREATE OR REPLACE FUNCTION hot_or_not_evaluator.get_hot_or_not(p_video_id VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    v_status BOOLEAN;
    v_now TIMESTAMPTZ := NOW();
BEGIN
    SELECT hot_or_not
    INTO v_status
    FROM hot_or_not_evaluator.video_hot_or_not_status
    WHERE video_id = p_video_id;
 
    -- If status is NULL, assign a random boolean value and persist it
    IF v_status IS NULL THEN
        v_status := (random() > 0.5);
        
        -- Insert the random status into the database to ensure consistency
        INSERT INTO hot_or_not_evaluator.video_hot_or_not_status (
            video_id, last_updated_mnt, hot_or_not,
            current_avg_ds_score, reference_predicted_avg_ds_score
        )
        VALUES (
            p_video_id, v_now, v_status,
            NULL, NULL
        )
        ON CONFLICT (video_id) DO UPDATE SET
            last_updated_mnt = EXCLUDED.last_updated_mnt,
            hot_or_not = EXCLUDED.hot_or_not;
    END IF;

    RETURN v_status;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hot_or_not_evaluator.get_hot_or_not(VARCHAR) IS
'Retrieves the latest calculated "Hot or Not" status (TRUE for Hot, FALSE for Not) for a specific video ID.
If the video has no status entry, generates a random boolean value, persists it to the database, and returns it.';
