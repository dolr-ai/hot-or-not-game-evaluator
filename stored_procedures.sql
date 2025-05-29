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
    v_10_mins_ago TIMESTAMPTZ := v_now - INTERVAL '10 minutes';
    v_15_mins_ago TIMESTAMPTZ := v_now - INTERVAL '15 minutes';
    v_20_mins_ago TIMESTAMPTZ := v_now - INTERVAL '20 minutes';
    v_1_day_ago TIMESTAMPTZ := v_now - INTERVAL '1 day';

    -- Video iter item
    v_video_record RECORD;

    -- Calculated values per video
    v_current_avg_ds NUMERIC;
    v_avg_ds_5_to_10_mins NUMERIC;
    v_avg_ds_10_to_15_mins NUMERIC;
    v_avg_ds_15_to_20_mins NUMERIC;
    v_ref_slope NUMERIC;
    v_ref_intercept NUMERIC;
    v_ref_count BIGINT; -- To check if enough data for regression
    v_ref_predicted_avg_ds NUMERIC;
    v_is_hot BOOLEAN;
    v_is_hot_5_to_10_mins BOOLEAN;
    v_is_hot_10_to_15_mins BOOLEAN;
    v_is_hot_15_to_20_mins BOOLEAN;
    v_previous_hot_status BOOLEAN;
    v_previous_hot_status_5_to_10 BOOLEAN;
    v_previous_hot_status_10_to_15 BOOLEAN;
    v_previous_hot_status_15_to_20 BOOLEAN;
BEGIN
    RAISE NOTICE 'Starting compute_hot_or_not at %', v_now;

    -- This avoids processing inactive videos unnecessarily.
    FOR v_video_record IN
        SELECT DISTINCT video_id
        FROM hot_or_not_evaluator.video_engagement_relation
        WHERE timestamp_mnt >= v_1_day_ago AND timestamp_mnt < v_now
    LOOP
        v_current_avg_ds := NULL;
        v_avg_ds_5_to_10_mins := NULL;
        v_avg_ds_10_to_15_mins := NULL;
        v_avg_ds_15_to_20_mins := NULL;
        v_ref_slope := NULL;
        v_ref_intercept := NULL;
        v_ref_count := 0;
        v_ref_predicted_avg_ds := NULL;
        v_is_hot := NULL; -- we don't know if the video is hot or not yet
        v_is_hot_5_to_10_mins := NULL;
        v_is_hot_10_to_15_mins := NULL;
        v_is_hot_15_to_20_mins := NULL;

        BEGIN
            -- Get the previous hot status to maintain if comparison can't be made
            SELECT hot_or_not, hot_or_not_5_to_10_mins_ago, hot_or_not_10_to_15_mins_ago, hot_or_not_15_to_20_mins_ago
            INTO v_previous_hot_status, v_previous_hot_status_5_to_10, v_previous_hot_status_10_to_15, v_previous_hot_status_15_to_20
            FROM hot_or_not_evaluator.video_hot_or_not_status
            WHERE video_id = v_video_record.video_id;
            
            -- Handle the case where this is the first record for the video
            -- Default to a random boolean value (TRUE/FALSE) for new videos instead of NULL
            v_previous_hot_status := COALESCE(v_previous_hot_status, (random() > 0.5));
            v_previous_hot_status_5_to_10 := COALESCE(v_previous_hot_status_5_to_10, (random() > 0.5));
            v_previous_hot_status_10_to_15 := COALESCE(v_previous_hot_status_10_to_15, (random() > 0.5));
            v_previous_hot_status_15_to_20 := COALESCE(v_previous_hot_status_15_to_20, (random() > 0.5));

            -- Calculate current average ds_score (last 5 minutes: 0 to -5 mins)
            SELECT AVG(ds_score)
            INTO v_current_avg_ds
            FROM hot_or_not_evaluator.video_engagement_relation
            WHERE video_id = v_video_record.video_id
            -- we do < v_now and not <= v_now because we want to avoid computing the average for the partially aggregated minute data
              AND timestamp_mnt >= v_5_mins_ago AND timestamp_mnt < v_now;

            -- Calculate average ds_score for -5 to -10 minutes window
            SELECT AVG(ds_score)
            INTO v_avg_ds_5_to_10_mins
            FROM hot_or_not_evaluator.video_engagement_relation
            WHERE video_id = v_video_record.video_id
              AND timestamp_mnt >= v_10_mins_ago AND timestamp_mnt < v_5_mins_ago;

            -- Calculate average ds_score for -10 to -15 minutes window
            SELECT AVG(ds_score)
            INTO v_avg_ds_10_to_15_mins
            FROM hot_or_not_evaluator.video_engagement_relation
            WHERE video_id = v_video_record.video_id
              AND timestamp_mnt >= v_15_mins_ago AND timestamp_mnt < v_10_mins_ago;

            -- Calculate average ds_score for -15 to -20 minutes window
            SELECT AVG(ds_score)
            INTO v_avg_ds_15_to_20_mins
            FROM hot_or_not_evaluator.video_engagement_relation
            WHERE video_id = v_video_record.video_id
              AND timestamp_mnt >= v_20_mins_ago AND timestamp_mnt < v_15_mins_ago;

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

            -- Determine if video is hot for current window (0 to -5 mins)
            IF v_current_avg_ds IS NOT NULL AND v_ref_predicted_avg_ds IS NOT NULL THEN
                v_is_hot := v_current_avg_ds > v_ref_predicted_avg_ds;
            ELSE
                v_is_hot := v_previous_hot_status; -- Maintain previous status if comparison cannot be made
            END IF;

            -- Determine if video is hot for -5 to -10 minutes window
            IF v_avg_ds_5_to_10_mins IS NOT NULL AND v_ref_predicted_avg_ds IS NOT NULL THEN
                v_is_hot_5_to_10_mins := v_avg_ds_5_to_10_mins > v_ref_predicted_avg_ds;
            ELSE
                v_is_hot_5_to_10_mins := v_previous_hot_status_5_to_10;
            END IF;

            -- Determine if video is hot for -10 to -15 minutes window
            IF v_avg_ds_10_to_15_mins IS NOT NULL AND v_ref_predicted_avg_ds IS NOT NULL THEN
                v_is_hot_10_to_15_mins := v_avg_ds_10_to_15_mins > v_ref_predicted_avg_ds;
            ELSE
                v_is_hot_10_to_15_mins := v_previous_hot_status_10_to_15;
            END IF;

            -- Determine if video is hot for -15 to -20 minutes window
            IF v_avg_ds_15_to_20_mins IS NOT NULL AND v_ref_predicted_avg_ds IS NOT NULL THEN
                v_is_hot_15_to_20_mins := v_avg_ds_15_to_20_mins > v_ref_predicted_avg_ds;
            ELSE
                v_is_hot_15_to_20_mins := v_previous_hot_status_15_to_20;
            END IF;

            -- Update status table (using UPSERT for the latest status)
            INSERT INTO hot_or_not_evaluator.video_hot_or_not_status (
                video_id, last_updated_mnt, hot_or_not,
                hot_or_not_5_to_10_mins_ago, hot_or_not_10_to_15_mins_ago, hot_or_not_15_to_20_mins_ago,
                current_avg_ds_score, reference_predicted_avg_ds_score,
                avg_ds_score_5_to_10_mins_ago, avg_ds_score_10_to_15_mins_ago, avg_ds_score_15_to_20_mins_ago
            )
            VALUES (
                v_video_record.video_id, v_now, v_is_hot,
                v_is_hot_5_to_10_mins, v_is_hot_10_to_15_mins, v_is_hot_15_to_20_mins,
                v_current_avg_ds, v_ref_predicted_avg_ds,
                v_avg_ds_5_to_10_mins, v_avg_ds_10_to_15_mins, v_avg_ds_15_to_20_mins
            )
            ON CONFLICT (video_id) DO UPDATE SET
                last_updated_mnt = EXCLUDED.last_updated_mnt,
                hot_or_not = EXCLUDED.hot_or_not,
                hot_or_not_5_to_10_mins_ago = EXCLUDED.hot_or_not_5_to_10_mins_ago,
                hot_or_not_10_to_15_mins_ago = EXCLUDED.hot_or_not_10_to_15_mins_ago,
                hot_or_not_15_to_20_mins_ago = EXCLUDED.hot_or_not_15_to_20_mins_ago,
                current_avg_ds_score = EXCLUDED.current_avg_ds_score,
                reference_predicted_avg_ds_score = EXCLUDED.reference_predicted_avg_ds_score,
                avg_ds_score_5_to_10_mins_ago = EXCLUDED.avg_ds_score_5_to_10_mins_ago,
                avg_ds_score_10_to_15_mins_ago = EXCLUDED.avg_ds_score_10_to_15_mins_ago,
                avg_ds_score_15_to_20_mins_ago = EXCLUDED.avg_ds_score_15_to_20_mins_ago;

        EXCEPTION
            WHEN OTHERS THEN
                RAISE WARNING 'Error processing video % in compute_hot_or_not: %', v_video_record.video_id, SQLERRM;
                CONTINUE; -- Move to the next video
        END;

    END LOOP;

    RAISE NOTICE 'Finished compute_hot_or_not at %', clock_timestamp();

END;
$$ LANGUAGE plpgsql;



CREATE OR REPLACE FUNCTION hot_or_not_evaluator.get_hot_or_not(p_video_id VARCHAR) -- old function for hot-or-not compatibility
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

COMMENT ON FUNCTION hot_or_not_evaluator.compute_hot_or_not() IS
'Calculates the "Hot or Not" status for all recently active videos across multiple time windows.
It compares the average ds_score of each window against a predicted score derived from an OLS linear regression.
Computes hot_or_not status for: current (0 to -5 mins), -5 to -10 mins, -10 to -15 mins, and -15 to -20 mins windows.
Also stores average ds_scores for all windows for analysis.
Updates the video_hot_or_not_status table. Intended to be run periodically (e.g., every 5 minutes).';

CREATE OR REPLACE FUNCTION hot_or_not_evaluator.get_hot_or_not_multiple_slots_v2(p_video_id VARCHAR)
RETURNS BOOLEAN AS $$
DECLARE
    v_status_current BOOLEAN;
    v_status_5_to_10 BOOLEAN;
    v_status_10_to_15 BOOLEAN;
    v_status_15_to_20 BOOLEAN;
    v_chosen_window INT;
    v_chosen_status BOOLEAN;
    v_now TIMESTAMPTZ := NOW();
    v_record_exists BOOLEAN := FALSE;
BEGIN
    -- Get all hot_or_not statuses for the video
    SELECT 
        hot_or_not, 
        hot_or_not_5_to_10_mins_ago, 
        hot_or_not_10_to_15_mins_ago, 
        hot_or_not_15_to_20_mins_ago,
        TRUE
    INTO 
        v_status_current, 
        v_status_5_to_10, 
        v_status_10_to_15, 
        v_status_15_to_20,
        v_record_exists
    FROM hot_or_not_evaluator.video_hot_or_not_status
    WHERE video_id = p_video_id;
    
    -- If no record exists, create one with random values for all windows
    IF NOT v_record_exists THEN
        v_status_current := (random() > 0.5);
        v_status_5_to_10 := (random() > 0.5);
        v_status_10_to_15 := (random() > 0.5);
        v_status_15_to_20 := (random() > 0.5);
        
        INSERT INTO hot_or_not_evaluator.video_hot_or_not_status (
            video_id, last_updated_mnt, hot_or_not,
            hot_or_not_5_to_10_mins_ago, hot_or_not_10_to_15_mins_ago, hot_or_not_15_to_20_mins_ago,
            current_avg_ds_score, reference_predicted_avg_ds_score,
            avg_ds_score_5_to_10_mins_ago, avg_ds_score_10_to_15_mins_ago, avg_ds_score_15_to_20_mins_ago
        )
        VALUES (
            p_video_id, v_now, v_status_current,
            v_status_5_to_10, v_status_10_to_15, v_status_15_to_20,
            NULL, NULL, NULL, NULL, NULL
        );
    END IF;
    
    -- Randomly choose one of the four windows (1-4) with equal probability
    v_chosen_window := floor(random() * 4) + 1;
    
    -- Get the status from the chosen window
    CASE v_chosen_window
        WHEN 1 THEN
            v_chosen_status := v_status_current;
            -- If current window status is NULL, generate random and update
            IF v_chosen_status IS NULL THEN
                v_chosen_status := (random() > 0.5);
                UPDATE hot_or_not_evaluator.video_hot_or_not_status 
                SET hot_or_not = v_chosen_status, last_updated_mnt = v_now
                WHERE video_id = p_video_id;
            END IF;
        WHEN 2 THEN
            v_chosen_status := v_status_5_to_10;
            -- If 5-10 mins window status is NULL, generate random and update
            IF v_chosen_status IS NULL THEN
                v_chosen_status := (random() > 0.5);
                UPDATE hot_or_not_evaluator.video_hot_or_not_status 
                SET hot_or_not_5_to_10_mins_ago = v_chosen_status, last_updated_mnt = v_now
                WHERE video_id = p_video_id;
            END IF;
        WHEN 3 THEN
            v_chosen_status := v_status_10_to_15;
            -- If 10-15 mins window status is NULL, generate random and update
            IF v_chosen_status IS NULL THEN
                v_chosen_status := (random() > 0.5);
                UPDATE hot_or_not_evaluator.video_hot_or_not_status 
                SET hot_or_not_10_to_15_mins_ago = v_chosen_status, last_updated_mnt = v_now
                WHERE video_id = p_video_id;
            END IF;
        WHEN 4 THEN
            v_chosen_status := v_status_15_to_20;
            -- If 15-20 mins window status is NULL, generate random and update
            IF v_chosen_status IS NULL THEN
                v_chosen_status := (random() > 0.5);
                UPDATE hot_or_not_evaluator.video_hot_or_not_status 
                SET hot_or_not_15_to_20_mins_ago = v_chosen_status, last_updated_mnt = v_now
                WHERE video_id = p_video_id;
            END IF;
    END CASE;

    RETURN v_chosen_status;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hot_or_not_evaluator.get_hot_or_not(VARCHAR) IS
'Retrieves a randomly chosen "Hot or Not" status from one of the four time windows with equal probability.
Windows: current (0 to -5 mins), -5 to -10 mins, -10 to -15 mins, -15 to -20 mins.
If the chosen window has no status, generates a random boolean value, persists it to the database, and returns it.
If no record exists for the video, creates a new record with random values for all windows.';

-- Function to update average reference predicted ds score in metric_const table
CREATE OR REPLACE FUNCTION hot_or_not_evaluator.update_avg_reference_predicted_ds_score()
RETURNS VOID AS $$
DECLARE
    v_current_time TIMESTAMPTZ := NOW();
    v_three_days_ago TIMESTAMPTZ := v_current_time - INTERVAL '3 days';
    
    -- Current values from metric_const
    v_current_avg NUMERIC;
    v_current_count BIGINT;
    v_last_updated TIMESTAMPTZ;
    
    -- New data from last 3 days
    v_new_avg NUMERIC;
    v_new_count BIGINT;
    
    -- Updated values
    v_updated_avg NUMERIC;
    v_updated_count BIGINT;
    
    -- For handling first run
    v_metric_const_exists BOOLEAN := FALSE;
BEGIN
    -- Get current values from metric_const
    SELECT 
        avg_reference_predicted_ds_score,
        avg_reference_predicted_ds_score_count,
        avg_reference_predicted_ds_score_last_updated,
        TRUE
    INTO 
        v_current_avg,
        v_current_count,
        v_last_updated,
        v_metric_const_exists
    FROM hot_or_not_evaluator.metric_const 
    WHERE id = 1;
    
    -- Initialize defaults if this is the first run or row doesn't exist
    IF NOT v_metric_const_exists THEN
        v_current_avg := NULL;
        v_current_count := 0;
        v_last_updated := NULL;
        RAISE NOTICE 'First run: metric_const row does not exist, will create it';
    END IF;
    
    -- Calculate new average and count from videos updated in the last 3 days
    -- Only include videos that have been updated since our last calculation
    SELECT 
        AVG(reference_predicted_avg_ds_score),
        COUNT(reference_predicted_avg_ds_score)
    INTO 
        v_new_avg,
        v_new_count
    FROM hot_or_not_evaluator.video_hot_or_not_status
    WHERE reference_predicted_avg_ds_score IS NOT NULL
      AND last_updated_mnt >= v_three_days_ago
      AND (v_last_updated IS NULL OR last_updated_mnt > v_last_updated);
    
    -- Handle case where no new data is available
    IF v_new_count = 0 THEN
        RAISE NOTICE 'No new data found since last update (%), keeping existing values', v_last_updated;
        
        -- Update only the timestamp to mark that we checked
        UPDATE hot_or_not_evaluator.metric_const 
        SET avg_reference_predicted_ds_score_last_updated = v_current_time
        WHERE id = 1;
        
        RETURN;
    END IF;
    
    -- Calculate updated average using incremental formula
    IF v_current_avg IS NULL OR v_current_count = 0 THEN
        -- First calculation or reset
        v_updated_avg := v_new_avg;
        v_updated_count := v_new_count;
        RAISE NOTICE 'Initial calculation: avg=%, count=%', v_updated_avg, v_updated_count;
    ELSE
        -- Incremental update: new_avg = (old_avg * old_count + new_avg * new_count) / (old_count + new_count)
        v_updated_count := v_current_count + v_new_count;
        v_updated_avg := (v_current_avg * v_current_count + v_new_avg * v_new_count) / v_updated_count;
        RAISE NOTICE 'Incremental update: old_avg=% (count=%), new_avg=% (count=%), updated_avg=% (count=%)', 
                     v_current_avg, v_current_count, v_new_avg, v_new_count, v_updated_avg, v_updated_count;
    END IF;
    
    -- Update metric_const table using UPSERT
    INSERT INTO hot_or_not_evaluator.metric_const (
        id, 
        like_ctr_center, 
        like_ctr_range, 
        watch_percentage_center, 
        watch_percentage_range, 
        avg_reference_predicted_ds_score,
        avg_reference_predicted_ds_score_count,
        avg_reference_predicted_ds_score_last_updated
    )
    VALUES (
        1, 0, 0.05, 0, 0.9, 
        v_updated_avg,
        v_updated_count,
        v_current_time
    )
    ON CONFLICT (id) DO UPDATE SET
        avg_reference_predicted_ds_score = EXCLUDED.avg_reference_predicted_ds_score,
        avg_reference_predicted_ds_score_count = EXCLUDED.avg_reference_predicted_ds_score_count,
        avg_reference_predicted_ds_score_last_updated = EXCLUDED.avg_reference_predicted_ds_score_last_updated;

    RAISE NOTICE 'Successfully updated metric_const: avg=%, count=%, timestamp=%', 
                 v_updated_avg, v_updated_count, v_current_time;

END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hot_or_not_evaluator.update_avg_reference_predicted_ds_score() IS
'Efficiently calculates and updates the average of reference_predicted_avg_ds_score using incremental updates.
Only processes videos updated in the last 3 days since the last calculation.
Uses the formula: new_avg = (old_avg * old_count + new_avg * new_count) / (old_count + new_count)
Maintains count and timestamp for efficient incremental processing.
Should be run every 3 days via cron job.';

-- Function to compare two videos and determine hot or not status
CREATE OR REPLACE FUNCTION hot_or_not_evaluator.compare_videos_hot_or_not(
    p_current_video_id VARCHAR,
    p_prev_video_id VARCHAR DEFAULT NULL
)
RETURNS BOOLEAN AS $$
DECLARE
    v_current_score NUMERIC;
    v_prev_score NUMERIC;
    v_result BOOLEAN;
    v_now TIMESTAMPTZ := NOW();
    v_current_video_exists BOOLEAN := FALSE;
    v_prev_video_exists BOOLEAN := FALSE;
BEGIN
    -- Input validation: Check if current_video_id is provided and not empty
    IF p_current_video_id IS NULL OR TRIM(p_current_video_id) = '' THEN
        v_result := (random() > 0.5);
        RAISE WARNING 'Invalid current_video_id provided (NULL or empty), returning random result: %', v_result;
        RETURN v_result;
    END IF;
    
    -- Input validation: Check if prev_video_id is empty string (treat as NULL)
    IF p_prev_video_id IS NOT NULL AND TRIM(p_prev_video_id) = '' THEN
        p_prev_video_id := NULL;
        RAISE NOTICE 'Empty prev_video_id converted to NULL, will compare against global average';
    END IF;
    
    -- Check if current video exists and get its score
    SELECT 
        reference_predicted_avg_ds_score,
        TRUE
    INTO 
        v_current_score,
        v_current_video_exists
    FROM hot_or_not_evaluator.video_hot_or_not_status
    WHERE video_id = TRIM(p_current_video_id);
    
    -- Handle non-existent current video
    IF NOT v_current_video_exists THEN
        v_result := (random() > 0.5);
        RAISE WARNING 'Current video_id "%" does not exist in video_hot_or_not_status table, returning random result: %', 
                     p_current_video_id, v_result;
        RETURN v_result;
    END IF;
    
    -- If prev_video_id is provided, validate and get its score
    IF p_prev_video_id IS NOT NULL THEN
        SELECT 
            reference_predicted_avg_ds_score,
            TRUE
        INTO 
            v_prev_score,
            v_prev_video_exists
        FROM hot_or_not_evaluator.video_hot_or_not_status
        WHERE video_id = TRIM(p_prev_video_id);
        
        -- Handle non-existent previous video
        IF NOT v_prev_video_exists THEN
            v_result := (random() > 0.5);
            RAISE WARNING 'Previous video_id "%" does not exist in video_hot_or_not_status table, returning random result: %', 
                         p_prev_video_id, v_result;
            RETURN v_result;
        END IF;
        
        -- Check if either score is missing (NULL)
        IF v_current_score IS NULL OR v_prev_score IS NULL THEN
            v_result := (random() > 0.5);
            RAISE NOTICE 'Missing score data for comparison - Current: "%" (score: %), Previous: "%" (score: %), returning random result: %', 
                         p_current_video_id, v_current_score, p_prev_video_id, v_prev_score, v_result;
            RETURN v_result;
        END IF;
        
        -- Compare current video score with previous video score
        v_result := (v_current_score >= v_prev_score);
        
        RAISE NOTICE 'Video comparison - Current: "%" (score: %), Previous: "%" (score: %), Result: % (%)', 
                     p_current_video_id, v_current_score, p_prev_video_id, v_prev_score, 
                     v_result, CASE WHEN v_result THEN 'HOT' ELSE 'NOT' END;
        
    ELSE
        -- No previous video provided, return random result
        v_result := (random() > 0.5);
        RAISE NOTICE 'No previous video provided for comparison with current video "%", returning random result: %', 
                     p_current_video_id, v_result;
    END IF;
    
    RETURN v_result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION hot_or_not_evaluator.compare_videos_hot_or_not(VARCHAR, VARCHAR) IS
'Compares two videos based on their reference_predicted_avg_ds_score values and returns hot (TRUE) or not (FALSE).
Parameters:
- p_current_video_id: The current video to evaluate (required, cannot be NULL or empty)
- p_prev_video_id: The previous video to compare against (optional)

Logic:
- Validates input parameters (NULL/empty checks)
- Verifies video existence in database before attempting comparison
- If prev_video_id is provided and valid: compares current vs previous video scores
- If prev_video_id is NULL: returns random result
- If any required data is missing or invalid: returns a random boolean result
- Returns TRUE (hot) if current score >= previous score, FALSE (not) otherwise

Corner Cases Handled:
- NULL or empty current_video_id → random result with warning
- Non-existent current_video_id → random result with warning  
- Non-existent prev_video_id → random result with warning
- Missing scores (NULL values) → random result with notice
- Empty string prev_video_id → treated as NULL, returns random result
- NULL prev_video_id → random result

Simple Strategy: Only one comparison mode (video-to-video). Everything else returns random results.';
