CREATE TABLE IF NOT EXISTS hot_or_not_evaluator.video_engagement_relation_cumulative (
    video_id                     VARCHAR(255) PRIMARY KEY,
    like_count                   BIGINT      DEFAULT 0,
    watch_count                  BIGINT      DEFAULT 0,

    like_perc                    NUMERIC,    -- like_count / watch_count (0–1, NULL when watch_count = 0)
    watch_perc                   NUMERIC,    -- average watch-percentage per view, scaled to 0–1
    ds_score                     NUMERIC,   

    ds_percentile_score          NUMERIC(5,2),            -- 0.00-100.00 (pre-computed by cron)
    last_updated_at              TIMESTAMPTZ DEFAULT NOW()
);


CREATE INDEX CONCURRENTLY idx_ve_rc_last_updated_at_brin
    ON hot_or_not_evaluator.video_engagement_relation_cumulative
    USING BRIN (last_updated_at);

   -- helps both the filter and keeps the rows pre-sorted on ds_score
CREATE INDEX CONCURRENTLY idx_ve_rc_lastupd_ds
    ON hot_or_not_evaluator.video_engagement_relation_cumulative (last_updated_at, ds_score)
    WHERE ds_score IS NOT NULL;

CREATE TABLE IF NOT EXISTS hot_or_not_evaluator.ds_score_percentile (
    percentile_bp SMALLINT PRIMARY KEY,       -- 0…10 000 → 0.00–100.00 percentile (basis-points)
    ds_cutoff     NUMERIC  NOT NULL,          -- raw ds_score threshold that corresponds to the percentile
    generated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX CONCURRENTLY idx_ds_score_percentile_cutoff_bp
    ON hot_or_not_evaluator.ds_score_percentile (ds_cutoff, percentile_bp DESC);

-- =========================================
--  Ingestion entry-point (called per view event)
-- =========================================
-- Parameters
--   p_video_id         – Unique identifier of the video that was viewed.
--   p_liked            – TRUE when the current viewer clicked like; FALSE otherwise.
--   p_watch_percentage – How much of the video was watched (0-100, already capped client-side).

CREATE OR REPLACE FUNCTION hot_or_not_evaluator.update_counter_v3(
    p_video_id VARCHAR,
    p_liked BOOLEAN,
    p_watch_percentage NUMERIC
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
BEGIN
  ------------------------------------------------------------------
  -- 1. UPSERT cumulative counters and compute all derived metrics
  ------------------------------------------------------------------
  INSERT INTO hot_or_not_evaluator.video_engagement_relation_cumulative (
          video_id, like_count, watch_count, watch_perc, like_perc, ds_score)
  VALUES  (p_video_id, 
           CASE WHEN p_liked THEN 1 ELSE 0 END, 
           1, 
           p_watch_percentage / 100.0,
           CASE WHEN p_liked THEN 1.0 ELSE 0.0 END,
           (CASE WHEN p_liked THEN 1.0 ELSE 0.0 END + p_watch_percentage / 100.0) / 2)
  ON CONFLICT (video_id) DO UPDATE SET
        like_count             = video_engagement_relation_cumulative.like_count  + EXCLUDED.like_count,
        watch_count            = video_engagement_relation_cumulative.watch_count + EXCLUDED.watch_count,
        watch_perc             = CASE
                                   WHEN video_engagement_relation_cumulative.watch_count = 0 THEN p_watch_percentage / 100.0
                                   ELSE (video_engagement_relation_cumulative.watch_perc * video_engagement_relation_cumulative.watch_count + p_watch_percentage / 100.0) / (video_engagement_relation_cumulative.watch_count + 1)
                                 END,
        like_perc              = CASE
                                   WHEN (video_engagement_relation_cumulative.watch_count + 1) = 0 THEN 0
                                   ELSE (video_engagement_relation_cumulative.like_count + EXCLUDED.like_count)::NUMERIC / (video_engagement_relation_cumulative.watch_count + 1)
                                 END,
        ds_score               = (
                                   -- part 1: like CTR component
                                   CASE
                                     WHEN (video_engagement_relation_cumulative.watch_count + 1) = 0 THEN 0
                                     ELSE (video_engagement_relation_cumulative.like_count + EXCLUDED.like_count)::NUMERIC / (video_engagement_relation_cumulative.watch_count + 1)
                                   END +
                                   -- part 2: average watch-percentage component
                                   CASE
                                     WHEN video_engagement_relation_cumulative.watch_count = 0 THEN p_watch_percentage / 100.0
                                     ELSE (video_engagement_relation_cumulative.watch_perc * video_engagement_relation_cumulative.watch_count + p_watch_percentage / 100.0) / (video_engagement_relation_cumulative.watch_count + 1)
                                   END
                                 ) / 2,
        last_updated_at        = NOW();
END;
$$;


-- ===============================================================
--  Rebuild of percentile lookup table (0.01 % resolution) | heavy operation | done once when distribution rebuild is required (every 1 hour for now, frequency will be reduced in the future once we have a more stable distribution)
-- CRON: every 1 hour
-- ===============================================================
CREATE OR REPLACE FUNCTION hot_or_not_evaluator.refresh_ds_score_percentiles()
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    v_pct_arr    NUMERIC[];  -- array[0.0000, 0.0001, …, 1.0000] – target percentiles
    v_cutoff_arr NUMERIC[];  -- matching ds_score cut-offs (same length as v_pct_arr)
BEGIN

    SELECT array_agg(i / 10000.0 ORDER BY i)
    INTO   v_pct_arr
    FROM   generate_series(0, 10000) AS gs(i);

    SELECT percentile_cont(v_pct_arr)
           WITHIN GROUP (ORDER BY ds_score)
    INTO   v_cutoff_arr
    FROM   hot_or_not_evaluator.video_engagement_relation_cumulative
    WHERE  last_updated_at >= NOW() - INTERVAL '7 days'
      AND  ds_score IS NOT NULL;

    IF v_cutoff_arr IS NULL THEN
        RAISE NOTICE
          'refresh_ds_score_percentiles(): no data in 7-day window – nothing updated';
        RETURN;
    END IF;

    MERGE INTO hot_or_not_evaluator.ds_score_percentile AS dst
    USING (
        SELECT gs.bp                       AS percentile_bp,
               v_cutoff_arr[gs.bp + 1]     AS ds_cutoff
        FROM   generate_series(0, 10000) AS gs(bp)
    ) AS src
    ON (dst.percentile_bp = src.percentile_bp)
    WHEN MATCHED THEN
         UPDATE SET ds_cutoff   = src.ds_cutoff,
                    generated_at = NOW()
    WHEN NOT MATCHED THEN
         INSERT (percentile_bp, ds_cutoff)
         VALUES (src.percentile_bp, src.ds_cutoff);

    RAISE NOTICE
      'refresh_ds_score_percentiles(): lookup table refreshed (% rows)',
      array_length(v_cutoff_arr, 1);
END;
$$;


-- Return a human-interpretable 0-100 score for a raw ds_score
-- Utility function
CREATE OR REPLACE FUNCTION hot_or_not_evaluator.ds_score_to_percentile(p_ds NUMERIC)
RETURNS NUMERIC  -- 0.00 – 100.00 (two decimals)
LANGUAGE SQL
STABLE
AS $$
    SELECT COALESCE(
               (SELECT percentile_bp / 100.0
                FROM   hot_or_not_evaluator.ds_score_percentile
                WHERE  ds_cutoff <= p_ds
                ORDER  BY percentile_bp DESC
                LIMIT  1),
               0.00
           );
$$;


-- Update cached percentile scores into the hot **video_engagement_relation_cumulative** table for recently modified videos
-- Note: The cadence for this should match the cadence for the cron schedule
-- CRON: every 5 minutes
CREATE OR REPLACE FUNCTION hot_or_not_evaluator.update_ds_percentile_scores()
RETURNS VOID
LANGUAGE SQL
AS $$
    UPDATE hot_or_not_evaluator.video_engagement_relation_cumulative v
    SET ds_percentile_score = hot_or_not_evaluator.ds_score_to_percentile(ds_score)
    WHERE last_updated_at >= NOW() - INTERVAL '7 minutes';
$$;


-- =========================================
--  Public API: decide HOT-or-NOT for a pair
-- =========================================
CREATE OR REPLACE FUNCTION hot_or_not_evaluator.compare_videos_hot_or_not_v3(
    p_current_video_id VARCHAR,
    p_prev_video_id VARCHAR
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    v_curr  NUMERIC;  -- percentile score of current candidate
    v_prev  NUMERIC;  -- percentile score of previous winner (can be NULL)
BEGIN
    SELECT ds_percentile_score INTO v_curr
    FROM   hot_or_not_evaluator.video_engagement_relation_cumulative
    WHERE  video_id = p_current_video_id;

    -- If current video not found, generate random score and insert
    IF v_curr IS NULL THEN
        v_curr := RANDOM() * 100; -- Generate random score between 0-100
        INSERT INTO hot_or_not_evaluator.video_engagement_relation_cumulative 
            (video_id, ds_percentile_score, last_updated_at)
        VALUES 
            (p_current_video_id, v_curr, '1900-01-01'::timestamp)
        ON CONFLICT (video_id) DO UPDATE
            SET ds_percentile_score = COALESCE(hot_or_not_evaluator.video_engagement_relation_cumulative.ds_percentile_score,
                                               EXCLUDED.ds_percentile_score);
    END IF;

    IF p_prev_video_id IS NOT NULL THEN
        SELECT ds_percentile_score INTO v_prev
        FROM   hot_or_not_evaluator.video_engagement_relation_cumulative
        WHERE  video_id = p_prev_video_id;
        
        -- If previous video not found, generate random score and insert
        IF v_prev IS NULL THEN
            v_prev := RANDOM() * 100; -- Generate random score between 0-100
            INSERT INTO hot_or_not_evaluator.video_engagement_relation_cumulative 
                (video_id, ds_percentile_score, last_updated_at)
            VALUES 
                (p_prev_video_id, v_prev, '1900-01-01'::timestamp)
            ON CONFLICT (video_id) DO UPDATE
                SET ds_percentile_score = COALESCE(hot_or_not_evaluator.video_engagement_relation_cumulative.ds_percentile_score,
                                                   EXCLUDED.ds_percentile_score);
        END IF;
        
        RETURN (v_curr >= v_prev);
    ELSE
        -- First round: treat ≥ 50th percentile as HOT
        RETURN (v_curr >= 50); -- Identity
    END IF;
END;
$$;




-- ===============================================================
--  CRON Job Scheduling
-- ===============================================================

-- Enable pg_cron extension if not already enabled
-- This should be run as superuser: CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Schedule refresh of ds_score percentiles every hour
-- This rebuilds the percentile lookup table with 0.01% resolution
SELECT cron.schedule(
    'refresh_ds_score_percentiles',
    '0 * * * *',  -- Every hour at minute 0
    'SELECT hot_or_not_evaluator.refresh_ds_score_percentiles();'
);

-- Schedule update of ds_percentile_scores every 15 minutes
-- This updates the pre-computed percentile scores for all videos
SELECT cron.schedule(
    'update_ds_percentile_scores', 
    '*/5 * * * *',  -- Every 5 minutes
    'SELECT hot_or_not_evaluator.update_ds_percentile_scores();'
);

-- View scheduled cron jobs
-- SELECT * FROM cron.job;

-- To remove a cron job (if needed):
-- SELECT cron.unschedule('refresh_ds_score_percentiles');
-- SELECT cron.unschedule('update_ds_percentile_scores');
