# Stored procedures exposed for the frontend server function


## 1. `hot_or_not_evaluator.update_counter`

Updates video engagement metrics based on a single user interaction.

**Signature:**

```sql
hot_or_not_evaluator.update_counter(
    p_video_id VARCHAR,
    p_liked BOOLEAN,
    p_watch_percentage NUMERIC -- Expecting 0-100
)
```

**Purpose:**

Called by the frontend server function whenever a user interacts with a video. It aggregates engagement data per minute and updates various metrics, including the `ds_score` used for the "Hot or Not" calculation.

**Parameters:**

*   `p_video_id` (VARCHAR): The unique identifier of the video being interacted with.
*   `p_liked` (BOOLEAN): `TRUE` if the user liked the video during this interaction, `FALSE` otherwise.
*   `p_watch_percentage` (NUMERIC): The percentage of the video the user watched during this session (range: 0-100).


## 2. `hot_or_not_evaluator.get_hot_or_not` (**OLD**)

Retrieves the current "Hot or Not" status for a specific video.

**Signature:**

```sql
hot_or_not_evaluator.get_hot_or_not(
    p_video_id VARCHAR
)
RETURNS BOOLEAN
```

**Purpose:**

Called by the frontend to determine if a specific video should be presented as "hot". The actual calculation is performed periodically by the backend `compute_hot_or_not` function.

**Parameters:**

*   `p_video_id` (VARCHAR): The unique identifier of the video whose status is being queried.

**Returns:**

*   `BOOLEAN`: `TRUE` if the video is currently considered "hot", `FALSE` if not. Returns `NULL` if the video doesn't have a status entry yet (e.g., it's new or hasn't been processed).



## 3. `hot_or_not_evaluator.get_hot_or_not_multiple_slots_v2` (**CURRENT**)

Enhanced version that retrieves "Hot or Not" status from multiple time windows with dynamic selection.

**Signature:**

```sql
hot_or_not_evaluator.get_hot_or_not_multiple_slots_v2(
    p_video_id VARCHAR
)
RETURNS BOOLEAN
```

**Purpose:**

Provides hot-or-not status retrieval that selects from four different time windows with equal probability (25% each), offering more dynamic and temporally-aware results compared to the standard function. This approach helps combat gaming behavior by introducing temporal unpredictability.

**Parameters:**

*   `p_video_id` (VARCHAR): The unique identifier of the video whose status is being queried.

**Returns:**

*   `BOOLEAN`: `TRUE` if the video is considered "hot" in the selected time window, `FALSE` otherwise. 

**Time Windows (selected with equal probability):**
1. **Current window** (0 to -5 minutes ago)
2. **5-10 minutes ago window** 
3. **10-15 minutes ago window**
4. **15-20 minutes ago window**

**Behavior:**
- **Dynamic Window Selection**: Each call chooses one of the four time windows with equal probability
- **Missing Data Handling**: If the selected window has no status (NULL), generates a boolean value and persists it to the database
- **New Videos**: If no record exists for the video, creates a new record with boolean values for all four windows
- **Consistency**: Generated values are stored in the database to ensure consistent results for subsequent calls until the next calculation cycle

**Key Differences from `get_hot_or_not`:**
- Uses multiple time windows instead of just current status
- Provides temporal variation as a defense against gaming behavior
- Maintains backward compatibility with same return type
- Ensures data persistence for consistent user experience
- Acts as an anti-gaming measure until more sophisticated solutions are implemented 


## 4. `hot_or_not_evaluator.compare_videos_hot_or_not` (**CURRENT**)

Compares two videos based on their scores and returns a hot-or-not result.

**Signature:**

```sql
hot_or_not_evaluator.compare_videos_hot_or_not(
    p_current_video_id VARCHAR,
    p_prev_video_id VARCHAR DEFAULT NULL
)
RETURNS BOOLEAN
```

**Purpose:**

Provides a comparative approach to hot-or-not evaluation by comparing the reference predicted scores of two videos. This enables relative ranking between videos rather than absolute ranking with it's past.

**Parameters:**

*   `p_current_video_id` (VARCHAR): The current video to evaluate (required, cannot be NULL or empty).
*   `p_prev_video_id` (VARCHAR, optional): The previous video to compare against. If NULL or not provided, returns a result based on available data.

**Returns:**

*   `BOOLEAN`: `TRUE` (hot) if current video's score >= previous video's score, `FALSE` (not) otherwise.


## 5. `hot_or_not_evaluator.compare_videos_hot_or_not_v2` (**ENHANCED**)

Enhanced version of video comparison that returns detailed results including both video scores.

**Signature:**

```sql
hot_or_not_evaluator.compare_videos_hot_or_not_v2(
    p_current_video_id VARCHAR,
    p_prev_video_id VARCHAR DEFAULT NULL
)
RETURNS hot_or_not_evaluator.video_comparison_result
```

**Return Type:**
```sql
TYPE video_comparison_result AS (
    hot_or_not BOOLEAN,
    current_video_score NUMERIC,
    previous_video_score NUMERIC
)
```

**Purpose:**

Enhanced comparative approach that not only determines hot-or-not status but also returns the actual scores used in the comparison. This provides transparency and enables more sophisticated frontend logic based on score differences.

**Parameters:**

*   `p_current_video_id` (VARCHAR): The current video to evaluate (required, cannot be NULL or empty).
*   `p_prev_video_id` (VARCHAR, optional): The previous video to compare against. If NULL, generates a comparison score.

**Returns:**

*   `video_comparison_result` containing:
    - `hot_or_not` (BOOLEAN): `TRUE` (hot) if current score >= previous score, `FALSE` (not) otherwise
    - `current_video_score` (NUMERIC): The normalized reference_predicted_avg_ds_score for the current video
    - `previous_video_score` (NUMERIC): The normalized reference_predicted_avg_ds_score for the previous video (or generated score if prev_video_id is NULL)

**Key Features:**
- **Score Transparency**: Returns actual scores used in comparison for frontend analysis
- **Score Normalization**: Automatically normalizes scores to 0-100 range for consistency
- **Concurrent Safety**: Uses `INSERT ... ON CONFLICT` with `COALESCE` to handle concurrent requests
- **Minimal Impact**: Only populates essential columns, leaving others NULL to avoid affecting existing systems

**Key Differences from `compare_videos_hot_or_not`:**
- Returns structured result with all three values instead of just boolean
- Provides score transparency
- Implements score normalization for consistent ranges
- Robust concurrent handling

## 6. `hot_or_not_evaluator.update_counter_v3`

**Signature:**

```sql
hot_or_not_evaluator.update_counter_v3(
    p_video_id VARCHAR,
    p_liked BOOLEAN,
    p_watch_percentage NUMERIC -- Expecting 0-100
)
```

**Purpose:**

New v3 endpoint to give smooth scores

---

## 7. `hot_or_not_evaluator.compare_videos_hot_or_not_v3` (**CURRENT**)

Compares two videos based on their `ds_percentile_score` and returns a hot-or-not decision.

**Signature:**

```sql
hot_or_not_evaluator.compare_videos_hot_or_not_v3(
    p_current_video_id VARCHAR,
    p_prev_video_id VARCHAR
)
RETURNS BOOLEAN
```

**Purpose:**

New v3 endpoint with smooth scores

**Parameters:**

* `p_current_video_id` (VARCHAR): Video being evaluated.
* `p_prev_video_id` (VARCHAR): Video from the previous round; may be `NULL`.

**Returns:**

* `BOOLEAN`: `TRUE` (hot) if current ≥ previous, `FALSE` (not) otherwise.  




