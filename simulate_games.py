import psycopg
import time
import random
import math
from datetime import datetime
from dotenv import load_dotenv
import os
import concurrent.futures

# Connection parameters (from explore.py)
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
conn_string = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password} sslmode=require"

# Define test video IDs for different patterns
VIDEO_IDS = {
    'linear_increase': 'test_linear_increase_like_and_watch_2',
    'quadratic_increase': 'test_quadratic_increase_like_and_watch_2',
    'linear_decrease': 'test_linear_decrease_like_and_watch_2',
    'quadratic_decrease': 'test_quadratic_decrease_like_and_watch_2'
}

# Function to add random noise to a value with 10% error rate
def add_noise(value, error_rate=0.1):
    noise = random.uniform(-error_rate * value, error_rate * value)
    return max(0, min(100, value + noise))  # Ensure result is between 0 and 100



# Function to generate engagement data based on pattern
def generate_engagement_target(pattern, time_step, total_steps):
    """
    Generate like probability and watch percentage based on pattern
    
    Parameters:
    - pattern: 'linear_increase', 'quadratic_increase', 'linear_decrease', 'quadratic_decrease'
    - time_step: current step (0 to total_steps-1)
    - total_steps: total number of steps
    
    Returns:
    - (like_probability, watch_percentage) both 0-100
    """
    progress = time_step / total_steps  # 0 to 1
    
    if pattern == 'linear_increase':
        # Start at 20, end at 80
        base_value = 20 + 60 * progress
    elif pattern == 'quadratic_increase':
        # Start at 20, end at 80, quadratic increase
        base_value = 20 + 60 * (progress ** 2)
    elif pattern == 'linear_decrease':
        # Start at 80, end at 20
        base_value = 80 - 60 * progress
    elif pattern == 'quadratic_decrease':
        # Start at 80, end at 20, quadratic decrease
        base_value = 80 - 60 * (progress ** 2)
    else:
        base_value = 50  # default
    
    # Add 10% noise to both values
    like_value = add_noise(base_value)
    watch_value = add_noise(base_value)
    
    return like_value, watch_value




# Function to simulate user interaction
def simulate_interaction(conn, video_id, pattern, time_step, total_steps):
    like_prob, watch_perc = generate_engagement_target(pattern, time_step, total_steps)
    
    # Determine if the video was liked based on probability
    liked = random.random() < (like_prob / 100)
    
    # Call update_counter procedure with explicit type casting
    with conn.cursor() as cur:
        # print(f"EXECUTING : Time: {datetime.now().strftime('%H:%M:%S')} - Video: {video_id} - Liked: {liked} - Watch %: {watch_perc:.2f}")
        query = f"SELECT hot_or_not_evaluator.update_counter(CAST('{video_id}' AS VARCHAR), CAST({str(liked).lower()} AS BOOLEAN), CAST({watch_perc} AS NUMERIC))"
        print(f"QUERY: {query}")
        cur.execute(query)
    
    # Explicitly commit the transaction
    conn.commit()
    

# Function to get hot or not status
def get_video_status(conn, video_id):
    with conn.cursor() as cur:
        cur.execute("SELECT hot_or_not_evaluator.get_hot_or_not(%s)", (video_id,))
        status = cur.fetchone()[0]
        
        # Also get the scores for context
        cur.execute(
            "SELECT current_avg_ds_score, reference_predicted_avg_ds_score FROM hot_or_not_evaluator.video_hot_or_not_status WHERE video_id = %s",
            (video_id,)
        )
        scores = cur.fetchone()
        
        if scores:
            current_score, ref_score = scores
            return status, current_score, ref_score
        return status, None, None

def main():
    try:
        start_time = datetime.now()
        # Connect to database
        with psycopg.connect(conn_string) as conn:
            # Disable autocommit to manage transactions explicitly
            conn.autocommit = False
            
            print(f"Connected to database. Starting simulation at {datetime.now().strftime('%H:%M:%S')}")
            
            # Total simulation time: 10 minutes with 10-second intervals
            total_steps = 65  
            
            # Run simulation for 10 minutes
            for step in range(total_steps):
                # Create a list of all tasks for concurrent execution
                all_tasks = []
                for pattern, video_id in VIDEO_IDS.items():
                    # Send 10 interactions for each video at each step
                    for _ in range(10):
                        all_tasks.append((video_id, pattern, step, total_steps))
                
                # Prepare all queries at once
                all_queries = []
                for vid, pat, st, tot in all_tasks:
                    like_prob, watch_perc = generate_engagement_target(pat, st, tot)
                    liked = random.random() < (like_prob / 100)
                    query = f"SELECT hot_or_not_evaluator.update_counter(CAST('{vid}' AS VARCHAR), CAST({str(liked).lower()} AS BOOLEAN), CAST({watch_perc} AS NUMERIC))"
                    all_queries.append(query)
                    print(f"QUERY: {query}")
                
                # Use a thread pool to execute all interactions concurrently
                with concurrent.futures.ThreadPoolExecutor(len(all_queries)) as executor:
                    # Submit all queries to be executed concurrently
                    for query in all_queries:
                        def execute_query(q):
                            with conn.cursor() as cur:
                                cur.execute(q)
                            conn.commit()
                        
                        executor.submit(execute_query, query)
                
                # Wait 10 seconds before next iteration
                if step < total_steps - 1:  # No need to wait after the last step
                    print(f"Step number {step} of {total_steps-1} - Waiting for 5 seconds before next iteration")
                    time.sleep(5)
            
            # After 10 minutes, get hot_or_not status for all videos
            print("\n--- Final Hot or Not Status ---")
            for pattern, video_id in VIDEO_IDS.items():
                status, current_score, ref_score = get_video_status(conn, video_id)
                status_text = "HOT" if status else "NOT HOT" if status is not None else "UNKNOWN"
                print(f"Video {video_id}: {status_text}")
                if current_score is not None and ref_score is not None:
                    print(f"  Current Score: {current_score:.5f}, Reference Score: {ref_score:.5f}")

            end_time = datetime.now()
            print(f"Total simulation time: {end_time - start_time}")
    
    except psycopg.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()