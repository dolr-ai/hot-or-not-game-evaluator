from datetime import datetime
import pg8000
from google.cloud.alloydb.connector import Connector
from google.oauth2 import service_account

import os
from dotenv import load_dotenv

def execute_query(cursor, query, params=None, description=None):
    """
    Execute a database query and measure execution time.
    
    Args:
        cursor: Database cursor
        query: SQL query to execute
        params: Optional parameters for the query
        description: Optional description of the query for logging
    
    Returns:
        tuple: (result, time_taken_ms)
    """
    print(f"\n--- {description or 'Executing query'} ---")
    current_time = datetime.now()
    
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
        
    result = cursor.fetchone() if query.lower().strip().startswith("select") else None
    
    time_taken = datetime.now() - current_time
    time_taken_ms = time_taken.total_seconds() * 1000
    
    if description:
        if result is not None:
            print(f"Result: {result[0]}")
        else:
            print("Query executed successfully")
        print(f"Time taken: {time_taken_ms:.2f} ms")
    
    return result, time_taken_ms

# Load environment variables from .env file
load_dotenv()

project_id = os.getenv("PROJECT_ID")
region = os.getenv("REGION")
cluster_id = os.getenv("CLUSTER_ID")
instance_id = os.getenv("INSTANCE_ID")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER_AUTH_PROXY")

credentials_json_fpath = os.getenv("SERVICE_CRED_FPATH")

instance_connection_name = f"projects/{project_id}/locations/{region}/clusters/{cluster_id}/instances/{instance_id}"
credentials = service_account.Credentials.from_service_account_file(credentials_json_fpath)

connector = Connector(credentials=credentials)

try:
    print(f"Connecting to {instance_connection_name}...")
    # Connect directly without creating a SQLAlchemy pool
    conn = connector.connect(
        instance_connection_name,
        "pg8000",
        user=db_user,
        db=db_name,
        ip_type="PUBLIC",
        enable_iam_auth=True
    )
    
    print("Connection successful!")
    # Create a cursor and execute a query
    cursor = conn.cursor()
    
    # First query: Get hot or not status
    result, time_taken_ms = execute_query(
        cursor,
        "select hot_or_not_evaluator.get_hot_or_not('sgx-test_video_decrease')",
        description="get hot or not status"
    )
    print(f"Initial status: {result[0]}")
    
    # Second query: Update counter
    result, time_taken_ms = execute_query(
        cursor,
        "select hot_or_not_evaluator.update_counter('sgx-test_video_decrease', true, 100)",
        description="update video counter"
    )
    
    # Third query: Get updated hot or not status
    result, time_taken_ms = execute_query(
        cursor,
        "select hot_or_not_evaluator.get_hot_or_not('sgx-test_video_decrease')",
        description="get hot or not status"
    )

    result, time_taken_ms = execute_query(
        cursor,
        "select hot_or_not_evaluator.get_hot_or_not('sgx-test_video_decrease')",
        description="get hot or not status"
    )

    print(f"Updated status: {result[0]}")
    
    # Close the cursor and connection
    cursor.close()
    conn.close()
    print("Connection closed.")
    
    # Close the connector when done
    connector.close()

except Exception as e:
    print(f"An error occurred: {e}")
    # Check IAM permissions for the identity running the script
    # Check if the AlloyDB API is enabled [7]
    # Check network connectivity (script needs VPC access for private IP)
