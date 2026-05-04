'''Main script to run the queries for Section B.'''
import os
from config.conf import PROJECT_PATH, CYPHER_DIR, CYPHER_QUERY_FILE_B
from ingest import execute_and_save_queries

if __name__ == "__main__":
    print(f"Project path: {PROJECT_PATH}")

    cypher_query_file_path = os.path.join(CYPHER_DIR, CYPHER_QUERY_FILE_B)
    
    results_dir = os.path.join(PROJECT_PATH, "results")

    execute_and_save_queries(cypher_query_file_path, results_dir)