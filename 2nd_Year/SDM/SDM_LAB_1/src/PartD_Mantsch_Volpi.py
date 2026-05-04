'''Main script to run the queries for Section C.'''
import os
from config.conf import PROJECT_PATH, CYPHER_DIR, CYPHER_QUERY_FILE_D
from ingest import execute_cypher_query_from_file

if __name__ == "__main__":
    print(f"Project path: {PROJECT_PATH}")

    cypher_query_file_path = os.path.join(CYPHER_DIR, CYPHER_QUERY_FILE_D)
    
    results_dir = os.path.join(PROJECT_PATH, "results")

    execute_cypher_query_from_file(cypher_query_file_path)
