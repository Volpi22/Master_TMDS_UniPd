'''Main script to add more data to Neo4j and run queries for Section A3.'''

from config.conf import PROJECT_PATH, CYPHER_DIR, CYPHER_TEMPLATE_FILE_A3, CYPHER_ABSOLUTE_FILE_A3
from ingest import execute_cypher_query_from_file
from util import generate_cypher_with_absolute_paths

if __name__ == "__main__":

    print(f"Project path: {PROJECT_PATH}")

    # Step 1: Generate Cypher file with absolute paths to CSV files (adjusts the template Cypher file by replacing relative paths with absolute paths)
    cypher_template_path = f"{CYPHER_DIR}/{CYPHER_TEMPLATE_FILE_A3}"
    cypher_absolute_path = f"{CYPHER_DIR}/{CYPHER_ABSOLUTE_FILE_A3}"
    generate_cypher_with_absolute_paths(cypher_template_path, cypher_absolute_path)
    #Step 2: Run Cypher queries to ingest data into Neo4j
    execute_cypher_query_from_file(cypher_absolute_path)
