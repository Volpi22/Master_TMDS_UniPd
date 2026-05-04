'''Script to ingest data into Neo4j from a JSON file.'''
import os
import re
import json
from typing import LiteralString
from neo4j import GraphDatabase, Driver, Session, Query

from config.conf import NEO4J_URI, DATABASE_NAME, CYPHER_DIR, CYPHER_ABSOLUTE_FILE_A1


def connect_to_neo4j(uri: str) -> Driver: 
    """Connect to Neo4j database."""
    driver = GraphDatabase.driver(uri, auth=None)  # set dbms.security.auth_enabled=false in neo4j.conf to disable authentication
    return driver

def execute_cypher_query_from_file(file_path: str, database: str = DATABASE_NAME) -> None:
    """Execute Cypher queries from a file (supports multiple statements)."""
    driver = connect_to_neo4j(NEO4J_URI)
    with driver.session(database=database) as session, open(file_path, 'r', encoding='utf-8') as f:
        full_query = f.read()
        
        # Safely remove comments: Only match '//' at the start of a line or preceded by whitespace
        full_query_no_comments = re.sub(r'(?m)(?:^|\s)//.*$', '', full_query)
        
        # Split by semicolon and filter out empty statements
        statements = [ stmt.strip() for stmt in full_query_no_comments.split(';') if stmt.strip() ]
        
        for i, statement in enumerate(statements, 1):
            try:
                result = session.run(statement)
                summary = result.consume()
                print(f"Statement {i}/{len(statements)} executed: {summary.counters}")
            except Exception as e:
                print(f"Error in statement {i}: {e}")
                print(f"Failed statement: {statement[:100]}...")
                raise
        
        session.close()

def ingest_data_into_neo4j():
    """Ingest data into Neo4j from CSV files."""
    cypher_file_path = os.path.join(CYPHER_DIR, CYPHER_ABSOLUTE_FILE_A1)
    execute_cypher_query_from_file(cypher_file_path, DATABASE_NAME)

def custom_json_serializer(obj):
    """Fallback JSON serializer for objects (like Dates) that json doesn't recognize."""
    return str(obj)

def execute_and_save_queries(file_path: str, output_dir: str, database: str = DATABASE_NAME) -> None:
    """Execute Cypher queries and save the results to JSON files."""
    driver = connect_to_neo4j(NEO4J_URI)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with driver.session(database=database) as session, open(file_path, 'r', encoding='utf-8') as f:
        full_query = f.read()
        
        # Safely remove comments
        full_query_no_comments = re.sub(r'(?m)(?:^|\s)//.*$', '', full_query)
        
        # Split by semicolon and filter out empty statements
        statements = [ stmt.strip() for stmt in full_query_no_comments.split(';') if stmt.strip() ]
        
        for i, statement in enumerate(statements, 1):
            try:
                # Run the query
                result = session.run(statement)
                
                # Extract the data as a list of dictionaries
                records = result.data() 
                
                # Save to a JSON file (e.g., query_1_results.json)
                output_file = os.path.join(output_dir, f"query_B{i}_results.json")
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(records, out_f, indent=4, default=custom_json_serializer)
                    
                print(f"Statement {i}/{len(statements)} executed. Saved {len(records)} records to {output_file}")
                
            except Exception as e:
                print(f"Error in statement {i}: {e}")
                print(f"Failed statement: {statement[:100]}...")
                raise

