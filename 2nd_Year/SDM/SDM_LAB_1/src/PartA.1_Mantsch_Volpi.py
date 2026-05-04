'''Main script to fetch data from Semantic Scholar API and ingest into Neo4j.'''
import time

import pandas as pd

from config.conf import N_PAPERS_PER_YEAR, PROJECT_PATH, IMPORT_DIR, CYPHER_DIR, CYPHER_TEMPLATE_FILE_A1, CYPHER_ABSOLUTE_FILE_A1, N_YEARS_TO_FETCH, START_YEAR
import fetch
import synthetic_data
from ingest import ingest_data_into_neo4j
from util import generate_cypher_with_absolute_paths

if __name__ == "__main__":

    print(f"Project path: {PROJECT_PATH}")

    # Step 1: Fetch paper ids from Semantic Scholar API for a range of years
    # Fetch is limited to fields of study defined in FIELDS_OF_STUDY
    paper_ids_df = fetch.fetch_paper_ids(n_papers=N_PAPERS_PER_YEAR, year=START_YEAR)
    for year in range(START_YEAR + 1, START_YEAR + N_YEARS_TO_FETCH):
        paper_ids_year_df = fetch.fetch_paper_ids(n_papers=N_PAPERS_PER_YEAR, year=year)
        paper_ids_df = pd.concat([paper_ids_df, paper_ids_year_df], ignore_index=True)
    #print(paper_ids_df.head(10))
    time.sleep(1)
    # Step 2: Fetch paper details for the fetched paper ids
    paper_details_df = fetch.fetch_paper_details(paper_ids_df)
    #print(paper_details_df.head(10))

    # Step 3: Build dataframes from retrieved data
    authors_df = fetch.build_authors_and_institutions_df(paper_details_df)
    publications_df = fetch.build_publications_df(paper_details_df)
    paper_details_df = fetch.build_paper_details_df(paper_details_df)

    authors_df, paper_details_df, publications_df = synthetic_data.add_synthetic_node_properties(authors_df, paper_details_df, publications_df)
    publications_df, paper_details_df = synthetic_data.generate_cities_and_editions(publications_df, paper_details_df)
    reviews_df = synthetic_data.generate_reviewers(paper_details_df, authors_df)
    affiliations_df = synthetic_data.generate_city_university_mapping()
    paper_details_df = synthetic_data.add_synthetic_keywords(paper_details_df)

    authors_df, paper_details_df, publications_df = synthetic_data.enforce_lab_guarantees(authors_df, paper_details_df, publications_df)

    # Step 4: Save dataframes to CSV files for later ingestion into Neo4j
    fetch.save_df_to_csv(paper_details_df, "papers.csv")
    fetch.save_df_to_csv(authors_df, "authors.csv")
    fetch.save_df_to_csv(publications_df, "publications.csv")
    fetch.save_df_to_csv(reviews_df, "reviews.csv")
    fetch.save_df_to_csv(affiliations_df, "affiliations.csv")

    # Step 5: Ingest data into Neo4j from CSV files
    # Step 5a: Copy .csvs to neo4j import directory (assumes default neo4j configuration with import directory at /var/lib/neo4j/import/)
    # This step is not necessary if neo4j config is adjusted according to README
    # for filename in ["papers.csv", "authors.csv", "publications.csv"]:
    #     src_path = f"{PROJECT_PATH}/data/{filename}"
    #     dst_path = f"{IMPORT_DIR}/{filename}"
    #     print(f"Copying {src_path} to {dst_path}...")
    #     shutil.copy(src_path, dst_path)

    # Step 5b: Generate Cypher file with absolute paths to CSV files (adjusts the template Cypher file by replacing relative paths with absolute paths)
    cypher_template_path = f"{CYPHER_DIR}/{CYPHER_TEMPLATE_FILE_A1}"
    cypher_absolute_path = f"{CYPHER_DIR}/{CYPHER_ABSOLUTE_FILE_A1}"
    generate_cypher_with_absolute_paths(cypher_template_path, cypher_absolute_path)
    #Step 5c: Run Cypher queries to ingest data into Neo4j
    ingest_data_into_neo4j()
