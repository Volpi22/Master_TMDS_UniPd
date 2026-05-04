'''Configuration variables for the Semantic Scholar API fetcher.'''

import os

API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
N_PAPERS_PER_YEAR = 200
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1
N_YEARS_TO_FETCH = 5
START_YEAR = 2018
API_KEY = "IHqNuOAfj52NwmrsyIZoM2yIZK63vTH01p2pbzuH" # Get API Key from Jan
PAPER_FIELDS = 'title,abstract,year,externalIds,authors.affiliations,authors.name,authors.homepage,references.paperId,citations.paperId,publicationVenue,publicationDate,publicationTypes,journal,fieldsOfStudy'
FIELDS_OF_STUDY = "Computer Science"
NEO4J_URI = "neo4j://localhost:7687"
DATABASE_NAME = "papers"
IMPORT_DIR = os.path.join("/var/lib/neo4j/import/") # Definitely needs adjustment if you're on Windowds or have a custom Neo4j configuration


PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CYPHER_DIR = os.path.join(PROJECT_PATH, "cypher")
CYPHER_TEMPLATE_FILE_A1 = "import_csv_a1.cypher"
CYPHER_ABSOLUTE_FILE_A1 = "import_csv_absolute_a1.cypher"
CYPHER_TEMPLATE_FILE_A3 = "import_csv_a3.cypher"
CYPHER_ABSOLUTE_FILE_A3 = "import_csv_absolute_a3.cypher"
CYPHER_QUERY_FILE_B = "queries_b.cypher"
CYPHER_QUERY_FILE_C = "queries_c.cypher"
CYPHER_QUERY_FILE_D = "queries_d.cypher"
