''' A bunch of helper functions for the project.'''
import pandas as pd
import os
import csv
import html
import re
import unicodedata

from config.conf import PROJECT_PATH, CYPHER_DIR

#####>                                                                         <#####
# Data cleaning and transformation functions for the Semantic Scholar API responses #
#####>                                                                         <#####
def extract_venue_name(publication_name_long: str) -> str:
    '''Extract journal name from the publication name field in the API response.'''
    if publication_name_long is None:
        return None
    # The publication name field can have different formats, e.g. "Proceedings of the XYZ Conference", "Journal of ABC, Volume 10", etc.
    # We can use some heuristics to extract the journal name, e.g. look for keywords like "Proceedings", "Journal", etc.
    if "Proceedings" in publication_name_long:
        return "International " + publication_name_long.split("International")[-1].split(",")[0].strip()
    elif "Journal" in publication_name_long:
        return publication_name_long.split("Journal of ")[-1].split(",")[0].strip()
    else:
        return publication_name_long.strip()
    
def extract_publication_and_venue(row: dict) -> tuple[str | None, str | None]:
    '''Extract publication name and venue name from the API response row. 
        Returns (publication_name, venue_name).'''
    if row["publicationVenue"] is not None and "name" in row["publicationVenue"]:
        venue_name = row["publicationVenue"]["name"]
        publication_name = f"{venue_name}"
    else:
        journal = row.get("journal")
        publication_name = (
            journal.get("name", "")
            if isinstance(journal, dict) and pd.notnull(journal.get("name"))
            else ""
        )
        venue_name = extract_venue_name(publication_name) if publication_name is not None else None

    return publication_name, venue_name

def extract_publication_volume(row: dict) -> str | None:
    '''Extract publication volume from the API response row.'''
    journal = row.get("journal")
    if isinstance(journal, dict) and pd.notnull(journal.get("volume")):
        return journal.get("volume", "Vol. XY")
    else:
        return "Vol. XY"


    
def get_keywords_from_title(title: str) -> list:
    '''Extract keywords from the title of the paper.'''
    if title is None:
        return []
    # Simple heuristic: split the title into words and filter out common stop words
    stop_words = set(["the", "and", "of", "in", "to", "a", "for", "with", "on", "by", "is", "are", "over", "from", "as", "that", "this", "we", "be", "an", "at", "which", "or", "it"])
    keywords = [word.strip(",.()").lower() for word in title.split() if word.lower() not in stop_words and len(word) > 2]
    return keywords

def combine_keywords(row):
    '''Combine fields of study and keywords from title into a single list of keywords.'''
    fields = row["fieldsOfStudy"] if isinstance(row["fieldsOfStudy"], list) else []
    title_keywords = get_keywords_from_title(row["title"])
    return fields + title_keywords

def sanitize_str(value) -> str:
    """Sanitize title/abstract text before writing CSV for Neo4j LOAD CSV."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""

    s = str(value)

    # Normalize unicode and decode HTML entities
    s = unicodedata.normalize("NFC", html.unescape(s))

    # Convert scraper-style escapes to plain characters
    s = s.replace('\\"', '"').replace("\\'", "'")

    # Remove problematic control chars / null bytes
    s = s.replace("\x00", "")
    s = re.sub(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]", " ", s)

    # Flatten line breaks (safer for CSV import tools)
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

    # Clean extra whitespace
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def generate_cypher_with_absolute_paths(template_file: str, output_file: str):
    """Generate Cypher file with absolute paths to CSV files."""
    project_root = PROJECT_PATH
    
    # We replace backslashes with forward slashes for Neo4j (Safe for Mac/Windows/Linux)
    data_dir = os.path.join(project_root, "data").replace('\\', '/')
    
    with open(template_file, 'r', encoding='utf-8') as f:
        cypher_content = f.read()
    
    # Replace relative paths with absolute paths
    cypher_content = cypher_content.replace(
        'file:///data/',
        f'file:///{data_dir}/'
    )
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cypher_content)
