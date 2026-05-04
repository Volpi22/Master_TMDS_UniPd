'''Script to fetch data from Semantic Scholar API.'''
import csv
import os
import time

import requests
import pandas as pd

import config.conf as cfg
import util as util

AUTH_HEADER = {"x-api-key": cfg.API_KEY}

# Steps to fetch data:
# 1. Fetch paper ids from /paper/search/bulk (GET) (let's start with 10'000 papers)
# 2. For each paper id, fetch paper details from /paper/batch (POST)
#    Details include: title,abstract,year,url,authors.affiliations,authors.name,authors.homepage,references.paperId,publicationVenue,publicationDate,journal,fieldsOfStudy


# 1. Fetch paper ids from /paper/search/bulk (GET) (let's start with 10'000 papers)
'''Fetch paper ids from Semantic Scholar API.'''
def fetch_paper_ids(n_papers: int, year: int) -> pd.DataFrame:
    url = f"{cfg.API_BASE_URL}/paper/search/bulk"
    params = {
        "query": "*",
        "fields": "paperId",
        "publicationTypes": "JournalArticle,Conference",
        "publicationDateOrYear": f"{year}",
        "fieldsOfStudy": cfg.FIELDS_OF_STUDY,
    }
    paper_ids = pd.DataFrame()
    n_fetched = 0
    n_retries = 0
    while n_fetched < n_papers:
        response = requests.get(url, params=params, headers=AUTH_HEADER)
        if response.status_code != 200:
            print(f"Error fetching paper ids: {response.status_code} - {response.text}")
            if n_retries < cfg.MAX_RETRIES:
                n_retries += 1
                print(f"Retrying in {cfg.RETRY_DELAY_SECONDS} seconds... (Retry {n_retries}/{cfg.MAX_RETRIES})")
                time.sleep(cfg.RETRY_DELAY_SECONDS)
                continue
            else:
                print("Max retries reached. Stopping fetch.")
                break
        else: n_retries = 0

        response_data = response.json()
        response_df = pd.DataFrame(response_data["data"])
        if n_fetched + len(response_df) > n_papers:
            response_df = response_df.iloc[:n_papers - n_fetched]
        paper_ids = pd.concat([paper_ids, response_df], ignore_index=True)
        n_fetched = len(paper_ids)
        if n_fetched < n_papers:
            print(f"Year:{year} - Fetched {n_fetched} paper ids so far...")

        if "token" not in response_data:
            print("No more papers to fetch.")
            break
        params["token"] = response_data["token"]

    print(f"Year:{year} - Fetched a total of {len(paper_ids)} paper ids.")
    return paper_ids

'''Fetch paper details for a list of paper ids from Semantic Scholar API.'''
def fetch_paper_details(paper_ids: pd.DataFrame) -> pd.DataFrame:
    url = f"{cfg.API_BASE_URL}/paper/batch"
    params = {
        "fields": cfg.PAPER_FIELDS,
    }
    paper_details_list = []
    n_fetched = 0
    n_retries = 0
    for i in range(0, len(paper_ids), 500): # Fetch details in batches of 500
        batch_ids = paper_ids["paperId"].iloc[i:i+500].tolist()
        response = requests.post(url, params=params, json={"ids": batch_ids}, headers=AUTH_HEADER)
        if response.status_code != 200:
            print(f"Error fetching paper details: {response.status_code} - {response.text}")
            if n_retries < cfg.MAX_RETRIES:
                n_retries += 1
                print(f"Retrying in {cfg.RETRY_DELAY_SECONDS} seconds... (Retry {n_retries}/{cfg.MAX_RETRIES})")
                time.sleep(cfg.RETRY_DELAY_SECONDS)
                continue
            else:
                print("Max retries reached. Stopping fetch.")
                break
        else: n_retries = 0

        response_data = response.json()
        paper_details_list.extend(response_data)
        n_fetched += len(response_data)
        print(f"Fetched details for {n_fetched} papers so far...")

    paper_details_df = pd.DataFrame(paper_details_list)
    print(f"Fetched details for a total of {len(paper_details_df)} papers.")
    return paper_details_df

'''Builds dataframe for authors with columns: authorId, name, affiliations, homepage'''
def build_authors_and_institutions_df(paper_details_df: pd.DataFrame) -> pd.DataFrame:
    authors = pd.DataFrame()
    exploded = paper_details_df.explode("authors")
    authors["authorId"] = exploded["authors"].apply(lambda author: author["authorId"] if isinstance(author, dict) else None)
    authors["name"] = exploded["authors"].apply(lambda author: author["name"] if isinstance(author, dict) else None)
    # authors["affiliations"] = exploded["authors"].apply(lambda author: author["affiliations"] if isinstance(author, dict) else None)
    # authors["homepage"] = exploded["authors"].apply(lambda author: author["homepage"] if isinstance(author, dict) else None)
    authors = authors.dropna(subset=["authorId"])
    authors = authors.drop_duplicates(subset=["authorId"])
    return authors

'''Builds dataframe for publications and venues'''
def build_publications_df(paper_details_df: pd.DataFrame) -> pd.DataFrame:
    publications = pd.DataFrame()
    publications["pub_name"] = paper_details_df.apply(lambda row: util.extract_publication_and_venue(row)[0], axis=1)
    publications["venue_name"] = paper_details_df.apply(lambda row: util.extract_publication_and_venue(row)[1], axis=1)
    publications["pub_year"] = paper_details_df["year"]
    publications["pub_date"] = paper_details_df["publicationDate"]
    publications["pub_vol"] = paper_details_df.apply(util.extract_publication_volume, axis=1)
    publications["pub_type"] = paper_details_df.apply(lambda row: "Proceedings" if row["publicationTypes"] and "Conference" in row["publicationTypes"] else "Journal" if row["publicationTypes"] else "", axis=1)
    publications = publications.dropna(subset=["pub_name"])
    publications = publications.drop_duplicates(subset="pub_name")
    return publications

def build_paper_details_df(paper_details_df: pd.DataFrame) -> pd.DataFrame:
    final_paper_details_df = pd.DataFrame()
    final_paper_details_df["paperId"] = paper_details_df["paperId"]
    final_paper_details_df["authors"] = paper_details_df["authors"].apply(lambda authors: [author["authorId"] for author in authors] if isinstance(authors, list) else [])
    final_paper_details_df["first_author"] = paper_details_df["authors"].apply(lambda authors: authors[0]["authorId"] if isinstance(authors, list) and len(authors) > 0 else None)
    final_paper_details_df["title"] = paper_details_df["title"].apply(util.sanitize_str)
    final_paper_details_df["abstract"] = paper_details_df["abstract"].apply(util.sanitize_str)
    final_paper_details_df["date"] = paper_details_df["publicationDate"]
    final_paper_details_df["keywords"] = paper_details_df["fieldsOfStudy"]# paper_details_df.apply(util.combine_keywords, axis=1) # Keywords cause long import time for later csv
    final_paper_details_df["references"] = paper_details_df["references"].apply(lambda refs: [ref["paperId"] for ref in refs] if isinstance(refs, list) else [])
    final_paper_details_df["pub_name"] = paper_details_df.apply(lambda row: util.extract_publication_and_venue(row)[0], axis=1)
    final_paper_details_df["venue_name"] = paper_details_df.apply(lambda row: util.extract_publication_and_venue(row)[1], axis=1)
    final_paper_details_df["pub_vol"] = paper_details_df.apply(util.extract_publication_volume, axis=1)
    final_paper_details_df["pub_type"] = paper_details_df.apply(lambda row: "Proceedings" if row["publicationTypes"] and "Conference" in row["publicationTypes"] else "Journal" if row["publicationTypes"] else "", axis=1)
    final_paper_details_df = final_paper_details_df.dropna(subset=["paperId"])
    return final_paper_details_df

def save_df_to_csv(df: pd.DataFrame, filename: str):
    save_path = os.path.join(cfg.PROJECT_PATH, "data", filename)
    df.to_csv(
        save_path,
        index=False,
        encoding="utf-8",
        quoting=csv.QUOTE_ALL,   # always quote fields
        doublequote=True,        # internal " -> ""
        lineterminator="\n",
    )
    print(f"Saved DataFrame to {save_path}")
