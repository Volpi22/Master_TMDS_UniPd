"""
synthetic_data.py

This module generates synthetic data to augment the real academic paper data fetched 
from the Semantic Scholar API. It ensures that the resulting graph database contains 
all necessary nodes, properties, and specific edge-case scenarios required to successfully 
execute the Part B and Part C Cypher queries for the lab assignment.
"""

import pandas as pd
import numpy as np
import random
import string
import config.conf as cfg

# --- Base Data Definitions for Synthetic Generation ---

# Predefined list of global cities for venues and affiliations
fake_cities = ["Barcelona", "New York", "Tokyo", "Berlin", "London", "Paris", "Rome", "Sydney", "Toronto"]

# Generate university names based on the cities
fake_universities = [f"University of {city}" for city in fake_cities]

# Generate corporate research labs by mapping tech giants to every city
tech_giants = ["Meta", "Amazon", "Nvidia", "Microsoft"]
fake_companies = [f"{company} {city}" for company in tech_giants for city in fake_cities]

# Combine both lists to form the complete pool of institutions
fake_institutions = fake_universities + fake_companies

# Define 5 distinct research communities and their specific domain keywords
communities = {
    "database": ["data management", "indexing", "data modeling", "big data", "data processing", "data storage", "data querying"],
    "machine_learning": ["deep learning", "neural networks", "reinforcement learning", "supervised learning", "natural language processing", "computer vision", "predictive modeling"],
    "cybersecurity": ["cryptography", "network security", "malware analysis", "intrusion detection", "access control", "vulnerability assessment", "blockchain"],
    "software_engineering": ["agile methodology", "continuous integration", "software architecture", "design patterns", "test-driven development", "code refactoring", "DevOps"],
    "bioinformatics": ["genomics", "computational biology", "sequence alignment", "protein folding", "transcriptomics", "systems biology", "molecular docking"]
}

# General academic keywords that can apply across any community
general_keywords = ["survey", "case study", "empirical analysis", "performance evaluation", "methodology", "future trends", "literature review", "optimization"]


def add_synthetic_keywords(paper_details_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns 4 random keywords to each paper (3 from a specific community, 1 general).
    To satisfy lab constraints, it ensures that venues have a >90% concentration 
    of a single specific community's keywords.
    """
    
    # 1. Map every unique publication venue to a single "dominant" community
    unique_pubs = paper_details_df['pub_name'].dropna().unique()
    pub_to_community = {pub: random.choice(list(communities.keys())) for pub in unique_pubs}

    def generate_keywords_for_paper(pub_name):
        # Determine the community for this paper based on its venue
        if pd.notnull(pub_name) and pub_name in pub_to_community:
            dominant_community = pub_to_community[pub_name]
            
            # 95% of the time, the paper matches the venue's dominant community.
            # This safely clears the 90% threshold required for Part C queries.
            if random.random() < 0.95:
                chosen_community = dominant_community
            else:
                # 5% of the time, simulate an outlier paper from a different community
                other_communities = [c for c in communities.keys() if c != dominant_community]
                chosen_community = random.choice(other_communities)
        else:
            # Fallback if publication name is missing
            chosen_community = random.choice(list(communities.keys()))
            
        # Select 3 domain-specific keywords and 1 general keyword
        domain_keywords = random.sample(communities[chosen_community], 3)
        general_keyword = random.sample(general_keywords, 1)
        
        return domain_keywords + general_keyword

    # Apply the keyword generation logic to every paper
    paper_details_df['keywords'] = paper_details_df['pub_name'].apply(generate_keywords_for_paper)
    return paper_details_df


def add_synthetic_node_properties(authors_df: pd.DataFrame, paper_details_df: pd.DataFrame, publications_df: pd.DataFrame) -> tuple:
    """Adds missing synthetic properties such as emails, page numbers, affiliations, and ISBNs."""
    
    # 1. Generate fake emails for authors based on their names
    authors_df['email'] = authors_df['name'].apply(
        lambda x: f"{str(x).replace(' ', '').lower()}@example.edu" if pd.notnull(x) else "unknown@example.edu"
    )
    
    # 2. Assign random page ranges to papers (e.g., "15-130")
    n_papers = len(paper_details_df)
    paper_details_df['pages'] = [f"{np.random.randint(1, 100)}-{np.random.randint(101, 200)}" for _ in range(n_papers)]
    
    # 3. Assign 9-digit fake ISBNs to Proceedings
    proceedings_mask = publications_df['pub_type'] == 'Proceedings'
    n_proceedings = proceedings_mask.sum()
    publications_df.loc[proceedings_mask, 'isbn'] = [
        f"978-{np.random.randint(100000000, 999999999)}" for _ in range(n_proceedings)
    ]

    # 4. Randomly assign each author to one of the predefined synthetic institutions
    authors_df['affiliation'] = np.random.choice(fake_institutions, size=len(authors_df))

    return authors_df, paper_details_df, publications_df


def generate_cities_and_editions(publications_df: pd.DataFrame, paper_df: pd.DataFrame) -> tuple:
    """Assigns locations/editions to proceedings and builds a synthetic citation network."""
    
    proceedings_mask = publications_df['pub_type'] == 'Proceedings'
    n_proceedings = proceedings_mask.sum()
    
    # Randomly assign host cities and edition numbers to proceedings
    publications_df.loc[proceedings_mask, 'city'] = np.random.choice(fake_cities, size=n_proceedings)
    publications_df.loc[proceedings_mask, 'edition_number'] = np.random.randint(cfg.START_YEAR, cfg.START_YEAR + cfg.N_YEARS_TO_FETCH, size=n_proceedings)
    
    # Randomly classify proceedings as either Conferences (80%) or Workshops (20%)
    venue_types = ['Conference'] * 80 + ['Workshop'] * 20
    publications_df.loc[proceedings_mask, 'venue_type'] = np.random.choice(venue_types, size=n_proceedings)

    # Extract all valid paper IDs to pool from for references
    paper_ids_set = set(paper_df['paperId'].dropna().unique())
    
    def get_random_references(current_paper_id):
        # Ensure a paper does not cite itself
        eligible_refs = list(paper_ids_set - {current_paper_id})
        # Pick between 1 and 10 random references
        num_refs = random.randint(1, min(10, len(eligible_refs)))
        return random.sample(eligible_refs, num_refs) if num_refs > 0 else []

    # Apply the citation generation logic to every paper
    paper_df['references'] = paper_df['paperId'].apply(get_random_references)
    
    return publications_df, paper_df


def generate_reviewers(paper_details_df: pd.DataFrame, authors_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns up to 3 random reviewers per paper, enforcing the rule that authors cannot review their own work."""
    
    all_author_ids_set = set(authors_df['authorId'].dropna().unique())
    reviews_data = []

    for _, row in paper_details_df.iterrows():
        paper_id = row['paperId']
        actual_authors = row['authors'] if isinstance(row['authors'], list) else []
        
        # Remove the paper's actual authors from the pool of eligible reviewers
        eligible_reviewers = list(all_author_ids_set - set(actual_authors))
        
        # Pick up to 3 reviewers
        num_reviewers = min(3, len(eligible_reviewers))
        if num_reviewers > 0:
            selected_reviewers = random.sample(eligible_reviewers, num_reviewers)
            
            for reviewer_id in selected_reviewers:
                # Generate a random 50-character string to simulate the review text
                random_text = ''.join(random.choices(string.ascii_letters + ' ', k=50)).strip()
                # Randomly suggest accepting or rejecting the paper
                review_state = random.choice(['accepted', 'rejected'])
                
                reviews_data.append({
                    'paperId': paper_id,
                    'reviewerId': reviewer_id,
                    'text': random_text,
                    'state': review_state
                })
                
    return pd.DataFrame(reviews_data)


def generate_city_university_mapping() -> pd.DataFrame:
    """Creates a mapping table linking institutions to their host cities and type (University/Company)."""
    data = []
    
    # Process academic institutions
    for institution in fake_universities:
        city = institution.split(" of ")[-1]
        data.append({'institution': institution, 'city': city, 'type': 'University'})
        
    # Process corporate institutions
    for institution in fake_companies:
        # Dynamically match the city string from the end of the company name
        city_found = next((city for city in fake_cities if institution.endswith(city)), "Unknown")
        data.append({'institution': institution, 'city': city_found, 'type': 'Company'})
        
    return pd.DataFrame(data)


def enforce_lab_guarantees(authors_df: pd.DataFrame, paper_details_df: pd.DataFrame, publications_df: pd.DataFrame) -> tuple:
    """
    Forcefully injects specific edge-cases into the dataset to guarantee that the
    Part B and Part C Cypher queries retrieve at least one result.
    """
    
    # ---------------------------------------------------------
    # 1. GUARANTEE B.3: The Impact Factor
    # (Forces Journal papers in 2021 to be cited by papers in 2022)
    # ---------------------------------------------------------
    journal_mask = publications_df['pub_type'] == 'Journal'
    n_journals = journal_mask.sum()
    if n_journals > 0:
        # Assign random volumes to all journals just to be safe
        publications_df.loc[journal_mask, 'pub_vol'] = np.random.randint(1, 50, size=n_journals).astype(str)
        
    journal_papers = paper_details_df[paper_details_df['pub_type'] == 'Journal'].head(10).index
    if len(journal_papers) >= 10:
        target_journal = "Journal of Guaranteed Impact"
        
        # Force 2 published papers in 2021
        pub_paper_1, pub_paper_2 = journal_papers[0], journal_papers[1]
        paper_details_df.at[pub_paper_1, 'date'] = "2021-05-01"
        paper_details_df.at[pub_paper_1, 'year'] = 2021
        paper_details_df.at[pub_paper_2, 'date'] = "2021-08-01"
        paper_details_df.at[pub_paper_2, 'year'] = 2021
        
        for idx in [pub_paper_1, pub_paper_2]:
            paper_details_df.at[idx, 'pub_name'] = target_journal
            paper_details_df.at[idx, 'venue_name'] = target_journal
            paper_details_df.at[idx, 'pub_vol'] = "Volume 1"

        # Add the Journal Volume to publications_df
        new_vol = pd.DataFrame([{
            'pub_name': target_journal,
            'venue_name': target_journal,
            'pub_year': 2021,
            'pub_date': "2021-01-01",
            'pub_type': 'Journal',
            'pub_vol': 'Volume 1'
        }])
        publications_df = pd.concat([publications_df, new_vol], ignore_index=True)

        # Force 8 citing papers in 2022
        for idx in journal_papers[2:10]:
            paper_details_df.at[idx, 'date'] = "2022-03-01"
            paper_details_df.at[idx, 'year'] = 2022
            refs = paper_details_df.at[idx, 'references']
            if isinstance(refs, list):
                refs.extend([paper_details_df.at[pub_paper_1, 'paperId'], paper_details_df.at[pub_paper_2, 'paperId']])
            else:
                paper_details_df.at[idx, 'references'] = [paper_details_df.at[pub_paper_1, 'paperId'], paper_details_df.at[pub_paper_2, 'paperId']]

    # ---------------------------------------------------------
    # 2. GUARANTEE B.2: The Loyal Author
    # (Author publishing in >=4 different editions of the same venue)
    # ---------------------------------------------------------
    loyal_author_id = "999999999"
    new_author = pd.DataFrame([{"authorId": loyal_author_id, "name": "The Loyal Author", "email": "loyal@example.edu", "affiliation": "University of Neo4j"}])
    authors_df = pd.concat([authors_df, new_author], ignore_index=True)

    target_venue = "The Summit of Graph Databases"
    years = [2018, 2019, 2020, 2021]
    
    proc_papers = paper_details_df[paper_details_df['pub_type'] == 'Proceedings'].head(4).index
    if len(proc_papers) == 4:
        for i, idx in enumerate(proc_papers):
            paper_details_df.at[idx, 'venue_name'] = target_venue
            # Force 4 distinctly different editions/proceedings nodes
            paper_details_df.at[idx, 'pub_name'] = f"Proceedings of {target_venue} {years[i]}"
            
            paper_details_df.at[idx, 'date'] = f"{years[i]}-06-01" 
            paper_details_df.at[idx, 'year'] = years[i]

            current_authors = paper_details_df.at[idx, 'authors']
            if isinstance(current_authors, list):
                current_authors.append(loyal_author_id)
            else:
                paper_details_df.at[idx, 'authors'] = [loyal_author_id]

            # Ensure the publication node exists
            new_pub = pd.DataFrame([{
                'pub_name': f"Proceedings of {target_venue} {years[i]}",
                'venue_name': target_venue,
                'pub_year': years[i],
                'pub_date': f"{years[i]}-06-01",
                'pub_type': 'Proceedings',
                'venue_type': 'Conference',
                'city': 'Barcelona',
                'edition_number': i+1
            }])
            publications_df = pd.concat([publications_df, new_pub], ignore_index=True)

    # ---------------------------------------------------------
    # 3. GUARANTEE C.4: The Database Guru
    # (Author with >=2 papers in the Top-100 Database papers)
    # ---------------------------------------------------------
    guru_venue = "International Summit of DB Gurus"
    
    # Grab 2 arbitrary papers to become our Guaranteed Top Papers
    db_papers_idx = paper_details_df.head(2).index.tolist()
    target_pids = [paper_details_df.at[idx, 'paperId'] for idx in db_papers_idx]
    
    # Select our Guru Author (e.g., the second author in the DB)
    guru_author = authors_df['authorId'].iloc[1]
    
    # Force these 2 papers to be perfectly valid Database papers in a pure DB venue
    for idx in db_papers_idx:
        # Force them into a 100% pure DB venue so it passes C.2
        paper_details_df.at[idx, 'pub_name'] = f"Proceedings of {guru_venue}"
        paper_details_df.at[idx, 'venue_name'] = guru_venue
        paper_details_df.at[idx, 'pub_type'] = 'Proceedings'
        paper_details_df.at[idx, 'date'] = '2020-01-01' 
        paper_details_df.at[idx, 'year'] = 2020
        
        # Force their keywords to be Database keywords so they pass C.1
        paper_details_df.at[idx, 'keywords'] = random.sample(communities["database"], 3)
        
        # Assign the Guru author
        current_authors = paper_details_df.at[idx, 'authors']
        if isinstance(current_authors, list):
            if guru_author not in current_authors:
                current_authors.append(guru_author)
        else:
            paper_details_df.at[idx, 'authors'] = [guru_author]

    # Add the Venue to publications_df so it exists in the graph
    new_guru_pub = pd.DataFrame([{
        'pub_name': f"Proceedings of {guru_venue}",
        'venue_name': guru_venue,
        'pub_year': 2020,
        'pub_date': "2020-01-01",
        'pub_type': 'Proceedings',
        'venue_type': 'Conference',
        'city': 'Rome',
        'edition_number': 1
    }])
    publications_df = pd.concat([publications_df, new_guru_pub], ignore_index=True)

    # Now make 50 OTHER papers cite these 2 papers
    citing_papers_idx = paper_details_df.tail(50).index.tolist()
    
    for idx in citing_papers_idx:
        # Force citing papers to have a DB keyword so C.3 matches them
        paper_details_df.at[idx, 'keywords'] = random.sample(communities["database"], 1) + ["survey"]
        
        refs = paper_details_df.at[idx, 'references']
        if isinstance(refs, list):
            for pid in target_pids:
                if pid not in refs and paper_details_df.at[idx, 'paperId'] != pid:
                    refs.append(pid)
        else:
            paper_details_df.at[idx, 'references'] = target_pids

    return authors_df, paper_details_df, publications_df