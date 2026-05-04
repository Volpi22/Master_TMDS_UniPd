// C.1: Create Database Community Node and relationships with Keywords
CREATE (c:Community {name: 'Database'})
WITH c
MATCH (k:Keyword)
WHERE k.name IN ['data management', 'indexing', 'data modeling', 'big data', 'data processing', 'data storage', 'data querying']
MERGE (c)-[:focuses_on]->(k); 

// C.2: Find venues relevant to the Database community and create relationships
MATCH (v:Workshop|Conference|Journal)<-[:`record-of`|`part-of`]-()<-[:`published-in`]-(dp:Paper)-[:about]->(k:Keyword)<-[:focuses_on]-(c:Community {name: 'Database'})
WITH c, v, COUNT(DISTINCT dp) AS database_papers
MATCH (v)<-[:`record-of`|`part-of`]-()<-[:`published-in`]-(tp:Paper)
WITH c, v, database_papers, COUNT(DISTINCT tp) AS total_papers
WHERE database_papers >= 0.9 * total_papers
MERGE (v)-[:`relevant_to`]->(c);

// C.3: Find top 100 most cited papers relevant to the Database community and create relationships
MATCH (p:Paper)-[:`published-in`]->()-[:`record-of`|`part-of`]->(v:Workshop|Conference|Journal)-[:`relevant_to`]->(c:Community {name: 'Database'})
WITH p, c
MATCH (c)-[:`focuses_on`]->(k:Keyword)<-[:`about`]-(op:Paper)-[:cites]->(p)
WITH p, c, COUNT(DISTINCT op) AS citations
ORDER BY citations DESC LIMIT 100
MERGE (c)-[:`top_paper`]->(p);

// C.4: Find authors of top papers and create recommended reviewer relationships with guru status
MATCH (a:Author)-[:writes]->(p:Paper)<-[:`top_paper`]-(c:Community {name: 'Database'})
WITH a, c, COUNT(DISTINCT p) AS top_papers_count
MERGE (c)-[e:`recommended_reviewer`]->(a)
    ON CREATE SET e.is_guru = (top_papers_count >= 2);
