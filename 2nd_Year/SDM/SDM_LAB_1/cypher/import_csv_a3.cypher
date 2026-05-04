// Importing data from CSV files into Neo4j using Cypher
// Matches Schema defined in Lab 1 Report

// ============================================================================
// NODES
// ============================================================================

// 01 Create institution nodes from affiliation list per author;
LOAD CSV WITH HEADERS FROM 'file:///data/affiliations.csv' AS row
MERGE (i:Institution {name: row.institution});

// RELATIONSHIPS

// 02 Author-Institution relationships ;
LOAD CSV WITH HEADERS FROM 'file:///data/authors.csv' AS row
MATCH (a:Author {authorId: row.authorId})
MATCH (i:Institution {name: row.affiliation})
MERGE (a)-[:`affiliated_with`]->(i);

// 03 Institution in City;
LOAD CSV WITH HEADERS FROM 'file:///data/affiliations.csv' AS row
MATCH (i:Institution {name: row.institution})
MATCH (c:City {name: row.city})
MERGE (i)-[:`is-in`]->(c);

// 04 Author reviews Paper evolution;
LOAD CSV WITH HEADERS FROM 'file:///data/reviews.csv' AS row
MATCH (a:Author {authorId: row.reviewerId})
MATCH (p:Paper {paperId: row.paperId})
MERGE (a)-[r:reviews]->(p)
SET r.text = row.text, r.recommendation = row.state;