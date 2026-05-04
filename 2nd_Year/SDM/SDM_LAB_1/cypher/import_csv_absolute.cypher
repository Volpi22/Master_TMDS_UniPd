// Importing data from CSV files into Neo4j using Cypher
// Matches Schema defined in Lab 1 Report

// ============================================================================
// NODES
// ============================================================================

// 01 Load papers from CSV;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
MERGE (p:Paper {paperId: row.paperId}) // ID/DOI
    ON CREATE SET p.title = row.title,
              p.abstract = row.abstract,
              p.pages = row.pages,
              p.publicationDate =
                CASE
                  WHEN row.date IS NULL OR trim(row.date) = "" THEN NULL
                  ELSE date(trim(row.date))
                END;

// 02 Create keyword nodes from keyword list per paper;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
WITH row, split(replace(replace(replace(row.keywords, "[", ""), "]", ""), "'", ""), ",") AS keywordList
UNWIND keywordList AS keyword
WITH trim(keyword) AS cleanKeyword WHERE cleanKeyword <> ""
MERGE (k:Keyword {name: cleanKeyword});

// 03 Load authors from CSV;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/authors.csv' AS row
MERGE (a:Author {authorId: row.authorId})
    ON CREATE SET a.name = row.name,
                a.email = row.email;


// 04 Load Journals;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.venue_name IS NOT NULL AND row.pub_type = 'Journal'
MERGE (j:Journal {name: row.venue_name})
    ON CREATE SET j.issn = 'ISSN-' + substring(row.venue_name, 0, 8); // Synthetic ISSN

// 05 Load Journal Volumes;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.pub_vol IS NOT NULL AND row.pub_name is NOT NULL AND row.pub_type = 'Journal'
MERGE (jv:JournalVolume {journalVolumeId: row.pub_name + ', Volume ' + row.pub_vol})
    ON CREATE SET jv.volume = row.pub_vol,
                jv.year = toInteger(row.pub_year),
                jv.date = 
                CASE
                  WHEN row.date IS NULL OR trim(row.pub_date) = "" THEN NULL
                  ELSE date(trim(row.pub_date))
                END;

// 06 Load Conferences;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.pub_type = 'Proceedings' AND row.venue_type = 'Conference'
MERGE (c:Conference {conferenceId: row.venue_name});

// 07 Load Workshops;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.pub_type = 'Proceedings' AND row.venue_type = 'Workshop'
MERGE (w:Workshop {workshopId: row.venue_name});

// 08 Load Proceedings;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.pub_name IS NOT NULL AND row.pub_type = 'Proceedings'
MERGE (p:Proceedings {proceedingsId: row.pub_name + ', ' + row.pub_year})
    ON CREATE SET p.name = row.pub_name + ', ' + row.pub_year,
                p.isbn = row.isbn,
                p.year = toInteger(row.pub_year),
                p.date = 
                CASE
                  WHEN row.date IS NULL OR trim(row.pub_date) = "" THEN NULL
                  ELSE date(trim(row.pub_date))
                END;

// 09 Load Cities;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/affiliations.csv' AS row
MERGE (c:City {name: row.city});

// (10) Create institution nodes from affiliation list per author;
// LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/affiliations.csv' AS row
// MERGE (i:Institution {name: row.institution});

// RELATIONSHIPS

// 10 Load Paper-Paper relationships (references);
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
MATCH (p1:Paper {paperId: row.paperId})
WITH row, p1, split(replace(replace(replace(row.references, "[", ""), "]", ""), "'", ""), ",") AS referenceList
UNWIND referenceList AS refId
WITH p1, trim(refId) AS cleanRefId WHERE cleanRefId <> ""
MATCH (p2:Paper {paperId: cleanRefId})
MERGE (p1)-[:cites]->(p2);

// 11 ### Paper-Keyword relationships ###;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
MATCH (p:Paper {paperId: row.paperId})
WITH row, p, split(replace(replace(replace(row.keywords, "[", ""), "]", ""), "'", ""), ",") AS keywordList
UNWIND keywordList AS keyword
WITH p, trim(keyword) AS cleanKeyword WHERE cleanKeyword <> ""
MATCH (k:Keyword {name: cleanKeyword})
MERGE (p)-[:about]->(k);

// 12 ### Paper-Author relationships ###;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
MATCH (p:Paper {paperId: row.paperId})
WITH row, p, split(replace(replace(replace(row.authors, "[", ""), "]", ""), "'", ""), ",") AS authorList
UNWIND authorList AS authorId
WITH row, p, trim(authorId) AS cleanAuthorId WHERE cleanAuthorId <> ""
MATCH (a:Author {authorId: cleanAuthorId})
MERGE (a)-[r:writes]->(p)
    ON CREATE SET r.is_corr = (cleanAuthorId = row.first_author);

// ### Paper-Publication relationships ###
// 13 Paper published in Journal Volume;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
WITH row WHERE row.pub_type = 'Journal' AND row.pub_name IS NOT NULL
MATCH (p:Paper {paperId: row.paperId})
MATCH (v:JournalVolume {journalVolumeId: row.pub_name + ', Volume ' + row.pub_vol})
MERGE (p)-[:`published-in`]->(v);

// 14 Paper published in Proceedings;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/papers.csv' AS row
WITH row WHERE row.pub_type = 'Proceedings' AND row.pub_name IS NOT NULL AND row.date IS NOT NULL AND trim(row.date) <> ""
MATCH (p:Paper {paperId: row.paperId})
WITH p, row, substring(trim(row.date), 0, 4) AS pubYear
MATCH (proc:Proceedings {proceedingsId: row.pub_name + ', ' + pubYear})
MERGE (p)-[:`published-in`]->(proc);

// 15 Journal Volume part of Journal;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.pub_type = 'Journal' AND row.pub_name IS NOT NULL AND row.venue_name IS NOT NULL
MATCH (v:JournalVolume {journalVolumeId: row.pub_name + ', Volume ' + row.pub_vol})
MATCH (j:Journal {name: row.venue_name})
MERGE (v)-[:`part-of`]->(j);

// 16 Proceedings part of Conference / Workshop;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
WITH row WHERE row.pub_type = 'Proceedings' AND row.pub_name IS NOT NULL
MATCH (proc:Proceedings {proceedingsId: row.pub_name + ', ' + row.pub_year})
OPTIONAL MATCH (conf:Conference {conferenceId: trim(row.venue_name)})
OPTIONAL MATCH (work:Workshop {workshopId: trim(row.venue_name)})
WITH proc, coalesce(conf, work) AS venue WHERE venue IS NOT NULL
MERGE (proc)-[:`record-of`]->(venue);

// 17 Author reviews Paper;
LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/reviews.csv' AS row
WITH row WHERE row.reviewerId IS NOT NULL AND row.paperId IS NOT NULL
MATCH (a:Author {authorId: trim(row.reviewerId)})
MATCH (p:Paper {paperId: trim(row.paperId)})
MERGE (a)-[:reviews]->(p);

// 12 ### Author-Institution relationships ###;
// LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/authors.csv' AS row
// MATCH (a:Author {authorId: row.authorId})
// MATCH (i:Institution {name: row.affiliation})
// MERGE (a)-[:`affiliated_with`]->(i);

// // 19 Proceedings in City;
// LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/publications.csv' AS row
// WITH row WHERE row.pub_type = 'Proceedings' AND row.city IS NOT NULL
// MATCH (proc:Proceedings {proceedingsId: row.pub_name + ', ' + row.pub_year})
// MATCH (c:City {name: row.city})
// MERGE (proc)-[:`is-in`]->(c);

// 20 Insititution in City;
// LOAD CSV WITH HEADERS FROM 'file:///C:/Users/DavideVolpi/Desktop/smd-lab1-neo4j-teamduro/data/affiliations.csv' AS row
// MATCH (i:Institution {name: row.institution})
// MATCH (c:City {name: row.city})
// MERGE (i)-[:`is-in`]->(c);
