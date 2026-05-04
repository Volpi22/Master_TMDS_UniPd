// 01 Find the top 3 most cited papers of each conference/workshop.
MATCH (p:Paper)-[:`published-in`]->(proc:Proceedings)-[:`record-of`]->(venue)
WHERE venue:Conference OR venue:Workshop
OPTIONAL MATCH (citing_paper:Paper)-[:cites]->(p)
WITH venue, p, COUNT(citing_paper) AS citations
ORDER BY coalesce(venue.conferenceId, venue.workshopId), citations DESC
WITH venue, COLLECT(p)[0..3] AS top_papers
RETURN coalesce(venue.conferenceId, venue.workshopId) AS venue, top_papers;

// 02 For each conference/workshop find its community: i.e., those authors that have 
//    published papers on that conference/workshop in, at least, 4 different editions.
MATCH (a:Author)-[:writes]->(p:Paper)-[:`published-in`]->(proc:Proceedings)-[:`record-of`]->(venue)
WHERE venue:Conference OR venue:Workshop
WITH venue, a, COUNT(DISTINCT proc) AS distinct_editions
WHERE distinct_editions >= 4
RETURN coalesce(venue.conferenceId, venue.workshopId) AS venue, COLLECT(a.name) AS community;

// 03 Find the impact factor of the journals in your graph.
MATCH (j:Journal)<-[:`part-of`]-(v:JournalVolume)<-[:`published-in`]-(p:Paper)
WITH j, p, v, 2022 AS target_year
WHERE v.year = target_year - 1 OR v.year = target_year - 2
WITH j, target_year, COUNT(p) AS citable_items, COLLECT(p) AS prev_papers
WHERE citable_items > 0
UNWIND prev_papers AS target_paper
OPTIONAL MATCH (citing_paper:Paper)-[:cites]->(target_paper)
WHERE citing_paper.publicationDate.year = target_year
WITH j, citable_items, COUNT(citing_paper) AS citations
RETURN j.name AS Journal, toFloat(citations)/citable_items AS ImpactFactor
ORDER BY ImpactFactor DESC;

// 04 Find the h-index of the authors in your graph.
MATCH (a:Author)-[:writes]->(p:Paper)
OPTIONAL MATCH (p)<-[:cites]-(citing_paper:Paper)
WITH a, p, count(citing_paper) AS citations
ORDER BY citations DESC
WITH a, collect(citations) AS citationList
RETURN a.name AS Author, 
       a.authorId AS AuthorID,
       size(citationList) AS TotalPapers,
       citationList AS CitationArray,
       size([i IN range(1, size(citationList)) WHERE citationList[i-1] >= i]) AS h_index
ORDER BY h_index DESC, TotalPapers DESC;