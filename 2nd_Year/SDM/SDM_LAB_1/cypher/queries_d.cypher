// Calls to the Graph Data Science Library

// 1. Node Similarity:
// First store the graph in the GDS catalog (if not already done)
CALL gds.graph.project('paperKeywordGraph', ['Paper', 'Keyword'], 'about');

// Then compute similarity between papers based on shared keywords using Jaccard similarity
CALL gds.nodeSimilarity.stream(
  'paperKeywordGraph',
  {
    similarityMetric: 'JACCARD'
  }
) 
YIELD node1, node2, similarity
WITH gds.util.asNode(node1) AS Paper1, gds.util.asNode(node2) AS Paper2, similarity
MATCH (p1:Paper {paperId: Paper1.paperId})-[:about]->(k:Keyword)<-[:about]-(p2:Paper {paperId: Paper2.paperId})
RETURN Paper1.title AS Paper1, Paper2.title AS Paper2, similarity, COLLECT(DISTINCT k.name) AS SharedKeywords
ORDER BY similarity DESC;



// 2. Djikstra's Shortest Path
// First store the graph in the GDS catalog (if not already done)
CALL gds.graph.project('citation_graph', 'Paper', 'cites');

// Djikstra's Shortest Path: Trace lineage between two dynamically selected papers
MATCH path = (source:Paper)-[:cites*3..5]->(target:Paper)
WITH source, target LIMIT 1

CALL gds.shortestPath.dijkstra.stream('citation_graph', {
    sourceNode: source,
    targetNodes: target
})
YIELD index, sourceNode, targetNode, totalCost, nodeIds
RETURN 
  index, 
  gds.util.asNode(sourceNode).title AS SourcePaper,
  gds.util.asNode(targetNode).title AS TargetPaper,
  totalCost AS Hops,
  [nodeId IN nodeIds | gds.util.asNode(nodeId).title] AS CitationChain
ORDER BY index;
