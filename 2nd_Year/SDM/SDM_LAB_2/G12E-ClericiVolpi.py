import random
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

def generate_knowledge_graph():
    # Initialize a new graph
    g = Graph()

    # Define a custom namespace for our biomedical data
    EX = Namespace("http://example.org/biomed#")
    g.bind("ex", EX)

    print("Building schema...")
    # ==========================================
    # 1. BUILD THE METAMODEL & MODEL LAYER
    # ==========================================
    
    # Base Classes
    g.add((EX.Drug, RDF.type, RDFS.Class))
    g.add((EX.Disease, RDF.type, RDFS.Class))

    # Extended Subclasses (Added Antiviral and ViralInfection to enrich the graph)
    drug_classes = [EX.Antibiotic, EX.AntiInflammatory, EX.Antiviral]
    for cls in drug_classes:
        g.add((cls, RDFS.subClassOf, EX.Drug))

    disease_classes = [EX.InflammatoryDisease, EX.InfectiousDisease, EX.ViralInfection]
    for cls in disease_classes:
        g.add((cls, RDFS.subClassOf, EX.Disease))

    # Base Property (with explicit Domain and Range)
    g.add((EX.affects, RDF.type, RDF.Property))
    g.add((EX.affects, RDFS.domain, EX.Drug))
    g.add((EX.affects, RDFS.range, EX.Disease))

    # Subproperties (Inherit Domain and Range automatically)
    subproperties = [EX.treats, EX.relieves, EX.worsens]
    for prop in subproperties:
        g.add((prop, RDFS.subPropertyOf, EX.affects))


    print("Inserting base instances...")
    # ==========================================
    # 2. ADD MANDATORY BASE INSTANCES (Facts)
    # ==========================================
    
    # Assign Types to base instances
    g.add((EX.Ibuprofen, RDF.type, EX.AntiInflammatory))
    g.add((EX.Amoxicillin, RDF.type, EX.Antibiotic))
    g.add((EX.Arthritis, RDF.type, EX.InflammatoryDisease))
    g.add((EX.GastricUlcer, RDF.type, EX.Disease))
    g.add((EX.BacterialInfection, RDF.type, EX.InfectiousDisease))

    # Assert Base Relationships
    g.add((EX.Ibuprofen, EX.relieves, EX.Arthritis))
    g.add((EX.Ibuprofen, EX.worsens, EX.GastricUlcer))
    g.add((EX.Amoxicillin, EX.treats, EX.BacterialInfection))


    print("Generating ~100 dummy instances and relationships...")
    # ==========================================
    # 3. GENERATE EXTENDED INSTANCES & FACTS
    # ==========================================
    
    # Set seed for reproducible graphs
    random.seed(42) 
    
    dummy_drugs = []
    dummy_diseases = []

    # Generate 50 dummy drugs
    for i in range(1, 51):
        drug_uri = URIRef(f"http://example.org/biomed#Drug_{i}")
        assigned_class = random.choice(drug_classes)
        g.add((drug_uri, RDF.type, assigned_class))
        g.add((drug_uri, RDFS.label, Literal(f"Experimental Drug {i}")))
        dummy_drugs.append(drug_uri)

    # Generate 50 dummy diseases
    for i in range(1, 51):
        disease_uri = URIRef(f"http://example.org/biomed#Disease_{i}")
        assigned_class = random.choice(disease_classes)
        g.add((disease_uri, RDF.type, assigned_class))
        g.add((disease_uri, RDFS.label, Literal(f"Condition {i}")))
        dummy_diseases.append(disease_uri)

    # Generate approximately 100 random relationships between the dummy instances
    for drug in dummy_drugs:
        # Give each drug 1 to 3 random effects on diseases
        num_effects = random.randint(1, 3) 
        target_diseases = random.sample(dummy_diseases, num_effects)
        
        for disease in target_diseases:
            assigned_effect = random.choice(subproperties)
            g.add((drug, assigned_effect, disease))

    # ==========================================
    # 4. EXPORT TO TURTLE FILE (.ttl)
    # ==========================================
    
    output_filename = "G12E-ClericiVolpi.ttl" 
    
    print(f"Serializing graph to {output_filename}...")
    g.serialize(destination=output_filename, format="turtle")
    
    total_triples = len(g)
    print(f"Success! Generated {total_triples} triples (excluding inferred data).")

if __name__ == "__main__":
    generate_knowledge_graph()