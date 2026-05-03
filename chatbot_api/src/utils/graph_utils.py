"""Graph utilities for retrieving Neo4j graph data."""
import os
from typing import Dict, List, Any
from langchain_community.graphs import Neo4jGraph

DISEASE_LABEL = os.getenv("GRAPH_DISEASE_LABEL", "Disease")
NAME_PROP = os.getenv("GRAPH_NAME_PROP", "name")
FALLBACK_CYPHER = f"""
MATCH (d:{DISEASE_LABEL})-[r]-(n)
WHERE toLower(d.{NAME_PROP}) CONTAINS toLower($keyword)
RETURN d.{NAME_PROP} as entity, type(r) as relationship,
       n.{NAME_PROP} as related LIMIT 8
"""


def get_disease_graph_data(disease_name: str) -> Dict[str, Any]:
    """Retrieve graph data (nodes and edges) for a specific disease.
    
    Returns a dictionary with:
    - nodes: list of node objects with id, label, and properties
    - edges: list of edge objects with source, target, and type
    - disease_found: boolean indicating if the disease exists
    """
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )
    
    # Query to get disease and all related nodes
    query = """
    MATCH (d:Disease {name: $disease_name})
    OPTIONAL MATCH (d)-[r1:HAS_SYMPTOM]->(s:Symptom)
    OPTIONAL MATCH (d)-[r2:HAS_PRECAUTION]->(p:Precaution)
    RETURN d, 
           collect(DISTINCT s) as symptoms, 
           collect(DISTINCT p) as precautions,
           collect(DISTINCT r1) as symptom_rels,
           collect(DISTINCT r2) as precaution_rels
    """
    
    try:
        result = graph.query(query, params={"disease_name": disease_name})
        
        if not result or not result[0].get('d'):
            return {
                "disease_found": False,
                "nodes": [],
                "edges": [],
                "message": f"Disease '{disease_name}' not found in database"
            }
        
        nodes = []
        edges = []
        node_ids = set()
        
        # Extract disease node
        disease_node = result[0]['d']
        disease_id = f"disease_{disease_node.get('name', disease_name)}"
        nodes.append({
            "id": disease_id,
            "label": "Disease",
            "name": disease_node.get('name', disease_name),
            "properties": dict(disease_node)
        })
        node_ids.add(disease_id)
        
        # Extract symptom nodes and edges
        symptoms = result[0].get('symptoms', [])
        for idx, symptom in enumerate(symptoms):
            if symptom:  # Check if symptom is not None
                symptom_id = f"symptom_{symptom.get('name', f'symptom_{idx}')}"
                if symptom_id not in node_ids:
                    nodes.append({
                        "id": symptom_id,
                        "label": "Symptom",
                        "name": symptom.get('name', f'Symptom {idx}'),
                        "properties": dict(symptom)
                    })
                    node_ids.add(symptom_id)
                
                edges.append({
                    "source": disease_id,
                    "target": symptom_id,
                    "type": "HAS_SYMPTOM"
                })
        
        # Extract precaution nodes and edges
        precautions = result[0].get('precautions', [])
        for idx, precaution in enumerate(precautions):
            if precaution:  # Check if precaution is not None
                precaution_text = precaution.get('name', precaution.get('text', f'Precaution {idx}'))
                precaution_id = f"precaution_{idx}_{precaution_text[:20]}"
                if precaution_id not in node_ids:
                    nodes.append({
                        "id": precaution_id,
                        "label": "Precaution",
                        "name": precaution_text,
                        "properties": dict(precaution)
                    })
                    node_ids.add(precaution_id)
                
                edges.append({
                    "source": disease_id,
                    "target": precaution_id,
                    "type": "HAS_PRECAUTION"
                })
        
        return {
            "disease_found": True,
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
        
    except Exception as e:
        return {
            "disease_found": False,
            "nodes": [],
            "edges": [],
            "error": str(e),
            "message": f"Error retrieving graph data: {str(e)}"
        }
