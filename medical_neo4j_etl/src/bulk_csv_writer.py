# medical_neo4j_etl/src/bulk_csv_writer.py
from neo4j import GraphDatabase
import csv, os

uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
if not uri or not username or not password:
    raise RuntimeError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD must be set")
driver = GraphDatabase.driver(uri, auth=(username, password))
DATA_PATH = "./data"


def _get_canonical_disease_ids() -> set[str]:
    path = os.path.join(DATA_PATH, "disease_precaution.csv")
    canonical_ids: set[str] = set()
    with open(path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            disease_id = (row.get("disease_id") or "").strip()
            if disease_id:
                canonical_ids.add(disease_id)
    return canonical_ids

def load_nodes(tx, file_name, label, extra_fields=None, include_ids=None):
    path = os.path.join(DATA_PATH, file_name)
    with open(path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if include_ids is not None and row.get("id") not in include_ids:
                continue
            props = {k: row[k] for k in reader.fieldnames if k != "id"}
            props_str = ", ".join([f"n.{k} = ${k}" for k in props.keys()])
            query = f"MERGE (n:{label} {{id:$id}}) SET {props_str}"
            tx.run(query, **row)

def load_relationships(tx, file_name, start_label, rel_type, end_label,
                       start_col, end_col):
    path = os.path.join(DATA_PATH, file_name)
    with open(path, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            query = f"""
            MATCH (a:{start_label} {{id:$start}}), (b:{end_label} {{id:$end}})
            MERGE (a)-[:{rel_type}]->(b)
            """
            tx.run(query, start=row[start_col], end=row[end_col])

def main():
    canonical_disease_ids = _get_canonical_disease_ids()

    with driver.session() as session:
        # create nodes
        session.execute_write(
            load_nodes,
            "diseases.csv",
            "Disease",
            None,
            canonical_disease_ids,
        )
        session.execute_write(load_nodes, "symptoms.csv", "Symptom")
        session.execute_write(load_nodes, "precautions.csv", "Precaution")
        # relationships
        session.execute_write(load_relationships, "disease_symptom.csv",
                              "Disease", "HAS_SYMPTOM", "Symptom",
                              "disease_id", "symptom_id")
        session.execute_write(load_relationships, "disease_precaution.csv",
                              "Disease", "HAS_PRECAUTION", "Precaution",
                              "disease_id", "precaution_id")
    print("✅ Data loaded into Neo4j successfully.")

if __name__ == "__main__":
    main()
