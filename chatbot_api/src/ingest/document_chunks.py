import os
from typing import Any, Dict, List

from neo4j import GraphDatabase


def split_text_into_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    text = text.strip()

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + chunk_size - overlap, start + 1)

    return chunks


def insert_document_chunks(filename: str, text: str, structured: Dict[str, Any]) -> Dict[str, int]:
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not username or not password:
        raise RuntimeError("NEO4J connection environment variables not set")

    chunks = split_text_into_chunks(text)
    if not chunks:
        return {"document_chunks": 0, "mentions_links": 0}

    diseases = []
    for disease in structured.get("diseases", []) if isinstance(structured, dict) else []:
        name = (disease.get("name") or "").strip() if isinstance(disease, dict) else ""
        if name:
            diseases.append(name)

    driver = GraphDatabase.driver(uri, auth=(username, password))
    links = 0

    def _write(tx, file_name: str, chunk_index: int, chunk_text: str, disease_names: List[str]):
        tx.run(
            """
            MERGE (dc:DocumentChunk {source_filename: $file_name, chunk_index: $chunk_index})
            SET dc.text = $chunk_text,
                dc.source_filename = $file_name,
                dc.chunk_index = $chunk_index
            """,
            file_name=file_name,
            chunk_index=chunk_index,
            chunk_text=chunk_text,
        )

        for disease_name in disease_names:
            tx.run(
                """
                MATCH (dc:DocumentChunk {source_filename: $file_name, chunk_index: $chunk_index})
                MATCH (d:Disease {name: $disease_name})
                MERGE (dc)-[:MENTIONS]->(d)
                """,
                file_name=file_name,
                chunk_index=chunk_index,
                disease_name=disease_name,
            )

    with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
        for index, chunk in enumerate(chunks):
            mention_candidates = [d for d in diseases if d.lower() in chunk.lower()]
            session.execute_write(_write, filename, index, chunk, mention_candidates)
            links += len(mention_candidates)

    driver.close()
    return {"document_chunks": len(chunks), "mentions_links": links}
