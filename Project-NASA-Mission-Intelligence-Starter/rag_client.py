__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import Dict, List, Optional
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared embedding helper
# ---------------------------------------------------------------------------

def _get_openai_client() -> OpenAI:
    """Return an OpenAI client pointed at the Vocareum proxy."""
    api_key = os.environ.get("CHROMA_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    return OpenAI(
        api_key=api_key,
        base_url="https://openai.vocareum.com/v1"
    )

def _embed(query: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Produce a single query embedding using the same OpenAI model and endpoint
    that was used when building the ChromaDB collection.

    Using query_embeddings= instead of query_texts= bypasses Chroma's default
    local embedding model (all-MiniLM-L6-v2, 384 dims) and ensures the vector
    dimensions match the stored embeddings (text-embedding-3-small, 1536 dims).
    """
    client = _get_openai_client()
    response = client.embeddings.create(input=[query], model=model)
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories that are directories and match "chroma" naming pattern
    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        try:
            # Initialize database client — no embedding_function so Chroma does
            # not attach its default local model to the collection handle
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}/{collection.name}"

                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = 0

                backends[key] = {
                    "chroma_dir": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} › {collection.name} ({doc_count} docs)",
                    "doc_count": str(doc_count),
                }

        except Exception as e:
            key = chroma_dir.name
            backends[key] = {
                "chroma_dir": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (error: {str(e)[:50]})",
                "doc_count": "0",
            }

    return backends


# ---------------------------------------------------------------------------
# RAG system initialisation
# ---------------------------------------------------------------------------

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """
    Load the ChromaDB collection WITHOUT attaching an embedding function.

    Embeddings are produced manually (via _embed()) so that the same
    OpenAI / Vocareum model and dimensionality are used for both indexing
    and querying.
    """
    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )

    # get_collection with no embedding_function — queries will use
    # query_embeddings= instead of query_texts=
    return client.get_collection(name=collection_name)


# ---------------------------------------------------------------------------
# Document retrieval
# ---------------------------------------------------------------------------

def retrieve_documents(
    collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
    embedding_model: str = "text-embedding-3-small"
) -> Optional[Dict]:
    """
    Retrieve relevant documents from ChromaDB with optional mission filtering.

    The query is embedded manually with the same OpenAI model used at index time
    (text-embedding-3-small, 1536 dims). Passing query_embeddings= instead of
    query_texts= prevents Chroma from using its local model (384 dims), which
    would cause a dimension mismatch error.
    """
    # Build optional metadata filter
    where_filter = None
    if mission_filter and mission_filter.lower() not in ("all", "all missions", ""):
        where_filter = {"mission": mission_filter}

    # Embed the query with OpenAI so dimensions match the stored vectors
    query_embedding = _embed(query, model=embedding_model)

    # Query using pre-computed embedding, not query_texts
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

    return results


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into a structured context block."""
    if not documents:
        return ""

    context_parts = ["=== RELEVANT CONTEXT FROM NASA MISSION DOCUMENTS ===\n"]

    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        mission = meta.get("mission", "unknown").replace("_", " ").title()
        source = meta.get("source", "unknown")
        category = meta.get("document_category", "unknown").replace("_", " ").title()

        source_header = f"[{idx}] Mission: {mission} | Source: {source} | Category: {category}"
        context_parts.append(source_header)
        context_parts.append("-" * len(source_header))

        max_doc_length = 1500
        truncated_doc = doc[:max_doc_length] + "... [truncated]" if len(doc) > max_doc_length else doc
        context_parts.append(truncated_doc)
        context_parts.append("")

    return "\n".join(context_parts)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    backends = discover_chroma_backends()
    print("Discovered backends:", backends)

    if not backends:
        print("No Chroma backends found.")
        sys.exit(0)

    first_backend = next(iter(backends.values()))
    collection = initialize_rag_system(
        first_backend["chroma_dir"],
        first_backend["collection_name"]
    )

    results = retrieve_documents(
        collection,
        query="What happened during Apollo 13?",
        n_results=3
    )

    print(results)