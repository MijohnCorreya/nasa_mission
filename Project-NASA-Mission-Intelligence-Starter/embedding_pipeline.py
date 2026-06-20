#!/usr/bin/env python3
"""
ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only

This script reads parsed text data from various NASA space mission folders and creates
a permanent ChromaDB collection with OpenAI embeddings for RAG applications.
Optimized to process only text files to avoid duplication with JSON versions.

Supported data sources:
- Apollo 11 extracted data (text files only)
- Apollo 13 extracted data (text files only)
- Apollo 11 Textract extracted data (text files only)
- Challenger transcribed audio data (text files only)
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import sys

try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ModuleNotFoundError:
    pass


import chromadb
from chromadb.config import Settings
import openai
from openai import OpenAI
import hashlib
import time
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chroma_embedding_text_only.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - Text files only"""

    def __init__(self,
                 openai_api_key: str,
                 chroma_persist_directory: str = "./chroma_db",
                 collection_name: str = "nasa_space_missions_text",
                 embedding_model: str = "text-embedding-3-small",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the embedding pipeline.

        Args:
            openai_api_key:           OpenAI API key
            chroma_persist_directory: Directory to persist ChromaDB
            collection_name:          Name of the ChromaDB collection
            embedding_model:          OpenAI embedding model to use
            chunk_size:               Maximum character size of each text chunk
            chunk_overlap:            Character overlap between consecutive chunks
        """
        # Initialize OpenAI client pointed at the Vocareum proxy
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url="https://openai.vocareum.com/v1"
        )

        # Store configuration parameters
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name

        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or retrieve the collection WITHOUT an embedding function because
        # embeddings are computed manually via self.get_embeddings() and passed
        # directly to collection.add() / collection.update(). This avoids the
        # OpenAIEmbeddingFunction trying to reach the standard OpenAI endpoint
        # instead of the Vocareum proxy.
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Initialized collection '{collection_name}' "
            f"with {self.collection.count()} existing documents."
        )

    # ------------------------------------------------------------------
    # Text chunking
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Split text into overlapping chunks, preferring sentence boundaries.

        Args:
            text:     Full text to chunk
            metadata: Base metadata applied to every chunk

        Returns:
            List of (chunk_text, chunk_metadata) tuples
        """
        # Short texts are returned as a single chunk
        if len(text) <= self.chunk_size:
            chunk_metadata = {**metadata, 'chunk_index': 0, 'total_chunks': 1}
            return [(text, chunk_metadata)]

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at a sentence boundary within the last 20% of the window
            if end < len(text):
                search_start = start + int(self.chunk_size * 0.8)
                best_break = -1

                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n', '\n\n']:
                    pos = text.rfind(punct, search_start, end)
                    if pos != -1 and pos > best_break:
                        best_break = pos + len(punct)

                # Fall back to the nearest newline
                if best_break == -1:
                    newline_pos = text.rfind('\n', search_start, end)
                    if newline_pos != -1:
                        best_break = newline_pos + 1

                if best_break != -1:
                    end = best_break

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_metadata = {
                    **metadata,
                    'chunk_index': chunk_index,
                    'chunk_start': start,
                    'chunk_end': end,
                }
                chunks.append((chunk_text, chunk_metadata))
                chunk_index += 1

            # Advance with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break

        # Patch total_chunks now that we know the final count
        total = len(chunks)
        return [
            (chunk_text, {**chunk_meta, 'total_chunks': total})
            for chunk_text, chunk_meta in chunks
        ]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Fetch OpenAI embeddings for a batch of texts via the Vocareum proxy.

        Args:
            texts: List of strings to embed

        Returns:
            List of embedding vectors (one per input string)
        """
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [item.embedding for item in response.data]
        except openai.RateLimitError:
            logger.warning("Rate limit hit — waiting 60 s before retrying...")
            time.sleep(60)
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error getting embeddings: {e}")
            raise

    # ------------------------------------------------------------------
    # Document ID generation
    # ------------------------------------------------------------------

    def generate_document_id(self, file_path: Path, metadata: Dict[str, Any]) -> str:
        """
        Generate a stable, human-readable document ID.

        Format: ``mission_source_chunk_XXXX``

        Args:
            file_path: Source file path (used as fallback for source name)
            metadata:  Chunk metadata containing mission, source, chunk_index

        Returns:
            Sanitised document ID string
        """
        mission = metadata.get('mission', 'unknown')
        source = metadata.get('source', file_path.stem)
        chunk_index = metadata.get('chunk_index', 0)

        mission_clean = mission.replace(' ', '_').replace('/', '_')
        source_clean = source.replace(' ', '_').replace('/', '_')

        return f"{mission_clean}_{source_clean}_chunk_{chunk_index:04d}"

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------

    def check_document_exists(self, doc_id: str) -> bool:
        """Return True if a document with the given ID is already in the collection."""
        result = self.collection.get(ids=[doc_id])
        return len(result['ids']) > 0

    def update_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """
        Replace an existing document's content and embedding in-place.

        Args:
            doc_id:   ID of the document to update
            text:     New text content
            metadata: New metadata

        Returns:
            True on success, False on failure
        """
        try:
            embedding = self.get_embeddings([text])[0]
            self.collection.update(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata],
                embeddings=[embedding]
            )
            logger.debug(f"Updated document: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        """
        Delete all documents whose ``source`` metadata field contains *source_pattern*.

        Args:
            source_pattern: Substring to match against the source field

        Returns:
            Number of documents deleted
        """
        try:
            all_docs = self.collection.get()
            ids_to_delete = [
                all_docs['ids'][i]
                for i, meta in enumerate(all_docs['metadatas'])
                if source_pattern in meta.get('source', '')
            ]

            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} documents "
                    f"matching source pattern: {source_pattern}"
                )
                return len(ids_to_delete)

            logger.info(f"No documents found matching source pattern: {source_pattern}")
            return 0

        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_file_documents(self, file_path: Path) -> List[str]:
        """
        Return all document IDs that originated from *file_path*.

        Args:
            file_path: Path to the source file

        Returns:
            List of matching document IDs
        """
        try:
            source = file_path.stem
            mission = self.extract_mission_from_path(file_path)
            all_docs = self.collection.get()

            return [
                all_docs['ids'][i]
                for i, meta in enumerate(all_docs['metadatas'])
                if meta.get('source') == source and meta.get('mission') == mission
            ]
        except Exception as e:
            logger.error(f"Error getting file documents: {e}")
            return []

    # ------------------------------------------------------------------
    # File processing
    # ------------------------------------------------------------------

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Read a plain-text file and return its chunks with enriched metadata.

        Args:
            file_path: Path to the .txt file

        Returns:
            List of (chunk_text, metadata) tuples, or [] on error / empty file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = {
                'source': file_path.stem,
                'file_path': str(file_path),
                'file_type': 'text',
                'content_type': 'full_text',
                'mission': self.extract_mission_from_path(file_path),
                'data_type': self.extract_data_type_from_path(file_path),
                'document_category': self.extract_document_category_from_filename(file_path.name),
                'file_size': len(content),
                'processed_timestamp': datetime.now().isoformat()
            }

            return self.chunk_text(content, metadata)

        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------

    def extract_mission_from_path(self, file_path: Path) -> str:
        """Infer mission name from the file path."""
        path_str = str(file_path).lower()
        if 'apollo11' in path_str or 'apollo_11' in path_str:
            return 'apollo_11'
        elif 'apollo13' in path_str or 'apollo_13' in path_str:
            return 'apollo_13'
        elif 'challenger' in path_str:
            return 'challenger'
        return 'unknown'

    def extract_data_type_from_path(self, file_path: Path) -> str:
        """Infer data type from the file path."""
        path_str = str(file_path).lower()
        if 'transcript' in path_str:
            return 'transcript'
        elif 'textract' in path_str:
            return 'textract_extracted'
        elif 'audio' in path_str:
            return 'audio_transcript'
        elif 'flight_plan' in path_str:
            return 'flight_plan'
        return 'document'

    def extract_document_category_from_filename(self, filename: str) -> str:
        """Infer document category from the filename."""
        fn = filename.lower()
        if 'pao' in fn:
            return 'public_affairs_officer'
        elif 'cm' in fn:
            return 'command_module'
        elif 'tec' in fn:
            return 'technical'
        elif 'flight_plan' in fn:
            return 'flight_plan'
        elif 'mission_audio' in fn:
            return 'mission_audio'
        elif 'ntrs' in fn:
            return 'nasa_archive'
        elif '19900066485' in fn:
            return 'technical_report'
        elif '19710015566' in fn:
            return 'mission_report'
        elif 'full_text' in fn:
            return 'complete_document'
        return 'general_document'

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        """
        Recursively scan mission sub-directories for .txt files.

        Skips hidden files and any file with "summary" in its name.

        Args:
            base_path: Root directory that contains apollo11/, apollo13/, challenger/

        Returns:
            Filtered list of Path objects ready for processing
        """
        base_path = Path(base_path)
        files_to_process = []

        for data_dir in ['apollo11', 'apollo13', 'challenger']:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                text_files = list(dir_path.glob('**/*.txt'))
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        filtered_files = [
            fp for fp in files_to_process
            if not fp.name.startswith('.')
            and 'summary' not in fp.name.lower()
            and fp.suffix.lower() == '.txt'
        ]

        logger.info(f"Total text files to process: {len(filtered_files)}")

        mission_counts: Dict[str, int] = {}
        for fp in filtered_files:
            m = self.extract_mission_from_path(fp)
            mission_counts[m] = mission_counts.get(m, 0) + 1

        logger.info("Files by mission:")
        for mission, count in mission_counts.items():
            logger.info(f"  {mission}: {count} files")

        return filtered_files

    # ------------------------------------------------------------------
    # Batch ingestion
    # ------------------------------------------------------------------

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = 'skip'
    ) -> Dict[str, int]:
        """
        Add (or update/replace) documents in the ChromaDB collection.

        Args:
            documents:   List of (chunk_text, metadata) tuples from process_text_file()
            file_path:   Source file path (used to look up existing docs for replace mode)
            batch_size:  Number of documents to embed and upsert per API call
            update_mode: One of:
                         - ``'skip'``    — leave existing documents unchanged (default)
                         - ``'update'``  — re-embed and overwrite existing documents
                         - ``'replace'`` — delete all existing docs from this file, then add all

        Returns:
            Dict with keys ``added``, ``updated``, ``skipped``
        """
        if not documents:
            return {'added': 0, 'updated': 0, 'skipped': 0}

        stats = {'added': 0, 'updated': 0, 'skipped': 0}

        # Delete existing file documents before re-adding in replace mode
        if update_mode == 'replace':
            existing_ids = self.get_file_documents(file_path)
            if existing_ids:
                self.collection.delete(ids=existing_ids)
                logger.info(
                    f"Replaced {len(existing_ids)} existing documents from {file_path.name}"
                )

        pending_ids: List[str] = []
        pending_texts: List[str] = []
        pending_metadatas: List[Dict] = []

        def flush_batch() -> None:
            """Embed and insert the current pending batch."""
            if not pending_ids:
                return
            try:
                batch_embeddings = self.get_embeddings(pending_texts)
                self.collection.add(
                    ids=pending_ids,
                    documents=pending_texts,
                    embeddings=batch_embeddings,
                    metadatas=pending_metadatas
                )
                stats['added'] += len(pending_ids)
                logger.debug(f"Flushed batch of {len(pending_ids)} documents")
            except Exception as e:
                logger.error(f"Failed to add batch from {file_path.name}: {e}")
            pending_ids.clear()
            pending_texts.clear()
            pending_metadatas.clear()

        for text, metadata in documents:
            doc_id = self.generate_document_id(file_path, metadata)

            # Handle existing documents according to update_mode
            if update_mode != 'replace' and self.check_document_exists(doc_id):
                if update_mode == 'update':
                    success = self.update_document(doc_id, text, metadata)
                    if success:
                        stats['updated'] += 1
                    else:
                        logger.warning(f"Failed to update document: {doc_id}")
                else:  # skip
                    stats['skipped'] += 1
                continue

            pending_ids.append(doc_id)
            pending_texts.append(text)
            pending_metadatas.append(metadata)

            if len(pending_ids) >= batch_size:
                flush_batch()

        # Flush any remaining documents
        flush_batch()

        return stats

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def process_all_text_data(
        self,
        base_path: str,
        update_mode: str = 'skip',
        batch_size: int = 50
    ) -> Dict[str, Any]:
        """
        Scan all mission directories, chunk every text file, and ingest into ChromaDB.

        Args:
            base_path:   Root directory containing apollo11/, apollo13/, challenger/
            update_mode: Passed through to add_documents_to_collection()
            batch_size:  Passed through to add_documents_to_collection()

        Returns:
            Aggregated statistics dict with global counters and per-mission breakdowns
        """
        stats: Dict[str, Any] = {
            'files_processed': 0,
            'documents_added': 0,
            'documents_updated': 0,
            'documents_skipped': 0,
            'errors': 0,
            'total_chunks': 0,
            'missions': {}
        }

        files_to_process = self.scan_text_files_only(base_path)

        if not files_to_process:
            logger.warning("No text files found to process.")
            return stats

        for file_path in files_to_process:
            logger.info(f"Processing: {file_path}")
            mission = self.extract_mission_from_path(file_path)

            # Ensure per-mission bucket exists
            if mission not in stats['missions']:
                stats['missions'][mission] = {
                    'files': 0, 'chunks': 0,
                    'added': 0, 'updated': 0, 'skipped': 0
                }

            try:
                # Chunk the file
                documents = self.process_text_file(file_path)

                if not documents:
                    logger.warning(f"No content extracted from {file_path}")
                    stats['errors'] += 1
                    continue

                # Ingest chunks into ChromaDB
                file_stats = self.add_documents_to_collection(
                    documents,
                    file_path,
                    batch_size=batch_size,
                    update_mode=update_mode
                )

                chunk_count = len(documents)

                # Update global stats
                stats['files_processed'] += 1
                stats['total_chunks'] += chunk_count
                stats['documents_added'] += file_stats['added']
                stats['documents_updated'] += file_stats['updated']
                stats['documents_skipped'] += file_stats['skipped']

                # Update per-mission stats
                stats['missions'][mission]['files'] += 1
                stats['missions'][mission]['chunks'] += chunk_count
                stats['missions'][mission]['added'] += file_stats['added']
                stats['missions'][mission]['updated'] += file_stats['updated']
                stats['missions'][mission]['skipped'] += file_stats['skipped']

                logger.info(
                    f"  -> {chunk_count} chunks | "
                    f"added={file_stats['added']} "
                    f"updated={file_stats['updated']} "
                    f"skipped={file_stats['skipped']}"
                )

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                stats['errors'] += 1

        return stats

    # ------------------------------------------------------------------
    # Querying and statistics
    # ------------------------------------------------------------------

    def get_collection_info(self) -> Dict[str, Any]:
        """Return basic information about the current collection."""
        return {
            'collection_name': self.collection_name,
            'document_count': self.collection.count(),
            'embedding_model': self.embedding_model,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
        }

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Run a semantic similarity query against the collection.

        Args:
            query_text: Natural-language query string
            n_results:  Maximum number of results to return

        Returns:
            Raw ChromaDB query result dict, or {} on error
        """
        try:
            # Embed the query manually to stay consistent with the Vocareum proxy
            query_embedding = self.get_embeddings([query_text])[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(n_results, self.collection.count())
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {}

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return detailed breakdown of documents by mission, data type, category, and file type."""
        try:
            all_docs = self.collection.get()

            if not all_docs['metadatas']:
                return {'error': 'No documents in collection'}

            stats: Dict[str, Any] = {
                'total_documents': len(all_docs['metadatas']),
                'missions': {},
                'data_types': {},
                'document_categories': {},
                'file_types': {}
            }

            for meta in all_docs['metadatas']:
                for key, field in [
                    ('missions', 'mission'),
                    ('data_types', 'data_type'),
                    ('document_categories', 'document_category'),
                    ('file_types', 'file_type'),
                ]:
                    val = meta.get(field, 'unknown')
                    stats[key][val] = stats[key].get(val, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    """Command-line interface for the embedding pipeline."""
    parser = argparse.ArgumentParser(description='ChromaDB Embedding Pipeline for NASA Data')
    parser.add_argument('--data-path', default='.', help='Path to data directories')
    parser.add_argument('--openai-key', required=True, help='OpenAI API key')
    parser.add_argument('--chroma-dir', default='./chroma_db_openai', help='ChromaDB persist directory')
    parser.add_argument('--collection-name', default='nasa_space_missions_text', help='Collection name')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='OpenAI embedding model')
    parser.add_argument('--chunk-size', type=int, default=500, help='Text chunk size (characters)')
    parser.add_argument('--chunk-overlap', type=int, default=100, help='Chunk overlap size (characters)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for embedding API calls')
    parser.add_argument('--update-mode', choices=['skip', 'update', 'replace'], default='skip',
                        help='How to handle existing documents: skip, update, or replace')
    parser.add_argument('--test-query', help='Run a test query after processing')
    parser.add_argument('--stats-only', action='store_true', help='Only show collection statistics, then exit')
    parser.add_argument('--delete-source', help='Delete all documents matching a source pattern, then exit')

    args = parser.parse_args()

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        openai_api_key=args.openai_key,
        chroma_persist_directory=args.chroma_dir,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    # --- Delete-source mode ---
    if args.delete_source:
        deleted = pipeline.delete_documents_by_source(args.delete_source)
        logger.info(f"Deleted {deleted} documents matching source pattern: {args.delete_source}")
        return

    # --- Stats-only mode ---
    if args.stats_only:
        logger.info("Collection Statistics:")
        for key, value in pipeline.get_collection_stats().items():
            logger.info(f"  {key}: {value}")
        return

    # --- Normal processing mode ---
    logger.info(f"Starting text data processing with update mode: {args.update_mode}")
    start_time = time.time()

    stats = pipeline.process_all_text_data(
        args.data_path,
        update_mode=args.update_mode,
        batch_size=args.batch_size
    )

    processing_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed:               {stats['files_processed']}")
    logger.info(f"Total chunks created:          {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated:             {stats['documents_updated']}")
    logger.info(f"Documents skipped:             {stats['documents_skipped']}")
    logger.info(f"Errors:                        {stats['errors']}")
    logger.info(f"Processing time:               {processing_time:.2f} seconds")

    logger.info("\nMission breakdown:")
    for mission, ms in stats['missions'].items():
        logger.info(f"  {mission}: {ms['files']} files, {ms['chunks']} chunks")
        logger.info(f"    Added: {ms['added']}, Updated: {ms['updated']}, Skipped: {ms['skipped']}")

    info = pipeline.get_collection_info()
    logger.info(f"\nCollection:               {info.get('collection_name', 'N/A')}")
    logger.info(f"Total docs in collection: {info.get('document_count', 'N/A')}")

    if args.test_query:
        logger.info(f"\nTesting query: '{args.test_query}'")
        results = pipeline.query_collection(args.test_query)
        if results and 'documents' in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results['documents'][0][:3]):
                logger.info(f"  Result {i + 1}: {doc[:200]}...")

    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()