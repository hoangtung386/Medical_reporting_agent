"""RAG (Retrieval-Augmented Generation) utilities.

Optional module for retrieving clinical guidelines and similar
historical cases. Used for ablation studies.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import chromadb

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning(
        "chromadb not installed. RAG features unavailable. "
        "Install with: pip install chromadb"
    )


class GuidelineRetriever:
    """Retrieve relevant clinical guidelines from a vector database.

    Useful for citing standardised protocols (e.g. Fleischner Society,
    Lung-RADS).

    Args:
        db_path: Path to ChromaDB persistent storage.
        collection_name: Name of the guideline collection.
    """

    def __init__(
        self,
        db_path: str = "./data/rag_db",
        collection_name: str = "clinical_guidelines",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name

        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB unavailable -- running in mock mode.")
            self.client = None
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=db_path)
            try:
                self.collection = self.client.get_collection(
                    collection_name
                )
                logger.info(
                    "Loaded collection '%s' (%d documents)",
                    collection_name,
                    self.collection.count(),
                )
            except Exception:
                self.collection = self.client.create_collection(
                    collection_name
                )
                logger.info("Created new collection '%s'", collection_name)

    def add_guidelines(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add clinical guideline documents to the database.

        Args:
            documents: Guideline texts.
            metadatas: Per-document metadata.
            ids: Unique document identifiers.
        """
        if self.collection is None:
            logger.warning("Cannot add documents (ChromaDB unavailable).")
            return

        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("Added %d guidelines to database", len(documents))

    def query(
        self,
        query_text: str,
        n_results: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Query for relevant guidelines.

        Args:
            query_text: Natural-language query.
            n_results: Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            List of dicts with ``document``, ``metadata``, and
            ``distance`` keys.
        """
        if self.collection is None:
            return [
                {
                    "document": (
                        "Mock guideline: Nodules <6mm require "
                        "no follow-up."
                    ),
                    "metadata": {"source": "mock", "relevance": 0.9},
                    "distance": 0.1,
                }
            ]

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata,
        )

        formatted: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            formatted.append(
                {
                    "document": results["documents"][0][i],
                    "metadata": (
                        results["metadatas"][0][i]
                        if results["metadatas"]
                        else {}
                    ),
                    "distance": (
                        results["distances"][0][i]
                        if results["distances"]
                        else 0.0
                    ),
                }
            )
        return formatted

    def query_by_pathology(
        self,
        pathology_type: str,
        anatomy: Optional[str] = None,
        n_results: int = 3,
    ) -> str:
        """Query guidelines for a specific pathology.

        Args:
            pathology_type: Finding type (e.g. ``"nodule"``).
            anatomy: Anatomical location (e.g. ``"lung"``).
            n_results: Number of guidelines to retrieve.

        Returns:
            Concatenated guideline text.
        """
        query_parts = [pathology_type]
        if anatomy:
            query_parts.append(anatomy)

        results = self.query(" ".join(query_parts), n_results=n_results)

        guidelines: list[str] = []
        for i, result in enumerate(results, 1):
            source = result["metadata"].get("source", "Unknown")
            text = result["document"]
            guidelines.append(f"[Guideline {i} - {source}]: {text}")

        return "\n\n".join(guidelines)


class CaseRetriever:
    """Retrieve similar historical cases for reference.

    Args:
        db_path: Path to ChromaDB persistent storage.
        collection_name: Name of the cases collection.
    """

    def __init__(
        self,
        db_path: str = "./data/rag_db",
        collection_name: str = "historical_cases",
    ) -> None:
        self.db_path = db_path
        self.collection_name = collection_name

        if not CHROMADB_AVAILABLE:
            self.client = None
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=db_path)
            try:
                self.collection = self.client.get_collection(
                    collection_name
                )
            except Exception:
                self.collection = self.client.create_collection(
                    collection_name
                )

    def add_cases(
        self,
        case_descriptions: List[str],
        diagnoses: List[str],
        case_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add historical cases to the database.

        Args:
            case_descriptions: Textual findings descriptions.
            diagnoses: Final diagnoses.
            case_ids: Unique case identifiers.
            metadatas: Additional metadata per case.
        """
        if self.collection is None:
            return

        documents = [
            f"{desc} [DIAGNOSIS: {diag}]"
            for desc, diag in zip(case_descriptions, diagnoses)
        ]

        if metadatas is None:
            metadatas = [{"diagnosis": diag} for diag in diagnoses]
        else:
            for i, diag in enumerate(diagnoses):
                metadatas[i]["diagnosis"] = diag

        self.collection.add(
            documents=documents,
            ids=case_ids,
            metadatas=metadatas,
        )

    def find_similar_cases(
        self,
        current_findings: str,
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find similar historical cases.

        Args:
            current_findings: Description of current-case findings.
            n_results: Number of similar cases to return.

        Returns:
            List of case dicts with ``case_id``, ``description``,
            ``diagnosis``, and ``similarity``.
        """
        if self.collection is None:
            return []

        results = self.collection.query(
            query_texts=[current_findings],
            n_results=n_results,
        )

        similar: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            similar.append(
                {
                    "case_id": results["ids"][0][i],
                    "description": results["documents"][0][i],
                    "diagnosis": results["metadatas"][0][i].get(
                        "diagnosis", "Unknown"
                    ),
                    "similarity": 1.0 - results["distances"][0][i],
                }
            )
        return similar


def load_guidelines_from_text(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Load and chunk clinical guidelines from a text file.

    Args:
        file_path: Path to the guideline text file.
        chunk_size: Characters per chunk.
        overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts with ``text`` and ``metadata``.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        text = fh.read()

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(
            {
                "text": text[start:end],
                "metadata": {
                    "source": file_path,
                    "chunk_id": chunk_id,
                    "start_char": start,
                    "end_char": end,
                },
            }
        )
        start += chunk_size - overlap
        chunk_id += 1

    return chunks
