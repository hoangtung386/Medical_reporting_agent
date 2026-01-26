"""
RAG (Retrieval-Augmented Generation) utilities.

Converts Agent 7 (RAG Retrieval Specialist) into optional utility module.
Used for ablation studies: "Does RAG improve report quality?"

Retrieves relevant clinical guidelines and similar cases.
"""

import warnings
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    warnings.warn("ChromaDB not installed. RAG features unavailable. Install: pip install chromadb")


class GuidelineRetriever:
    """
    Retrieve relevant clinical guidelines from vector database.
    
    Useful for citing standardized protocols (e.g., Fleischner Society, Lung-RADS).
    """
    
    def __init__(
        self,
        db_path: str = "./data/rag_db",
        collection_name: str = "clinical_guidelines"
    ):
        """
        Args:
            db_path: Path to ChromaDB storage
            collection_name: Name of the collection
        """
        self.db_path = db_path
        self.collection_name = collection_name
        
        if not CHROMADB_AVAILABLE:
            print("WARNING: ChromaDB not available. Running in mock mode.")
            self.client = None
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=db_path)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(collection_name)
                print(f"Loaded collection '{collection_name}' with {self.collection.count()} documents")
            except:
                self.collection = self.client.create_collection(collection_name)
                print(f"Created new collection '{collection_name}'")
    
    def add_guidelines(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """
        Add clinical guidelines to the database.
        
        Args:
            documents: List of guideline texts
            metadatas: Metadata for each document (e.g., source, year, category)
            ids: Unique IDs for each document
        """
        if self.collection is None:
            print("ChromaDB not available. Cannot add documents.")
            return
        
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"Added {len(documents)} guidelines to database")
    
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query for relevant guidelines.
        
        Args:
            query_text: Query string (e.g., "lung nodule management")
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of retrieved documents with metadata
        """
        if self.collection is None:
            # Mock results
            return [
                {
                    'document': "Mock guideline: Nodules <6mm require no follow-up.",
                    'metadata': {'source': 'mock', 'relevance': 0.9},
                    'distance': 0.1
                }
            ]
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else 0.0
            })
        
        return formatted
    
    def query_by_pathology(
        self,
        pathology_type: str,
        anatomy: Optional[str] = None,
        n_results: int = 3
    ) -> str:
        """
        Query guidelines for specific pathology.
        
        Args:
            pathology_type: Type of finding (e.g., "nodule", "mass", "lesion")
            anatomy: Anatomical location (e.g., "lung", "liver")
            n_results: Number of guidelines to retrieve
        
        Returns:
            Concatenated guideline text
        """
        # Build query
        query_parts = [pathology_type]
        if anatomy:
            query_parts.append(anatomy)
        query_text = " ".join(query_parts)
        
        # Retrieve
        results = self.query(query_text, n_results=n_results)
        
        # Concatenate documents
        guidelines = []
        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            text = result['document']
            guidelines.append(f"[Guideline {i} - {source}]: {text}")
        
        return "\n\n".join(guidelines)


class CaseRetriever:
    """
    Retrieve similar historical cases for reference.
    
    "This finding is similar to case #12345 which was diagnosed as..."
    """
    
    def __init__(
        self,
        db_path: str = "./data/rag_db",
        collection_name: str = "historical_cases"
    ):
        self.db_path = db_path
        self.collection_name = collection_name
        
        if not CHROMADB_AVAILABLE:
            self.client = None
            self.collection = None
        else:
            self.client = chromadb.PersistentClient(path=db_path)
            try:
                self.collection = self.client.get_collection(collection_name)
            except:
                self.collection = self.client.create_collection(collection_name)
    
    def add_cases(
        self,
        case_descriptions: List[str],
        diagnoses: List[str],
        case_ids: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add historical cases to database.
        
        Args:
            case_descriptions: Textual descriptions of findings
            diagnoses: Final diagnoses
            case_ids: Unique case identifiers
            metadatas: Additional metadata (age, sex, outcome, etc.)
        """
        if self.collection is None:
            return
        
        # Combine description and diagnosis for better retrieval
        documents = [
            f"{desc} [DIAGNOSIS: {diag}]"
            for desc, diag in zip(case_descriptions, diagnoses)
        ]
        
        if metadatas is None:
            metadatas = [{'diagnosis': diag} for diag in diagnoses]
        else:
            for i, diag in enumerate(diagnoses):
                metadatas[i]['diagnosis'] = diag
        
        self.collection.add(
            documents=documents,
            ids=case_ids,
            metadatas=metadatas
        )
    
    def find_similar_cases(
        self,
        current_findings: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical cases.
        
        Args:
            current_findings: Description of current case findings
            n_results: Number of similar cases to return
        
        Returns:
            List of similar cases with diagnoses
        """
        if self.collection is None:
            return []
        
        results = self.collection.query(
            query_texts=[current_findings],
            n_results=n_results
        )
        
        similar_cases = []
        for i in range(len(results['ids'][0])):
            similar_cases.append({
                'case_id': results['ids'][0][i],
                'description': results['documents'][0][i],
                'diagnosis': results['metadatas'][0][i].get('diagnosis', 'Unknown'),
                'similarity': 1.0 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return similar_cases


def load_guidelines_from_text(
    file_path: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Load and chunk clinical guidelines from text file.
    
    Args:
        file_path: Path to guideline text file
        chunk_size: Number of characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of document chunks with metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    chunks = []
    start = 0
    chunk_id = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        chunks.append({
            'text': chunk,
            'metadata': {
                'source': file_path,
                'chunk_id': chunk_id,
                'start_char': start,
                'end_char': end
            }
        })
        
        start += chunk_size - overlap
        chunk_id += 1
    
    return chunks


# Example usage
if __name__ == "__main__":
    # Example: Setup guideline database
    guideline_db = GuidelineRetriever(db_path="./test_rag_db")
    
    # Add some mock guidelines
    sample_guidelines = [
        "Fleischner Society: Lung nodules <6mm in diameter require no routine follow-up in low-risk patients.",
        "Lung-RADS: Category 3 nodules (6-8mm) should be followed up at 6 months.",
        "LI-RADS: Arterial phase hyperenhancement with washout indicates HCC in cirrhotic liver."
    ]
    
    sample_metadata = [
        {'source': 'Fleischner 2017', 'category': 'lung', 'year': 2017},
        {'source': 'Lung-RADS v1.1', 'category': 'lung', 'year': 2019},
        {'source': 'LI-RADS v2018', 'category': 'liver', 'year': 2018}
    ]
    
    guideline_db.add_guidelines(
        documents=sample_guidelines,
        metadatas=sample_metadata,
        ids=['fleischner_1', 'lungrads_1', 'lirads_1']
    )
    
    # Query
    results = guideline_db.query("small lung nodule follow-up", n_results=2)
    print("Retrieved guidelines:")
    for result in results:
        print(f"- {result['document']}")
        print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
