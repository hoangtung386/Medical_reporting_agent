"""
Agent 7: RAG Retrieval Specialist.
Retrieves relevant clinical guidelines and similar cases from VectorDB.
"""
from .base import BaseAgent
from typing import Any, Dict
import os

try:
    import chromadb
except ImportError:
    chromadb = None

class RAGSpecialistAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Agent 7: RAG Retrieval")
        self.client = None
        self.collection = None
        
        if chromadb:
            print(f"[{self.name}] Connecting to Knowledge Base (ChromaDB)...")
            # In-memory for demo, persistent in production
            self.client = chromadb.Client() 
            self.collection = self.client.get_or_create_collection(name="clinical_guidelines")
            
            # Optional: Populate with dummy data if empty
            if self.collection.count() == 0:
                 self.collection.add(
                     documents=["Fleischner Society guidelines for solid nodules > 8mm: Recommend PET/CT.", 
                                "Lung-RADS v1.1 Category 4B: Suspicious, > 15% probability of malignancy."],
                     metadatas=[{"source": "Fleischner"}, {"source": "Lung-RADS"}],
                     ids=["guideline_1", "guideline_2"]
                 )
        else:
             print(f"[{self.name}] WARNING: ChromaDB not installed. RAG disabled.")

    def forward(self, vision_features: Any, seg_output: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[{self.name}] Retrieving similar cases and guidelines...")
        
        gathered_info = {
            "guideline": "",
            "similar_cases": []
        }
        
        if self.collection:
            # Query based on pathology findings (mocked text construction from seg/vision)
            # In real system: query_embedding = embedder(text_description)
            query_text = "solid nodule > 8mm"
            
            results = self.collection.query(
                query_texts=[query_text],
                n_results=1
            )
            
            if results['documents']:
                 gathered_info["guideline"] = results['documents'][0][0]
        else:
            gathered_info["guideline"] = "Fleischner Society guidelines for solid nodules > 8mm: Recommend PET/CT. (Mocked)"
            
        return gathered_info
