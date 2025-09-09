from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="ClarityCheck AI Service")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Global variables to store chunks and index
chunks_store = []
faiss_index = None

class TextInput(BaseModel):
    text: str

class BiasChunksResponse(BaseModel):
    ethics_chunks: List[str]
    bias_chunks: List[str]
    fallacy_chunks: List[str]
    combined_top_chunks: List[str]
    total_chunks_found: int

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    # Split text into chunks with some overlap for context
    chunks = []
    overlap = 50  # 50 character overlap between chunks
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk.strip())
            
    return chunks