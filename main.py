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

def search_for_query(query: str, top_k: int = 5) -> tuple:
    # Search for chunks matching a specific query
    if faiss_index is None:
        return [], []
    
    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = faiss_index.search(query_embedding, min(top_k, len(chunks_store)))
    
    # Get chunks with scores above threshold (0.3 is reasonable for similarity)
    relevant_chunks = []
    for i, score in enumerate(scores[0]):
        if score > 0.3:  # Only return reasonably similar chunks
            relevant_chunks.append(chunks_store[indices[0][i]])
    
    return relevant_chunks, scores[0].tolist()

@app.post("/load-text")
async def load_text(input_data: TextInput):
    # Load and chunk text, create FAISS index
    global chunks_store, faiss_index
    
    # Chunk the text
    chunks_store = chunk_text(input_data.text)
    
    if not chunks_store:
        return {"error": "No valid chunks created from text"}
    
    # Create embeddings
    embeddings = model.encode(chunks_store)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    return {"message": f"Loaded {len(chunks_store)} chunks into FAISS index"}

@app.get("/")
async def root():
    return {"message": "ClarityCheckFastAPIMicroservice"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
