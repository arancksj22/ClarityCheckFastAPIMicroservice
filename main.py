from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="ClarityCheckAIMicroservice")

# Load embedding model to use
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

@app.post("/find-problematic-chunks")
async def find_problematic_chunks() -> BiasChunksResponse:
    # Find chunks that might contain bias, ethics issues, or logical fallacies
    
    if faiss_index is None:
        return BiasChunksResponse(
            ethics_chunks=[],
            bias_chunks=[],
            fallacy_chunks=[],
            combined_top_chunks=[],
            total_chunks_found=0
        )
    
    # Define queries for each category
    ethics_queries = [
        "ethical concerns and moral issues",
        "ethics violations and misconduct",
        "informed consent and participant rights"
    ]
    
    bias_queries = [
        "gender bias and discrimination",
        "racial bias and stereotyping",
        "age bias and demographic assumptions",
        "cultural bias and prejudice"
    ]
    
    fallacy_queries = [
        "logical fallacies and flawed reasoning",
        "unsupported claims and assumptions",
        "correlation versus causation errors",
        "overgeneralization and hasty conclusions"
    ]
    
    # Search for each category
    ethics_chunks = set()
    for query in ethics_queries:
        chunks, _ = search_for_query(query, top_k=3)
        ethics_chunks.update(chunks)
    
    bias_chunks = set()
    for query in bias_queries:
        chunks, _ = search_for_query(query, top_k=3)
        bias_chunks.update(chunks)
    
    fallacy_chunks = set()
    for query in fallacy_queries:
        chunks, _ = search_for_query(query, top_k=3)
        fallacy_chunks.update(chunks)
    
    # Combine all unique chunks
    all_problematic_chunks = ethics_chunks.union(bias_chunks).union(fallacy_chunks)
    
    # If we have too many chunks, prioritize by getting the most relevant ones
    if len(all_problematic_chunks) > 5:
        # Re-search with a general "problematic content" query to get top chunks
        combined_query = "bias, ethics violations, logical fallacies, discrimination, unfair assumptions"
        top_chunks, _ = search_for_query(combined_query, top_k=5)
    else:
        top_chunks = list(all_problematic_chunks)
    
    return BiasChunksResponse(
        ethics_chunks=list(ethics_chunks)[:3],
        bias_chunks=list(bias_chunks)[:3], 
        fallacy_chunks=list(fallacy_chunks)[:3],
        combined_top_chunks=top_chunks[:5],  # Top 5 most problematic chunks
        total_chunks_found=len(all_problematic_chunks)
    )


@app.get("/")
async def root():
    return {"message": "ClarityCheckFastAPIMicroservice"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
