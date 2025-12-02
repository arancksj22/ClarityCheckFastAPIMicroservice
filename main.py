from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Set

app = FastAPI(title="ClarityCheckAIMicroservice")

# --- 1. Load Model Globally (Once at startup) ---
# We load this here so we don't reload it for every request (which would be slow)
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded!")

# --- 2. Data Models ---
class TextInput(BaseModel):
    text: str

class BiasChunksResponse(BaseModel):
    ethics_chunks: List[str]
    bias_chunks: List[str]
    fallacy_chunks: List[str]
    combined_top_chunks: List[str]
    total_chunks_found: int

# --- 3. Helper Functions ---

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """Splits text into overlapping chunks."""
    chunks = []
    overlap = 50 
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 50:  # Only keep substantial chunks
            chunks.append(chunk.strip())
    return chunks

def search_local_index(query: str, index, chunks: List[str], top_k: int = 3) -> List[str]:
    """
    Searches a SPECIFIC index (passed as argument) for the query.
    This ensures we only search the current user's document.
    """
    if index is None or not chunks:
        return []
    
    # Encode query
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search
    # We use min() to ensure we don't ask for more results than we have chunks
    k_to_search = min(top_k, len(chunks))
    if k_to_search == 0:
        return []

    scores, indices = index.search(query_embedding, k_to_search)
    
    # Filter results by similarity score threshold
    relevant_chunks = []
    for i, score in enumerate(scores[0]):
        if score > 0.25:  # Threshold: Only return if somewhat relevant
            idx = indices[0][i]
            if 0 <= idx < len(chunks):
                relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

# --- 4. The Main Endpoint ---

@app.post("/analyze", response_model=BiasChunksResponse)
async def analyze_document(input_data: TextInput):
    """
    Receives raw text, chunks it, builds a temporary index, 
    analyzes it for bias/ethics, and returns the report.
    """
    
    # A. Chunk the text
    current_chunks = chunk_text(input_data.text)
    
    # Handle empty or too short text
    if not current_chunks:
        return BiasChunksResponse(
            ethics_chunks=[], bias_chunks=[], fallacy_chunks=[], 
            combined_top_chunks=[], total_chunks_found=0
        )

    # B. Create Embeddings & Index (In-Memory for this request only)
    embeddings = model.encode(current_chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    local_index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine Similarity)
    faiss.normalize_L2(embeddings)
    local_index.add(embeddings)

    # C. Define Queries
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
        "overgeneralization"
    ]

    # D. Run Searches
    # We use Sets to avoid duplicate chunks if multiple queries find the same sentence
    ethics_set = set()
    for q in ethics_queries:
        found = search_local_index(q, local_index, current_chunks, top_k=3)
        ethics_set.update(found)

    bias_set = set()
    for q in bias_queries:
        found = search_local_index(q, local_index, current_chunks, top_k=3)
        bias_set.update(found)

    fallacy_set = set()
    for q in fallacy_queries:
        found = search_local_index(q, local_index, current_chunks, top_k=3)
        fallacy_set.update(found)

    # E. Calculate "Top Combined" chunks
    # We create a super-query to find the absolute worst offenders
    all_problematic = ethics_set.union(bias_set).union(fallacy_set)
    
    combined_query = "bias, ethics violations, logical fallacies, discrimination, unfair assumptions"
    top_chunks = search_local_index(combined_query, local_index, current_chunks, top_k=5)

    # F. Construct & Return Response
    return BiasChunksResponse(
        ethics_chunks=list(ethics_set)[:3],
        bias_chunks=list(bias_set)[:3],
        fallacy_chunks=list(fallacy_set)[:3],
        combined_top_chunks=top_chunks,
        total_chunks_found=len(all_problematic)
    )

@app.get("/")
async def root():
    return {"message": "ClarityCheck AI Microservice is Running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)