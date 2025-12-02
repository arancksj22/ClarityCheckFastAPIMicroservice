from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="ClarityCheckAIMicroservice")

# --- 1. Load Model Globally ---
print("Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model Loaded!")

# --- 2. Data Models ---
class TextInput(BaseModel):
    text: str

# NEW: Simple response format for the "Smart Filter" approach
class FilteredTextResponse(BaseModel):
    filtered_text: str

# --- 3. Helper Functions ---
def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    chunks = []
    overlap = 50 
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
    return chunks

def search_local_index(query: str, index, chunks: List[str], top_k: int = 3) -> List[str]:
    if index is None or not chunks:
        return []
    
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    k_to_search = min(top_k, len(chunks))
    if k_to_search == 0:
        return []

    scores, indices = index.search(query_embedding, k_to_search)
    
    relevant_chunks = []
    for i, score in enumerate(scores[0]):
        if score > 0.25: # Threshold
            idx = indices[0][i]
            if 0 <= idx < len(chunks):
                relevant_chunks.append(chunks[idx])
    
    return relevant_chunks

# --- 4. Main Endpoint ---
@app.post("/analyze", response_model=FilteredTextResponse)
async def analyze_document(input_data: TextInput):
    """
    Accepts full text, finds the 7 most problematic paragraphs, 
    and returns them as a single string.
    """
    
    # A. Chunking
    current_chunks = chunk_text(input_data.text)
    
    if not current_chunks:
        return FilteredTextResponse(filtered_text="")

    # B. Indexing (Stateless)
    embeddings = model.encode(current_chunks)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    local_index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)
    local_index.add(embeddings)

    # C. Queries
    ethics_queries = ["ethical concerns", "misconduct", "informed consent"]
    bias_queries = ["gender bias", "racial bias", "demographic assumptions"]
    fallacy_queries = ["logical fallacies", "unsupported claims", "correlation vs causation"]

    # D. Search & Aggregate
    found_chunks = set()
    
    for q in ethics_queries:
        found_chunks.update(search_local_index(q, local_index, current_chunks, top_k=2))
        
    for q in bias_queries:
        found_chunks.update(search_local_index(q, local_index, current_chunks, top_k=2))
        
    for q in fallacy_queries:
        found_chunks.update(search_local_index(q, local_index, current_chunks, top_k=2))

    # E. Final Selection (Top 7 combined)
    # If we found too few specific ones, do a general sweep to fill the buffer
    if len(found_chunks) < 5:
        general_query = "bias ethics fallacies flaws"
        found_chunks.update(search_local_index(general_query, local_index, current_chunks, top_k=5))

    # Limit to top 7 to fit context window
    final_list = list(found_chunks)[:7]
    
    # Join with double newlines to simulate paragraphs
    result_string = "\n\n".join(final_list)

    return FilteredTextResponse(filtered_text=result_string)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)