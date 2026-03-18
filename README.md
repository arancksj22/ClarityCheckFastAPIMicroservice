# ClarityCheck FastAPI Microservice 

The **ClarityCheck AI Microservice** is the high-performance "Smart Filter" at the heart of the ClarityCheck platform. It specializes in transforming massive research PDFs into digestible, semantically rich data chunks. By using a Retrieval-Augmented Generation (RAG) approach, it identifies the most problematic sections of a document—focusing on ethics, bias, and logical fallacies—before they are processed by the core orchestration layer.

## Key Features

- **Semantic Chunking:** Automatically segments text into 500-character chunks with a 50-character overlap to preserve context across boundaries.
- **Vector Search (FAISS):** Implements `IndexFlatIP` (Inner Product) similarity search for lightning-fast retrieval of relevant text.
- **Multi-Query Targeting:** Executes parallel semantic searches for three distinct categories:
  - **Ethics:** Identifies misconduct, consent issues, and ethical breaches.
  - **Bias:** Highlights gender, racial, and demographic assumptions.
  - **Fallacies:** Detects unsupported claims and correlation-causation errors.
- **Stateless Execution:** Designed for horizontal scaling within a Dockerized environment.

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **Framework** | FastAPI (Python 3.9+) |
| **Embeddings** | Hugging Face `all-MiniLM-L6-v2` (384-dimensional) |
| **Vector DB** | FAISS (Facebook AI Similarity Search) |
| **Processing** | NumPy, SentenceTransformers |
| **Server** | Uvicorn |

## Architecture Role

As shown in the system architecture, this microservice acts as the bridge between **AWS S3** (document storage) and the **Spring Boot Microservice** (agent orchestration). 

1. **Input:** Receives raw text extracted from PDFs.
2. **Embedding:** Generates vector representations of every paragraph.
3. **Retrieval:** Uses predefined "Problematic Queries" to find the top 7 most relevant chunks via FAISS.
4. **Output:** Returns a filtered "Context Buffer" that fits perfectly within LLM context windows for final analysis.

## Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional)

### Installation
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/arancksj22/ClarityCheck-AI-Microservice.git](https://github.com/arancksj22/ClarityCheck-AI-Microservice.git)
   cd ClarityCheck-AI-Microservice
