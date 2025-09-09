from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

app = FastAPI(title="ClarityCheck AI Service")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')