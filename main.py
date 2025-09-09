from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List