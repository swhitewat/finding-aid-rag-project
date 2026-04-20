import os
import fitz  # pymupdf
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

# extract text from PDF
def extract_text(pdf_path):
    doc=fitz.open(pdf_path)
    pages=[]
    for page in doc:
        text=page.get_text()
        if text.strip():
            pages.append(text.strip())
    return pages

# heuristic chunking 

# load

# main 
