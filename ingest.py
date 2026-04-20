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
def chunk_pages(pages, min_length=100):
    chunks=[]
    for page_text in pages:
        raw_chunks=page_text.split("\n\n")
        for chunk in raw_chunks:
            cleaned=chunk.strip()
            if len(cleaned)>=min_length:
                chunks.append(cleaned)
    return chunks

# load
def build_vector_store(chunks):
    client=chromadb.PersistentClient(path="./chroma_db")

    # delete collection if already exist
    try:
        client.delete_collection("finding_aid")
    except:
        pass

    ef = embedding_functions.DefaultEmbeddingFunction()
    collection=client.get_or_create_collection(
        name="finding_aid",
        embedding_function=ef
    )

    # prepend chunk index
    documents=[f"[Chunk {i+1}] {chunk}" for i, chunk in enumerate(chunks)]
    ids=[f"chunk_{i}" for i in range(len(chunks))]

    collection.add(documents=documents, ids=ids)
    print(f"stored {len(chunks)} chunks in ChromaDB")
    return collection

# main 
if __name__=="__main__":
    pdf_path="data/Frida-Kahlo-Finding-Aid-June-2022.pdf"

    print("extracting text from PDF...")
    pages=extract_text(pdf_path)
    print(f"   Found {len(pages)} pages with text.")

    print("chunking...")
    chunks=chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks.")

    print("building vector store...")
    build_vector_store(chunks)

    print("\ningest complete! ready to query")