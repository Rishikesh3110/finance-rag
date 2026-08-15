"""
ingest.py
Loads PDFs, chunks them, embeds the chunks, and stores them in a
persisted ChromaDB collection.

Used by app.py (Streamlit) and api/main.py (FastAPI /ingest).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ---- Config (change here if you want, keep within assignment limits) ----
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_reports")
CHUNK_SIZE = 1200          # within the 800-1200 range; keeps tables intact
CHUNK_OVERLAP = 150        # within the 100-200 range

# ---- Embedding model ----
# NOTE ON DEVIATION FROM SPEC: the assignment asks for OpenAI's
# text-embedding-3-small. That requires a funded OpenAI account. I did not
# have budget for that, so I substituted Google's free embedding-001
# model (same Google API key already used for the answering model -- no
# extra signup, no local ML packages/torch, no cost). It does the same
# job -- turns each chunk into a vector for semantic search in Chroma.
# Disclosed in the README as required.
EMBEDDING_MODEL = "models/embedding-001"


def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)


def get_vectorstore():
    """Open (or create) the persisted Chroma collection."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
    )


def load_and_chunk(pdf_paths):
    """
    pdf_paths: list of file paths (str or Path) to PDF files.
    Returns a list of LangChain Document chunks, each with
    metadata: {"source": <filename>, "page": <page number, 1-indexed>}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for path in pdf_paths:
        path = Path(path)
        loader = PyPDFLoader(str(path))
        pages = loader.load()  # one Document per page, metadata["page"] is 0-indexed

        for page_doc in pages:
            page_doc.metadata["source"] = path.name
            # PyPDFLoader gives 0-indexed pages; make it human-friendly (1-indexed)
            page_doc.metadata["page"] = page_doc.metadata.get("page", 0) + 1

        chunks = splitter.split_documents(pages)
        all_chunks.extend(chunks)

    return all_chunks


def ingest_files(pdf_paths):
    """
    Full pipeline: load -> chunk -> embed -> store.
    Returns a summary dict, e.g. {"files": 3, "chunks": 214}
    """
    chunks = load_and_chunk(pdf_paths)
    if not chunks:
        return {"files": len(pdf_paths), "chunks": 0}

    vectordb = get_vectorstore()
    vectordb.add_documents(chunks)

    return {"files": len(pdf_paths), "chunks": len(chunks)}


def collection_stats():
    vectordb = get_vectorstore()
    count = vectordb._collection.count()
    return {
        "collection_name": COLLECTION_NAME,
        "total_chunks": count,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": "gemini-2.5-flash",
    }


if __name__ == "__main__":
    # Quick manual test: index everything currently sitting in data/
    data_dir = Path(__file__).parent / "data"
    pdfs = sorted(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {data_dir}. Put your quarterly PDFs there first.")
    else:
        print(f"Indexing {len(pdfs)} file(s) from {data_dir} ...")
        result = ingest_files(pdfs)
        print("Done:", result)
        print("Stats:", collection_stats())
