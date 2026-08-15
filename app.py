"""
app.py
Streamlit interface: upload PDFs, index them, ask questions, see
answers with sources. Indexed data persists in ./chroma_db across restarts.
"""

import os
from pathlib import Path
import streamlit as st

from ingest import ingest_files, collection_stats
from rag import answer_question

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Quarterly Results RAG", layout="wide")
st.title("📊 Quarterly Financial Reports — RAG Assistant")
st.caption("Upload quarterly PDFs, index them, then ask questions in plain English.")

# ---------------- Upload + Index ----------------
st.header("1. Upload & Index")
uploaded_files = st.file_uploader(
    "Upload one or more quarterly report PDFs", type=["pdf"], accept_multiple_files=True
)

if st.button("Index uploaded files", disabled=not uploaded_files):
    saved_paths = []
    for f in uploaded_files:
        out_path = DATA_DIR / f.name
        with open(out_path, "wb") as fh:
            fh.write(f.getbuffer())
        saved_paths.append(out_path)

    with st.spinner("Reading, chunking, and embedding..."):
        result = ingest_files(saved_paths)

    st.success(f"{result['files']} files processed, {result['chunks']} chunks stored.")

# Show current collection status (works even after a restart, since Chroma is persisted)
try:
    stats = collection_stats()
    st.info(
        f"Collection **{stats['collection_name']}** currently has "
        f"**{stats['total_chunks']}** chunks stored "
        f"(embeddings: {stats['embedding_model']}, LLM: {stats['llm_model']})."
    )
except Exception:
    st.info("No documents indexed yet.")

st.divider()

# ---------------- Ask ----------------
st.header("2. Ask a question")
question = st.text_input("Your question", placeholder="What was total revenue in the most recent quarter?")
top_k = st.slider("Number of chunks to retrieve (top_k)", min_value=3, max_value=8, value=5)

if st.button("Ask", disabled=not question):
    with st.spinner("Retrieving relevant chunks and asking GPT-4o..."):
        result = answer_question(question, top_k=top_k)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Sources")
    if result["sources"]:
        for s in result["sources"]:
            st.markdown(f"- **{s['file']}**, page {s['page']}")
    else:
        st.markdown("_No sources — nothing relevant was found in the index._")
