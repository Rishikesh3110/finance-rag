"""
rag.py
Retrieval + prompt construction + call to GPT-4o.

Used by app.py (Streamlit) and api/main.py (FastAPI /ask).
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from ingest import get_vectorstore

load_dotenv()

# NOTE ON DEVIATION FROM SPEC: the assignment asks for GPT-4o, which needs a
# funded OpenAI account. I substituted Google's Gemini 1.5 Flash instead,
# accessed through a free API key from https://aistudio.google.com/apikey
# (no credit card required, generous free daily quota). The rest of the
# pipeline — retrieval, prompt construction, the honest-refusal rule, and
# source citation — is unchanged. Disclosed in the README as required.
LLM_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.1  # within the required 0-0.2 range

SYSTEM_PROMPT = """You are a careful financial research assistant.
Answer ONLY using the context excerpts provided below, which were retrieved
from the uploaded quarterly report PDFs. Cite specific figures exactly as
they appear in the context.

If the context does not contain the answer, reply exactly:
"This information is not available in the uploaded documents."
Do not guess, estimate, or use outside knowledge. Never invent a number."""


def _format_context(docs):
    """Turn retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")
        blocks.append(f"[Chunk {i} | {src} | page {page}]\n{d.page_content}")
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int = 5):
    """
    Retrieves top_k chunks for the question, asks GPT-4o to answer strictly
    from that context, and returns {"answer": str, "sources": [...]}.
    """
    vectordb = get_vectorstore()
    docs = vectordb.similarity_search(question, k=top_k)

    if not docs:
        return {
            "answer": "This information is not available in the uploaded documents.",
            "sources": [],
        }

    context = _format_context(docs)

    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]
    response = llm.invoke(messages)

    sources = [
        {"file": d.metadata.get("source", "unknown"), "page": d.metadata.get("page", "?")}
        for d in docs
    ]

    return {"answer": response.content, "sources": sources}


if __name__ == "__main__":
    q = input("Ask a question: ")
    result = answer_question(q)
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for s in result["sources"]:
        print(" -", s["file"], "page", s["page"])
