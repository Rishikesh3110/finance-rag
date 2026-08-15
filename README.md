# 📊 HCLTech Finance RAG

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about **HCLTech's FY26 quarterly financial reports**.

The system retrieves relevant information from HCLTech's financial reports and uses **Llama 3**, running locally through **Ollama**, to generate grounded, source-cited answers — no paid API, no internet dependency once set up.

---

## 🎯 Project Objective

The objective of this project is to build a financial question-answering system using RAG.

Instead of asking an AI model to answer from its general knowledge (where it could hallucinate numbers), this application retrieves the actual relevant passages from HCLTech's FY26 financial reports and uses **only that retrieved context** to generate the answer.

The system supports questions related to:
- Revenue
- EBIT
- Net Income
- Quarterly financial performance
- QoQ (quarter-on-quarter) growth
- YoY (year-on-year) growth
- Management commentary and outlook

If a question can't be answered from the loaded reports, the system says so instead of guessing.

---

## 🧠 What is RAG?

RAG stands for **Retrieval-Augmented Generation**. The system follows these steps:

1. Load HCLTech's FY26 quarterly financial reports (PDFs).
2. Extract text from each PDF.
3. Split the extracted text into smaller chunks.
4. Generate embeddings for each chunk using `nomic-embed-text`.
5. Store the embeddings in **ChromaDB**.
6. Retrieve the most relevant chunks for a user's question.
7. Send the retrieved context, along with the question, to **Llama 3**.
8. Llama 3 generates a grounded financial answer using only that context.
9. Display the answer along with the source document(s) it came from.

---

## 🏗️ System Architecture

```text
HCLTech FY26 PDF Reports
         │
         ▼
    PDF Extraction        (extract_text.py)
         │
         ▼
     Text Files            (data/*.txt)
         │
         ▼
     Text Chunking         (chunk_text.py)
         │
         ▼
   chunks.txt
         │
         ▼
  Nomic Text Embeddings    (create_vector_db.py)
         │
         ▼
      ChromaDB
         │
         ▼
     User Question   ◄──────────────┐
         │                          │
         ▼                          │
  Question Embedding                │
         │                          │
         ▼                          │
Relevant Document Chunks            │  (rag.py)
         │                          │
         ▼                          │
      Llama 3 (via Ollama)          │
         │                          │
         ▼                          │
   Financial Answer + Source Chunks ┘
         │
         ▼
   Displayed in app.py (UI)
```

---

## 📁 Project Structure

```
HCLTech-Finance-RAG/
├── data/
│   ├── HCLTech_Q1_FY26.pdf
│   ├── HCLTech_Q1_FY26.txt
│   ├── HCLTech_Q2_FY26.pdf
│   ├── HCLTech_Q2_FY26.txt
│   ├── HCLTech_Q3_FY26.pdf
│   ├── HCLTech_Q3_FY26.txt
│   ├── HCLTech_Q4_FY26.pdf
│   ├── HCLTech_Q4_FY26.txt
│   └── chunks.txt              # all chunked text, ready for embedding
├── src/
│   ├── extract_text.py         # PDF -> raw .txt per report
│   ├── chunk_text.py           # raw .txt -> chunks.txt
│   ├── create_vector_db.py     # chunks.txt -> ChromaDB (embeddings)
│   └── rag.py                  # retrieval + prompt + Llama 3 call
├── app.py                      # UI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component        | Tool                         |
|-------------------|-------------------------------|
| LLM                | Llama 3 (via [Ollama](https://ollama.com)) |
| Embedding model    | `nomic-embed-text` (via Ollama) |
| Vector database    | ChromaDB                     |
| PDF extraction     | PyPDF / pdfplumber            |
| App framework      | Streamlit                    |
| Language           | Python 3.11+                 |

Everything runs **locally** — no OpenAI key, no billing, no rate limits.

---

## 🚀 Setup and Run Instructions

### 1. Install Ollama
Download and install from [ollama.com](https://ollama.com) (Windows/Mac/Linux).

### 2. Pull the required models
```bash
ollama pull llama3
ollama pull nomic-embed-text
```
Keep the Ollama app/service running in the background — the scripts below call it locally.

### 3. Clone the repo and set up a virtual environment
```bash
git clone <your-repo-url>
cd HCLTech-Finance-RAG

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Build the pipeline (run once, in order)
```bash
python src/extract_text.py       # PDFs in data/ -> .txt files
python src/chunk_text.py         # .txt files -> data/chunks.txt
python src/create_vector_db.py   # chunks.txt -> embeddings stored in ChromaDB
```

### 6. Run the app
```bash
streamlit run app.py
```
Open the URL Streamlit prints (usually `http://localhost:8501`), type a question, and get an answer with cited sources.

---

## ❓ Sample Questions to Try

1. What was HCLTech's revenue in Q1 FY26?
2. How did net income change from Q1 to Q4 FY26?
3. What was the QoQ revenue growth between Q3 and Q4 FY26?
4. What was the YoY growth in EBIT for FY26?
5. What did management say about the demand outlook?
6. Summarize HCLTech's Q4 FY26 performance in three lines.
7. *(Trap question)* What is the CEO's personal shareholding in 2015?
   → Should be refused: *"This information is not available in the loaded reports."*

> TODO: run each of these in the app and paste the exact answer + cited source here.

---

## 📝 Notes

- All embeddings and generation run **fully offline** through Ollama — no API key required anywhere in this project.
- Answers are grounded strictly in the retrieved chunks; if the context doesn't contain the answer, the app says so instead of guessing.
- `chunks.txt` and the ChromaDB folder are regenerated by running the pipeline scripts — they don't need to be committed if excluded in `.gitignore` (recommended for the DB folder specifically; `chunks.txt` can stay for transparency).

---

## 🎥 Demo Video
> TODO: link to a short video showing PDF ingestion, a few questions being answered with sources, and the trap question being correctly refused.
