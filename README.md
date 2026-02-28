# DocSearch — Semantic Search App (FastAPI)

## Description
DocSearch is a smart semantic search engine for documents.  
It helps users quickly find relevant information in large document collections (reports, procedures, recommendations, use cases, etc.).

## Features
- Upload files directly to the app: **PDF, DOCX, TXT, JSON**.
- Extract text from uploaded files and split it into chunks (small text segments).
- Convert each chunk into a semantic embedding using **SentenceTransformer (`all-MiniLM-L6-v2`)**.
- Store embeddings in memory and use **FAISS** for fast similarity search.

### Query workflow
- Generate an embedding for the query.
- Compare the query embedding against the document chunk embeddings.
- Rank results by cosine similarity.
- Return the **top 3 most relevant chunks**, including:
  - text
  - source
  - similarity score

## How to Run

### 1. Install dependencies

pip install -r requirements.txt


2. Run the app
python app.py

3. Open in browser

http://localhost:8000

Upload files and type a query to get results.

Core Dependencies (requirements.txt)
fastapi==0.115.5
uvicorn==0.32.1
sentence-transformers==3.3.1
faiss-cpu==1.13.2
numpy==1.26.0
pdfplumber==0.10.1
PyPDF2==3.1.1
python-docx==0.8.12
pydantic==1.10.11
Additional Notes

No static/ folder or index.html needed — the frontend UI is embedded directly inside app.py.

Embeddings are cached on disk to speed up future searches.

Anyone can upload files and start searching immediately without complex setup.
