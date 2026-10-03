# 🤖 Campus Ordinance & Syllabus RAG Assistant

> A Retrieval-Augmented Generation (RAG) assistant that indexes college PDFs and syllabi to provide accurate, cited answers to student queries.

[![Status: Blueprint Starter](https://img.shields.io/badge/Status-Starter_Blueprint-brightgreen.svg?style=flat-square)](#)
[![Domain: Generative AI](https://img.shields.io/badge/Domain-GenAI_|_RAG-red.svg?style=flat-square)](#)

---

## 🎯 Problem Statement
Students often struggle to find answers buried in 100+ page university grading schemes, syllabus regulations, and academic calendars.

## 💡 Solution
- **Vector Database:** ChromaDB / FAISS for storing document chunks.
- **Embeddings:** Hugging Face `sentence-transformers/all-MiniLM-L6-v2`.
- **Generation:** Local Ollama (Llama 3 / Mistral) or cloud LLM APIs with citation sources.

## 🚀 Quickstart

### Option 1: Run Instantly (Zero Dependencies)
```bash
python rag_pipeline.py
```

### Option 2: Full Vector DB & Embeddings Environment
```bash
pip install -r requirements.txt
python rag_pipeline.py
```

## 🤝 Open Contributions (Good First Issues)
- [ ] Add PDF ingestion script using PyPDF or LangChain PDFLoader to ingest official RGPV/OCT PDFs.
- [ ] Implement a Streamlit chat UI with chat history.
- [ ] Add source citation display showing exact page numbers and PDF links.
