🩺 AI Hospital Assistant (RAG + LLM Hybrid System)

An intelligent AI-powered hospital assistant that answers medical and hospital-related queries using Retrieval-Augmented Generation (RAG) combined with Large Language Models (LLMs).

It retrieves accurate answers from hospital PDF data and uses AI fallback when information is not available in the database.

🚀 Features
📄 PDF-based hospital knowledge ingestion
🧠 Semantic search using HuggingFace embeddings
📚 Vector database using ChromaDB
🤖 Hybrid AI system (RAG + LLM fallback)
💬 Chat-like interface using Streamlit
⚡ Smart query routing (decides RAG vs AI automatically)
🔒 Safe medical assistant (no harmful prescriptions)
🏗️ Project Architecture
PDF Document
     ↓
Chunking (Text Splitter)
     ↓
Embeddings (MiniLM)
     ↓
Chroma Vector DB
     ↓
User Query
     ↓
Similarity Search
     ↓
┌───────────────┬────────────────┐
│ High Match     │ Low/No Match   │
│ RAG Response   │ LLM Response   │
└───────────────┴────────────────┘
🛠️ Tech Stack
Python
Streamlit
LangChain
ChromaDB
HuggingFace Embeddings
Groq LLM (Llama 3.3)
PyMuPDF
