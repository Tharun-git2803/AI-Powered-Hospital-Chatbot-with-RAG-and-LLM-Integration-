from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ======================
# LOAD PDF
# ======================
pdf_path = "RAG_Dataset_Hospital.pdf"

loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

# ======================
# SPLIT TEXT
# ======================
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

# ======================
# EMBEDDINGS
# ======================
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ======================
# STORE IN CHROMA DB
# ======================
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)

vectorstore.persist()

print("✅ Ingestion completed successfully!")