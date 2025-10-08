import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: Upload & Load raw PDF(s)
pdfs_directory = 'pdfs/'

def upload_pdf(file):
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

file_path = os.getenv("PDF_FILE_PATH", "universal_declaration_of_human_rights.pdf")
documents = load_pdf(file_path)

# Step 2: Create Chunks
def create_chunks(documents): 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)

# Step 3: Setup Embeddings Model (Use DeepSeek R1 with Ollama)
def get_embedding_model():
    # You can switch to OllamaEmbeddings if needed
    # return OllamaEmbeddings(model=ollama_model_name)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Index Documents
FAISS_DB_PATH = os.getenv("FAISS_DB_PATH", "vectorstore/db_faiss")
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model())
faiss_db.save_local(FAISS_DB_PATH)
