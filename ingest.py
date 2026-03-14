import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def start_ingestion():
    # 1. Path to your medical PDF
    pdf_path = "data/medhal.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: {pdf_path} not found!")
        return

    print("📖 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()

    # 2. Split into 500-character chunks with overlap
    # This prevents facts from being cut off mid-sentence
    print("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(data)

    # 3. Create Vector Store using MiniLM
    print("🧬 Generating embeddings (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # This creates a folder called 'chroma_db' to store the 'Source of Truth'
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    
    print(f"✅ Success! {len(chunks)} chunks stored in ./chroma_db")

if __name__ == "__main__":
    start_ingestion()