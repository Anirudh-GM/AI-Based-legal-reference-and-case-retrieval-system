import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ------------------------------
# 1️⃣ Setup
# ------------------------------
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("⚠️ PINECONE_API_KEY not found in .env")

# Pinecone client
pc = Pinecone(api_key=api_key)

index_name = "legal-index-v2"  # ✅ NEW INDEX NAME

# Create Pinecone index if not exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # same as embedding model output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created new index '{index_name}'")
else:
    print(f"ℹ️ Index '{index_name}' already exists — uploading to it.")

# Embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# ------------------------------
# 2️⃣ Load supported document types
# ------------------------------
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        try:
            return TextLoader(file_path, encoding="utf-8").load()
        except:
            return TextLoader(file_path, encoding="latin-1").load()
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load()
    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path).load()
    else:
        print(f"⚠️ Skipping unsupported file type: {file_path}")
        return []

# ------------------------------
# 3️⃣ Process local folder
# ------------------------------
def process_local_files(input_dir="data_pinecone"):
    docs = []
    if not os.path.exists(input_dir):
        print(f"❌ Folder '{input_dir}' not found.")
        return docs

    for root, _, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            if os.path.isfile(file_path):
                try:
                    file_docs = load_document(file_path)
                    for d in file_docs:
                        d.metadata = {
                            "source": "local",
                            "filename": filename,
                            "path": file_path
                        }
                    docs.extend(file_docs)
                    print(f"📄 Loaded {filename} ({len(file_docs)} pages)")
                except Exception as e:
                    print(f"❌ Failed to load {filename}: {e}")
    return docs

# ------------------------------
# 4️⃣ Main upload pipeline
# ------------------------------
def main():
    print("🚀 Starting document upload to Pinecone...")

    # Load all local documents
    local_docs = process_local_files("data_pinecone")
    print(f"🔹 Total documents before splitting: {len(local_docs)}")

    if not local_docs:
        print("❌ No documents found in data_pinecone/")
        return

    # Split into smaller chunks
    chunks = splitter.split_documents(local_docs)
    print(f"🔹 Total chunks after splitting: {len(chunks)}")

    # Upload to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
        namespace="legal_cases",
    )

    print(f"✅ Successfully uploaded {len(chunks)} chunks to Pinecone index '{index_name}'")

if __name__ == "__main__":
    main()
