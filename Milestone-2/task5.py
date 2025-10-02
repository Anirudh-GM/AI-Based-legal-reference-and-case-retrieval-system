import os
from dotenv import load_dotenv
from datasets import load_dataset

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# ------------------------------
# Setup
# ------------------------------
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("⚠️ API key not found. Please set PINECONE_API_KEY in your .env file.")

# Pinecone client
pc = Pinecone(api_key=api_key)
index_name = "legal-index"

# Create Pinecone index if not exists
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # must match embedding model output size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created index '{index_name}'")

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# ------------------------------
# Helper: Load local documents
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
        raise ValueError(f"Unsupported file type: {ext}")

# ------------------------------
# Process local files
# ------------------------------
def process_local_files(input_dir="data"):
    docs = []
    if not os.path.exists(input_dir):
        return docs

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        if os.path.isfile(file_path):
            try:
                docs.extend(load_document(file_path))
                print(f"📄 Loaded {filename}")
            except Exception as e:
                print(f"❌ Failed {filename}: {e}")
    return docs

# ------------------------------
# Process HuggingFace dataset
# ------------------------------
def process_hf_dataset():
    dataset = load_dataset("NahOR102/Indian-IPC-Laws")
    docs = []
    for row in dataset["train"]["messages"]:
        if isinstance(row, list):
            text = "\n".join([f"[{m.get('role', 'unknown')}] {m.get('content', '')}" for m in row])
        elif isinstance(row, dict) and "content" in row:
            text = row["content"]
        else:
            text = str(row)

        if text.strip():
            docs.append(Document(page_content=text))
    print(f"📚 Loaded {len(docs)} documents from HuggingFace dataset")
    return docs

# ------------------------------
# Main pipeline
# ------------------------------
def main():
    # 1. Collect docs
    local_docs = process_local_files("data")
    hf_docs = process_hf_dataset()
    all_docs = local_docs + hf_docs

    # 2. Split into chunks
    chunks = splitter.split_documents(all_docs)
    print(f"🔹 Total chunks: {len(chunks)}")

    # 3. Upload to Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=index_name,
    )
    print(f"✅ Uploaded {len(chunks)} chunks to Pinecone index '{index_name}'")

if __name__ == "__main__":
    main()
