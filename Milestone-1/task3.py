import os
import json
from datasets import load_dataset
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings   # ✅ new import
from langchain.schema import Document

# Folders
INPUT_DIR = "data"               # local folder with your documents
OUTPUT_DIR = "output_vectors"    # folder to store JSON embeddings
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunking parameters
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_document(file_path):
    """Load local document based on extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        try:
            return TextLoader(file_path, encoding="utf-8").load(), ext
        except:
            return TextLoader(file_path, encoding="latin-1").load(), ext
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load(), ext
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load(), ext
    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path).load(), ext
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def embed_and_save(chunks, output_file):
    """Convert chunks into embeddings and save as JSON"""
    vectors = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk.page_content.strip()
        if not text:
            continue
        embedding = embedding_model.embed_documents([text])[0]  # ✅ better for docs
        vectors.append({
            "id": i,
            "text": text,
            "embedding": embedding
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(vectors, f, indent=2, ensure_ascii=False)


def process_local_files():
    """Process documents from local folder"""
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]
    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            docs, ext = load_document(file_path)
            chunks = splitter.split_documents(docs)

            output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_vectors.json")
            embed_and_save(chunks, output_file)

            print(f"✅ Local: {filename} → {output_file} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"❌ Failed {filename}: {e}")


def process_hf_dataset():
    """Process Hugging Face dataset NahOR102/Indian-IPC-Laws"""
    dataset = load_dataset("NahOR102/Indian-IPC-Laws")

    all_texts = []
    for row in dataset["train"]["messages"]:
        if isinstance(row, list):  # expected format: list of {role, content}
            text = "\n".join([f"[{m.get('role', 'unknown')}] {m.get('content', '')}" for m in row])
        elif isinstance(row, dict) and "content" in row:
            # sometimes dataset may contain dict instead of list
            text = row["content"]
        else:
            # fallback: convert to string
            text = str(row)

        if text.strip():
            all_texts.append(text)

    # Wrap texts as pseudo-documents
    docs = [Document(page_content=str(t)) for t in all_texts]

    # Split into chunks
    chunks = splitter.split_documents(docs)

    output_file = os.path.join(OUTPUT_DIR, "Indian_IPC_Laws_vectors.json")
    embed_and_save(chunks, output_file)

    print(f"✅ HuggingFace dataset processed → {output_file} ({len(chunks)} chunks)")



if __name__ == "__main__":
    process_local_files()
    process_hf_dataset()