import os
import json
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Folders
INPUT_DIR = "data"               # folder with your documents
OUTPUT_DIR = "output_vectors"    # folder to store JSON embeddings
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunking parameters
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

# Load the Hugging Face embeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_document(file_path):
    """Load document based on extension"""
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


def process_files():
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            # Load and split into chunks
            docs, ext = load_document(file_path)
            chunks = splitter.split_documents(docs)

            vectors = []
            for i, chunk in enumerate(chunks, start=1):
                text = chunk.page_content.strip()
                embedding = embedding_model.embed_query(text)  # convert text to vector
                vectors.append({
                    "id": i,
                    "text": text,
                    "embedding": embedding
                })

            # Save vectors in a JSON file per document
            output_file = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_vectors.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(vectors, f, indent=2, ensure_ascii=False)

            print(f"✅ Processed {filename} → {output_file} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"❌ Failed {filename}: {e}")


if __name__ == "__main__":
    process_files()
