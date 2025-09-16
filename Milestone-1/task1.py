import os
import json
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter

# Extra imports for saving
from docx import Document
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Input & output folders
input_folder = r"E:\BE\AI Based legal reference and case retrieval system\data"
output_folder = r"E:\BE\AI Based legal reference and case retrieval system\output_chunks"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Chunk parameters
splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)


def load_document(file_path):
    """Choose the correct loader based on file extension"""
    ext = file_path.split('.')[-1].lower()

    if ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "html":
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"❌ Unsupported file type: {ext}")
    return loader.load(), ext


def save_output(chunks, base_name, ext):
    """Save chunks back into the same format as input"""
    output_path = os.path.join(output_folder, f"{base_name}_chunks.{ext}")

    if ext == "txt":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(chunks))

    elif ext == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"chunks": chunks}, f, indent=2, ensure_ascii=False)

    elif ext == "docx":
        doc = Document()
        for chunk in chunks:
            doc.add_paragraph(chunk)
        doc.save(output_path)

    elif ext == "pdf":
        doc = SimpleDocTemplate(output_path)
        styles = getSampleStyleSheet()
        story = [Paragraph(chunk, styles["Normal"]) for chunk in chunks]
        doc.build(story)

    elif ext == "html":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("<html><body>\n")
            for chunk in chunks:
                f.write(f"<h3>{chunk.split(':',1)[0]}</h3><p>{chunk.split(':',1)[1]}</p>\n")
            f.write("</body></html>")

    else:
        raise ValueError(f"❌ Saving not implemented for: {ext}")

    return output_path


def process_file(filename):
    """Load, split into chunks, number them, and save in same format"""
    file_path = os.path.join(input_folder, filename)
    docs, ext = load_document(file_path)

    # Split into chunks
    chunks = splitter.split_documents(docs)

    # Add chunk numbers
    numbered_chunks = [f"Chunk {i+1}:\n{chunk.page_content}" for i, chunk in enumerate(chunks)]

    # Save output
    base_name = os.path.splitext(filename)[0]
    output_path = save_output(numbered_chunks, base_name, ext)

    print(f"✅ Saved {len(numbered_chunks)} chunks for '{filename}' → '{output_path}'")


def main():
    for filename in os.listdir(input_folder):
        try:
            process_file(filename)
        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")


if __name__ == "__main__":
    main()
