import os
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredHTMLLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

# For saving in docx
from docx import Document
# For saving in PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# Input & output folders
INPUT_DIR = "data"
OUTPUT_DIR = "output_chunks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Chunk splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)


def load_file(file_path):
    """Load file based on extension"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        try:
            return TextLoader(file_path, encoding="utf-8").load(), ext
        except Exception:
            return TextLoader(file_path, encoding="latin-1").load(), ext
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load(), ext
    elif ext == ".docx":
        return UnstructuredWordDocumentLoader(file_path).load(), ext
    elif ext == ".html":
        return UnstructuredHTMLLoader(file_path).load(), ext
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def save_chunks(chunks, output_file, ext):
    """Save the chunks in the same format as the original file"""
    numbered_chunks = [f"Chunk {i+1}:\n{chunk.page_content.strip()}" for i, chunk in enumerate(chunks)]

    if ext == ".txt":
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(numbered_chunks))

    elif ext == ".docx":
        doc = Document()
        for chunk_text in numbered_chunks:
            doc.add_paragraph(chunk_text)
        doc.save(output_file)

    elif ext == ".pdf":
        doc = SimpleDocTemplate(output_file)
        styles = getSampleStyleSheet()
        story = [Paragraph(chunk_text, styles["Normal"]) for chunk_text in numbered_chunks]
        doc.build(story)

    elif ext == ".html":
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("<html><body>\n")
            for chunk_text in numbered_chunks:
                # Split first line as heading if possible
                lines = chunk_text.split("\n", 1)
                title = lines[0] if len(lines) > 0 else ""
                content = lines[1] if len(lines) > 1 else ""
                f.write(f"<h3>{title}</h3><p>{content}</p>\n")
            f.write("</body></html>")

    else:
        raise ValueError(f"Saving not implemented for {ext}")


def process_files():
    files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

    for filename in files:
        file_path = os.path.join(INPUT_DIR, filename)
        try:
            docs, ext = load_file(file_path)
            chunks = splitter.split_documents(docs)

            output_file = os.path.join(OUTPUT_DIR, filename)
            save_chunks(chunks, output_file, ext)

            print(f"✅ Processed {filename} → {output_file} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")


if __name__ == "__main__":
    process_files()
