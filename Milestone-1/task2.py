from datasets import load_dataset
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load dataset
dataset = load_dataset("NahOR102/Indian-IPC-Laws")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    length_function=len
)

output = []

for i, item in enumerate(dataset["train"]):
    messages = item["messages"]

    # Extract question and answer from messages
    question = ""
    answer = ""
    for msg in messages:
        if msg["role"] == "user":
            question = msg["content"]
        elif msg["role"] == "assistant":
            answer = msg["content"]

    if not question or not answer:
        continue  # skip if missing

    # Combine into Q/A text
    text = f"Q: {question}\nA: {answer}"

    # Split into chunks
    chunks = text_splitter.split_text(text)

    # Save chunks with metadata
    for j, chunk in enumerate(chunks, start=1):
        output.append({
            "question": question,
            "answer": answer,
            "chunk_number": j,
            "chunk_text": chunk
        })

# Save output
with open("ipc_chunks.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"✅ Processed {len(output)} chunks. Saved to ipc_chunks.json")
