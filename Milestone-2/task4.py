import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load API key from .env
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("⚠️ API key not found. Please set PINECONE_API_KEY in your .env file.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Index name
index_name = "test-index"

# Create index if it doesn’t exist
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=8,  # dimension of your vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"✅ Created index '{index_name}'")

# Connect to index
index = pc.Index(index_name)
print(f"✅ Connected to index: {index_name}")

# ---------- CRUD Operations ----------

# C - Create / Insert vectors
vectors_to_upsert = [
    {"id": "vec1", "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], "metadata": {"label": "first"}},
    {"id": "vec2", "values": [0.2, 0.1, 0.4, 0.6, 0.9, 0.7, 0.5, 0.3], "metadata": {"label": "second"}},
]
index.upsert(vectors=vectors_to_upsert)
print("✅ Inserted vectors")

# R - Read / Query
query_results = index.query(vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], top_k=2, include_metadata=True)
print("🔍 Query results:", query_results)

# U - Update (just upsert with same ID)
updated_vector = {"id": "vec1", "values": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], "metadata": {"label": "updated"}}
index.upsert(vectors=[updated_vector])
print("✅ Updated vec1")

# D - Delete
index.delete(ids=["vec2"])
print("🗑️ Deleted vec2")

# Final check
print("📌 Index stats:", index.describe_index_stats())
