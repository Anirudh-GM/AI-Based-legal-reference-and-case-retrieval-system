import os
import warnings
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Suppress LangChain + Pydantic deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")

if not pinecone_key:
    raise ValueError("⚠️ PINECONE_API_KEY not found in .env")

# ✅ Use your Pinecone index
index_name = "legal-index"

# ✅ HuggingFace embeddings (no OpenAI needed)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Connect to existing index
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embedding
)
print(f"✅ Connected to Pinecone index: {index_name}")

# ✅ Create retriever (new API)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# ✅ Take user query
query = input("\n🔎 Enter your legal query: ")

# ✅ Fetch results (new API: retriever.invoke instead of get_relevant_documents)
results = retriever.invoke(query)

print("\n📌 Top 4 search results:")
for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content[:500])  # show first 500 chars
