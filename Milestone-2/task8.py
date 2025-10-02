import os
import warnings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from system_template import SYSTEM_TEMPLATE

# ------------------------------
# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------------
# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY not found in .env")
if not PINECONE_API_KEY:
    raise ValueError("⚠️ PINECONE_API_KEY not found in .env")

# ------------------------------
# Pinecone index and embeddings
INDEX_NAME = "legal-index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)
print(f"✅ Connected to vectorstore: {INDEX_NAME}")

# ------------------------------
# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("✅ Retriever created (k=5)")

# ------------------------------
# Chat LLM
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.5,
    api_key=OPENAI_API_KEY
)

# ------------------------------
# Custom prompt for final QA
QA_PROMPT = PromptTemplate(
    template=SYSTEM_TEMPLATE + "\n\n**Context from retrieved documents:**\n{context}\n\n**User Query:** {question}\n\n**AI Response:**",
    input_variables=["context", "question"]
)

# ------------------------------
# RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=False
)

# ------------------------------
# User input loop
print("\n🔎 Enter your legal query (or type 'exit' to quit):")
while True:
    query = input("\n> ").strip()
    if query.lower() in ["exit", "quit"]:
        print("Exiting... ✅")
        break
    if not query:
        continue

    response = qa_chain.run(query)
    print("\n📜 AI Response:\n")
    print(response)