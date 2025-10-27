# app.py
import os
import sqlite3
import hashlib
import streamlit as st
import warnings
import re
import threading
from dotenv import load_dotenv
import base64

# --- RAG / LLM imports ---
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# --- System prompt ---
try:
    from system_template import SYSTEM_TEMPLATE
except Exception:
    SYSTEM_TEMPLATE = "You are a legal assistant AI. Provide helpful and factual answers to law-related queries."

warnings.filterwarnings("ignore")

# ============================= Database Setup =============================
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")
db_lock = threading.Lock()

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# Create tables
with get_connection() as conn:
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            profile_pic TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()

# ============================= Helper Functions =============================
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_email(email: str) -> bool:
    return re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email) is not None

def is_valid_password(password: str) -> bool:
    return len(password) >= 6 and re.search(r'[A-Za-z]', password) and re.search(r'\d', password)

# ============================= User Functions =============================
def register_user(email, password, first_name, last_name, profile_pic_path):
    hashed = hash_password(password)
    try:
        with db_lock, get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO users (email, password, first_name, last_name, profile_pic) VALUES (?, ?, ?, ?, ?)",
                (email, hashed, first_name, last_name, profile_pic_path)
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(email, password):
    hashed = hash_password(password)
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, hashed))
        return c.fetchone() is not None

def fetch_user(email):
    if not email:
        return None
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT email, first_name, last_name, profile_pic FROM users WHERE email=?", (email,))
        return c.fetchone()

def update_profile(email, first_name, last_name, profile_pic_path=None):
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        if profile_pic_path:
            c.execute("UPDATE users SET first_name=?, last_name=?, profile_pic=? WHERE email=?",
                      (first_name, last_name, profile_pic_path, email))
        else:
            c.execute("UPDATE users SET first_name=?, last_name=? WHERE email=?",
                      (first_name, last_name, email))
        conn.commit()

def update_password(email, old_password, new_password):
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE email=?", (email,))
        result = c.fetchone()
        if not result or hash_password(old_password) != result[0]:
            return False
        hashed_new = hash_password(new_password)
        c.execute("UPDATE users SET password=? WHERE email=?", (hashed_new, email))
        conn.commit()
    return True

def update_email(old_email, new_email):
    if not is_valid_email(new_email):
        return "Invalid email format."
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE email=?", (new_email,))
        if c.fetchone():
            return "Email already in use."
        c.execute("SELECT profile_pic FROM users WHERE email=?", (old_email,))
        result = c.fetchone()
        old_pic = result[0] if result else None
        c.execute("UPDATE users SET email=? WHERE email=?", (new_email, old_email))
        conn.commit()
        if old_pic and os.path.exists(old_pic):
            ext = os.path.splitext(old_pic)[-1]
            new_path = f"profile_pics/{new_email.replace('@','_')}{ext}"
            os.rename(old_pic, new_path)
            c.execute("UPDATE users SET profile_pic=? WHERE email=?", (new_path, new_email))
            conn.commit()
    return "success"

def save_profile_photo(file, email):
    if file is None:
        return None
    ext = os.path.splitext(file.name)[-1]
    os.makedirs("profile_pics", exist_ok=True)
    file_path = f"profile_pics/{email.replace('@','_')}{ext}"
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())
    return file_path

def delete_profile_photo(email):
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT profile_pic FROM users WHERE email=?", (email,))
        result = c.fetchone()
        if result and result[0] and os.path.exists(result[0]):
            os.remove(result[0])
        c.execute("UPDATE users SET profile_pic=NULL WHERE email=?", (email,))
        conn.commit()

# ============================= Conversation Functions =============================
def create_conversation(email, title):
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO conversations (user_email, title) VALUES (?, ?)", (email, title))
        conn.commit()
        return c.lastrowid

def add_message(conversation_id, role, content):
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                  (conversation_id, role, content))
        conn.commit()

def get_conversations(email):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, title, created_at FROM conversations WHERE user_email=? ORDER BY created_at DESC", (email,))
        return c.fetchall()

def get_messages(conversation_id):
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id", (conversation_id,))
        return [{"role": r[0], "content": r[1]} for r in c.fetchall()]

def delete_conversation(conversation_id):
    with db_lock, get_connection() as conn:
        c = conn.cursor()
        c.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        c.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        conn.commit()

# ============================= Environment Setup =============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-index-new")

if not OPENAI_API_KEY:
    st.error("❌ Missing OPENAI API key. Configure your .env file!")
    st.stop()

# Optional Pinecone
vectorstore = None
retriever = None
if PINECONE_API_KEY:
    try:
        @st.cache_resource(show_spinner=False)
        def load_vectorstore():
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return PineconeVectorStore.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
        vectorstore = load_vectorstore()
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        st.warning(f"Could not initialize Pinecone vectorstore: {e}")

# ============================= LLM Setup =============================
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(model_name="gpt-3.5-turbo", api_key=OPENAI_API_KEY)

llm = load_llm()

QA_PROMPT = PromptTemplate(
    template=SYSTEM_TEMPLATE + "\n\n**Context:**\n{context}\n\n**Question:** {query}\n\n**Answer:**",
    input_variables=["context", "query"]
)

@st.cache_resource(show_spinner=False)
def get_llm_chain():
    return LLMChain(llm=llm, prompt=QA_PROMPT)

llm_chain = get_llm_chain()

# ============================= Streamlit UI & CSS =============================
st.set_page_config(page_title="LegalBot", page_icon="⚖️", layout="wide")
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Poppins', sans-serif !important; }
body, .main { color: #e9eef6; }

.official-header {
    background: linear-gradient(90deg,#1e3c72 0%,#2a5298 100%);
    color: #fff;
    padding:2.2rem;
    border-radius:0 0 20px 20px;
    text-align:center;
    box-shadow:0 4px 18px #0002;
    margin-bottom:0.8rem;
}

.login-box, .register-box, .profile-box {
    background: rgba(27, 35, 55, 0.85);
    color:#e9e9ef;
    padding:2.1rem;
    border-radius: 15px;
    box-shadow:0 6px 26px rgba(0,0,0,0.6);
    max-width:480px;
    margin:2rem auto;
    backdrop-filter: blur(6px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.login-box:hover, .register-box:hover, .profile-box:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
}

.stButton>button {
    background: linear-gradient(90deg,#FFD700 0%,#FFAA00 100%);
    color: #000 !important;
    border:none;
    border-radius:7px;
    padding:0.5rem 0.9rem;
    font-weight:600;
}
.stButton>button:hover { transform: scale(1.03); }

.profile-pic {
    border-radius:80px;
    border:2.5px solid #FFD700;
    margin-bottom:12px;
    width:120px;
    height:120px;
    object-fit:cover;
}

.chat-header {
    background: rgba(255, 215, 0, 0.1);
    color: #FFD700;
    text-align:center;
    padding:1.2rem;
    border-radius:0 0 25px 25px;
    font-size:1.6rem;
    box-shadow:0 4px 20px rgba(0,0,0,0.3);
    margin-bottom:1rem;
}

.stChatMessage {
    border-radius: 15px;
    padding: 1rem;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(6px);
    color: black !important;  /* <-- text color set to black */
}

.stChatMessage.user {
    background: rgba(255, 215, 0, 0.15);
    border: 1px solid rgba(255, 215, 0, 0.3);
    color: black !important;  /* <-- ensure user text is black */
}

.stChatMessage.assistant {
    background: rgba(255, 255, 255, 0.8); /* lighter for contrast */
    border: 1px solid rgba(0, 0, 0, 0.2);
    color: black !important;  /* <-- assistant text black */
}

/* ---------- Dynamic Backgrounds ---------- */
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1559526324-593bc073d938?auto=format&fit=crop&w=1950&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    transition: background-image 0.5s ease-in-out;
}

/* You can change login/register/chat images dynamically using body class if needed */
[data-testid="stAppViewContainer"].login { background-image: url('https://images.unsplash.com/photo-1591696205602-c5c518fb38f0?auto=format&fit=crop&w=1950&q=80'); }
[data-testid="stAppViewContainer"].register { background-image: url('https://images.unsplash.com/photo-1581091215360-6e2b2a67e27b?auto=format&fit=crop&w=1950&q=80'); }
[data-testid="stAppViewContainer"].chat { background-image: url('https://images.unsplash.com/photo-1522071820081-009f0129c71c?auto=format&fit=crop&w=1950&q=80'); }

</style>
""", unsafe_allow_html=True)


# ============================= Session State =============================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "email" not in st.session_state:
    st.session_state.email = ""
if "page" not in st.session_state:
    st.session_state.page = "Chat"
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None

# ============================= Dynamic Background =============================
def get_base64_image(image_file):
    """Read a local image file and return base64 string"""
    if not os.path.exists(image_file):
        return None
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_dynamic_background():
    """
    Sets Streamlit background dynamically based on current page/auth mode
    using base64 encoded images.
    """
    # Select image based on page/auth status
    if not st.session_state.authenticated:
        if st.session_state.auth_mode.lower() == "register":
            bg_file = "static/login_bg.jpg"
        else:
            bg_file = "static/login_bg.jpg"
    else:
        bg_file = "static/law_bg.jpg"

    encoded_image = get_base64_image(bg_file)
    if not encoded_image:
        return  # If file not found, skip

    # Inject CSS with base64 image
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            transition: background 0.5s ease-in-out;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call function to set background
set_dynamic_background()

# ============================= Logout Function =============================
def logout():
    st.session_state.authenticated = False
    st.session_state.email = ""
    st.session_state.current_conversation = None
    st.session_state.messages = []
    st.session_state.page = "Chat"
    st.session_state.auth_mode = "Login"
    st.success("Logged out successfully!")
    st.rerun()



# ============================= Header =============================
st.markdown("""<div class="official-header">
<h1>⚖️ LegalBot - AI Legal Reference & Case Retrieval System</h1>
<span style="font-size:1rem;">Your trusted AI companion for law queries</span>
</div>""", unsafe_allow_html=True)

# ============================= Auth Pages =============================
def show_login():
    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.subheader("Login to Continue")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if not email or not password:
            st.warning("Please enter both fields.")
        elif login_user(email, password):
            st.session_state.authenticated = True
            st.session_state.email = email
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.success("✅ Logged in!")
            st.rerun()
        else:
            st.error("Invalid email or password.")
    if st.button("Create New Account"):
        st.session_state.auth_mode = "Register"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

def show_register():
    st.markdown('<div class="register-box">', unsafe_allow_html=True)
    st.subheader("Create New Account")
    first_name = st.text_input("First Name", key="reg_first")
    last_name = st.text_input("Last Name", key="reg_last")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_pass")
    profile_pic = st.file_uploader("Upload Profile Photo (Optional)", type=["jpg","jpeg","png"], key="reg_pic")
    if st.button("Register"):
        if not first_name or not last_name or not email or not password:
            st.warning("All fields except photo are mandatory.")
        elif not is_valid_email(email):
            st.warning("Please enter a valid email address.")
        elif not is_valid_password(password):
            st.warning("Password must be at least 6 characters long and contain letters and numbers.")
        else:
            pic_path = save_profile_photo(profile_pic, email) if profile_pic else None
            if register_user(email, password, first_name, last_name, pic_path):
                st.success("Account created! Please log in.")
                st.session_state.auth_mode = "Login"
                st.rerun()
            else:
                st.error("Email already registered.")
    if st.button("Back to Login"):
        st.session_state.auth_mode = "Login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ============================= Profile Page =============================
def show_profile():
    user = fetch_user(st.session_state.email)
    if not user:
        st.error("User not found.")
        return
    st.markdown('<div class="profile-box">', unsafe_allow_html=True)
    if user and user[3] and os.path.exists(user[3]):
        st.image(user[3], width=120)
        if st.button("Delete Profile Photo"):
            delete_profile_photo(user[0])
            st.success("Profile photo deleted!")
            st.rerun()
    st.subheader(f"{user[1]} {user[2]}")
    st.caption(user[0])
    st.divider()
    st.markdown("#### Edit Profile")
    first_name = st.text_input("First Name", value=user[1], key="prof_first")
    last_name = st.text_input("Last Name", value=user[2], key="prof_last")
    uploaded_pic = st.file_uploader("Update Profile Photo", type=["jpg","jpeg","png"], key="prof_pic")
    if st.button("Save Changes"):
        pic_path = user[3]
        if uploaded_pic:
            pic_path = save_profile_photo(uploaded_pic, user[0])
        update_profile(user[0], first_name, last_name, pic_path)
        st.success("Profile updated!")
        st.rerun()
    st.divider()
    st.markdown("#### Change Email")
    new_email = st.text_input("New Email", value=user[0], key="prof_new_email")
    if st.button("Update Email"):
        if new_email == user[0]:
            st.info("No changes detected.")
        else:
            result = update_email(user[0], new_email)
            if result == "success":
                st.success("Email updated successfully! Please log in again.")
                logout()
            else:
                st.warning(result)
    st.divider()
    st.markdown("#### Change Password")
    old_pass = st.text_input("Old Password", type="password", key="prof_old_pass")
    new_pass = st.text_input("New Password", type="password", key="prof_new_pass")
    confirm_pass = st.text_input("Confirm New Password", type="password", key="prof_confirm_pass")
    if st.button("Update Password"):
        if not old_pass or not new_pass or not confirm_pass:
            st.warning("Fill all password fields.")
        elif new_pass != confirm_pass:
            st.warning("New passwords do not match.")
        elif not is_valid_password(new_pass):
            st.warning("Password must be at least 6 chars long, with letters & numbers.")
        elif update_password(user[0], old_pass, new_pass):
            st.success("Password updated successfully!")
        else:
            st.error("Old password incorrect.")
    st.markdown('</div>', unsafe_allow_html=True)

# ============================= Chat Page =============================

LEGAL_KEYWORDS = [
    "law", "legal", "statute", "case", "regulation", "rights", "contract",
    "liability", "penalty", "court", "claim", "agreement", "compliance"
]

def is_legal_question(query: str) -> bool:
    """Return True if the query seems like a legal question."""
    query_lower = query.lower()
    return any(k in query_lower for k in LEGAL_KEYWORDS)



def handle_chat():
    user = fetch_user(st.session_state.email)
    if not user:
        st.error("User not found. Please log in.")
        return

    st.markdown(f"<div class='chat-header'>Welcome, {user[1]} 👋 Ask your question below ⚖️</div>", unsafe_allow_html=True)

    # Load previous messages if conversation exists
    if st.session_state.current_conversation:
        st.session_state.messages = get_messages(st.session_state.current_conversation)
    else:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("Type your legal question or say hello...")

    if not query:
        return

    # Create new conversation if needed
    if not st.session_state.current_conversation:
        title = (query[:30] + "...") if len(query) > 30 else query
        st.session_state.current_conversation = create_conversation(user[0], title)

    # Save user message
    add_message(st.session_state.current_conversation, "user", query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ====================== Casual or Legal Intent Classification ======================
    intent_prompt = PromptTemplate(
        input_variables=["query"],
        template=(
            "Classify the following user message as either 'legal' or 'casual'. "
            "Respond with one word only: legal or casual.\n\n"
            "Message: {query}"
        ),
    )
    intent_chain = LLMChain(llm=llm, prompt=intent_prompt)
    try:
        intent = intent_chain.run({"query": query}).strip().lower()
    except Exception:
        intent = "legal"

    # ====================== If Casual → Skip Retrieval, Just Chat ======================
    if "casual" in intent:
        chat_prompt = (
            "You are LegalBot ⚖️, an AI legal research assistant. "
            "If the user greets you, asks who you are, or makes small talk, respond warmly and naturally. "
            "Briefly introduce yourself and what you can do (help with laws, legal summaries, etc.). "
            "Keep it friendly but professional.\n\n"
            f"User: {query}"
        )
        response = llm.predict(chat_prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        add_message(st.session_state.current_conversation, "assistant", response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        return

    # ====================== Retrieve Relevant Legal Context ======================
    context_combined = ""
    if retriever:
        try:
            context_docs = retriever.get_relevant_documents(query)
            if context_docs:
                context_combined = "\n\n".join(
                    [f"{i+1}. {doc.page_content}" for i, doc in enumerate(context_docs)]
                )
        except Exception as e:
            st.warning(f"Error fetching legal documents: {e}")

    history_text = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state.messages])
    full_context = history_text
    if context_combined:
        full_context += "\n\n**Relevant Legal Documents:**\n" + context_combined

    adjusted_query = query
    if any(keyword in query.lower() for keyword in ["difference", "compare"]):
        adjusted_query += "\nPlease clearly outline key differences using bullet points or headings."

    add_disclaimer = is_legal_question(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm_chain.run({"context": full_context, "query": adjusted_query})
            except Exception as e:
                st.error(f"LLM error: {e}")
                response = "Sorry, I couldn't generate a response right now."

            if add_disclaimer:
                response += (
                    "\n\n⚖️ **Legal Disclaimer:** This information is for research purposes only. "
                    "I am not a licensed attorney, and this does not constitute legal advice. "
                    "For matters affecting your legal rights, please consult a qualified attorney."
                )

            st.markdown(response)
            add_message(st.session_state.current_conversation, "assistant", response)
            st.session_state.messages.append({"role": "assistant", "content": response})



# ============================= Sidebar & Routing =============================
if not st.session_state.authenticated:
    if st.session_state.auth_mode == "Register":
        show_register()
    else:
        show_login()
else:
    user = fetch_user(st.session_state.email)
    with st.sidebar:
        if user and user[3] and os.path.exists(user[3]):
            st.image(user[3], width=80)
        if user:
            st.markdown(f"#### {user[1]} {user[2]}")
            st.markdown(f"`{user[0]}`")
        if st.button("Chat", use_container_width=True):
            st.session_state.page = "Chat"
        if st.button("Profile", use_container_width=True):
            st.session_state.page = "Profile"
        if st.button("🆕 New Chat", use_container_width=True):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.session_state.page = "Chat"
            st.success("Started a new chat!")
            st.rerun()
        st.divider()
        st.sidebar.markdown("### 🗂 Your Conversations")
        if user:
            conversations = get_conversations(user[0])
            for conv in conversations:
                col1, col2 = st.sidebar.columns([4, 1])
                if col1.button(conv[1], key=f"conv_{conv[0]}"):
                    st.session_state.current_conversation = conv[0]
                    st.session_state.messages = get_messages(conv[0])
                if col2.button("❌", key=f"del_{conv[0]}"):
                    delete_conversation(conv[0])
                    st.rerun()
        st.divider()
        if st.button("Logout"):
            logout()

    if st.session_state.page == "Profile":
        show_profile()
    else:
        handle_chat()