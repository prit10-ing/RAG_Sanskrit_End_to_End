"""
Futuristic Sanskrit AI Assistant
Completely Different Layout Design
Run:
    streamlit run app_ui.py
"""

import os
import streamlit as st
from src.pipeline.rag_pipeline import RAGPipeline
from src.config import VECTOR_DB_DIR

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Sanskrit AI",
    page_icon="🪔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

/* Main App */
.stApp {
    background: #0d0d0d;
    color: white;
}

/* Remove Streamlit Header */
header {
    visibility: hidden;
}

/* Main Container */
.main-container {
    max-width: 1100px;
    margin: auto;
    padding-top: 20px;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, #ff6b00, #ffb347);
    padding: 40px;
    border-radius: 25px;
    text-align: center;
    margin-bottom: 25px;
    color: black;
    box-shadow: 0px 8px 25px rgba(255,140,0,0.3);
}

.hero-title {
    font-size: 50px;
    font-weight: bold;
}

.hero-subtitle {
    font-size: 18px;
    margin-top: 10px;
}

/* Feature Cards */
.card {
    background: #1a1a1a;
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #2e2e2e;
    transition: 0.3s;
    height: 180px;
}

.card:hover {
    transform: translateY(-5px);
    border: 1px solid #ff8800;
    box-shadow: 0px 6px 20px rgba(255,136,0,0.25);
}

.card-title {
    color: #ffb347;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Chat Area */
.chat-container {
    background: #151515;
    border-radius: 20px;
    padding: 20px;
    margin-top: 25px;
    border: 1px solid #262626;
}

/* Buttons */
.stButton button {
    width: 100%;
    border-radius: 15px;
    background: linear-gradient(135deg, #ff6b00, #ffb347);
    color: black;
    border: none;
    font-weight: bold;
    padding: 12px;
    transition: 0.3s;
}

.stButton button:hover {
    transform: scale(1.02);
}

/* Chat Input */
.stChatInput textarea {
    background: #1f1f1f !important;
    color: white !important;
    border: 1px solid #ff8800 !important;
    border-radius: 15px !important;
}

/* Messages */
[data-testid="stChatMessage"] {
    background: #1b1b1b;
    border-radius: 15px;
    padding: 12px;
    border-left: 4px solid #ff8800;
}

/* Status Box */
.status-box {
    padding: 15px;
    border-radius: 15px;
    background: #1a1a1a;
    border: 1px solid #333;
    margin-bottom: 15px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #ff8800;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------

def vector_db_exists():
    return os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


pipeline = load_pipeline()


def get_answer(question):
    try:
        return pipeline.query(question)
    except Exception as e:
        return f"❌ Error: {e}"


def ingest_documents():
    with st.spinner("📥 Processing documents..."):
        pipeline.ingest()

    st.success("✅ Documents processed successfully!")


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------

st.markdown("""
<div class="main-container">

<div class="hero-box">
    <div class="hero-title">🪔 Sanskrit AI Assistant</div>
    <div class="hero-subtitle">
        Smart RAG-Based Sanskrit Question Answering System
    </div>
</div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# FEATURE CARDS
# ---------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <div class="card-title">📚 Document Search</div>
        Ask questions directly from Sanskrit PDFs and documents.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🌐 Multi Language</div>
        Ask questions in English, Hindi, or Sanskrit.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <div class="card-title">⚡ AI Powered</div>
        Uses RAG Pipeline + Vector Database + LLM.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------
# CONTROL PANEL
# ---------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

left, middle, right = st.columns([2,2,2])

with left:

    if vector_db_exists():
        st.success("✅ Vector DB Ready")
    else:
        st.error("❌ Vector DB Missing")

with middle:

    if st.button("🔄 Ingest Documents"):
        ingest_documents()

with right:

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------
# SAMPLE QUESTIONS
# ---------------------------------------------------

st.markdown("## 💡 Quick Questions")

q1, q2, q3, q4 = st.columns(4)

sample_questions = [
    "Who was Kalidasa?",
    "What is the moral of story?",
    "कालीदासः कः आसीत्?",
    "देवः कब मदद करता है?"
]

columns = [q1, q2, q3, q4]

for col, q in zip(columns, sample_questions):

    with col:
        if st.button(q):

            st.session_state.messages.append({
                "role": "user",
                "content": q
            })

            answer = get_answer(q)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

# ---------------------------------------------------
# CHAT SECTION
# ---------------------------------------------------

st.markdown("""
<div class="chat-container">
""", unsafe_allow_html=True)

st.markdown("## 💬 AI Conversation")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        # Copyable Answer
        if msg["role"] == "assistant":
            st.code(msg["content"], language="text")

question = st.chat_input("Ask anything about Sanskrit documents...")

if question:

    # User Message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Assistant Message
    with st.chat_message("assistant"):

        with st.spinner("🔍 Thinking..."):

            if not vector_db_exists():
                answer = "⚠️ Please ingest documents first."
            else:
                answer = get_answer(question)

        st.markdown(answer)

        # Copy Option
        st.code(answer, language="text")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

st.markdown("</div>", unsafe_allow_html=True)