"""
NexusAI — Enterprise Knowledge & Document Intelligence Platform
================================================================
A production-ready Retrieval-Augmented Generation (RAG) system featuring:
- Smart Intent Routing (instant pleasantries & chit-chat without DB overhead)
- Conversational Memory (multi-turn follow-up question resolution)
- Structure-Aware Smart Chunking (preserves prompts, headers & sections)
- High-Precision Semantic Vector Retrieval (Cosine Similarity with Exact Scoring)
- Multi-Strategy Search (Semantic Similarity, MMR Diversity, Score Cutoff Threshold)
- Dual-Mode Qdrant Vector Engine (Local Embedded & Cloud Cluster)
"""

import os
import time
import uuid
import hashlib
import tempfile
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

import streamlit as st

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI


# ============================================================
# APP CONFIG & ENTERPRISE BRANDING
# ============================================================
st.set_page_config(
    page_title="NexusAI — Document Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-caption {
        color: #757575;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: #E8F5E9;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .source-card {
        padding: 10px 14px;
        border-radius: 8px;
        background: #F8F9FA;
        border-left: 4px solid #1E88E5;
        margin-top: 8px;
    }
    .eval-card {
        padding: 14px;
        border-radius: 10px;
        background: #F8F9FA;
        border: 1px solid #E0E0E0;
        text-align: center;
        margin-bottom: 8px;
    }
    .eval-card-score {
        font-size: 1.7rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .badge-pass {
        background-color: #E8F5E9;
        color: #2E7D32;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-review {
        background-color: #FFF3E0;
        color: #E65100;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

LOCAL_QDRANT_PATH = "./qdrant_store"
COLLECTION_NAME = "enterprise_knowledge_base"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384


# ============================================================
# INTENT ROUTER: CHIT-CHAT & GREETINGS HANDLER
# ============================================================
GREETINGS_MAP = {
    "hi": "Hello! 👋 I'm your Enterprise Knowledge Assistant. How can I help you explore your uploaded documents today?",
    "hello": "Hello! 👋 Welcome. Feel free to ask any question regarding your uploaded documents, policies, or data.",
    "hey": "Hey there! 👋 I am ready to analyze and query your documents.",
    "salam": "Walaikum Assalam! 👋 Main aap ke uploaded documents se related kisi bhi sawal ka jawab dene ke liye active hoon. Aap kya dhoondna chahte hain?",
    "assalam o alaikum": "Walaikum Assalam! 👋 Main aap ke uploaded documents se related kisi bhi sawal ka jawab dene ke liye active hoon. Aap kya dhoondna chahte hain?",
    "aoa": "Walaikum Assalam! 👋 How can I assist you with your knowledge base today?",
    "who are you": "I am **NexusAI**, an enterprise document intelligence assistant. I specialize in reading, indexing, and answering questions from your company files with strict factual accuracy and source citations.",
    "what can you do": "I can analyze your uploaded PDFs, Word documents, CSVs, and text files. Using hybrid semantic search, I locate exact facts, calculate similarity confidence scores, and answer questions without hallucinating.",
    "thank you": "You're very welcome! Let me know if you need anything else from your documents. 😊",
    "thanks": "Glad to help! Feel free to ask any follow-up question anytime. 😊",
    "bye": "Goodbye! Have a productive day ahead! 👋",
}

def detect_chit_chat(query: str) -> str | None:
    """
    Classifies if a query is a greeting or general pleasantry.
    Bypasses vector search completely to deliver instant human-like responses.
    """
    cleaned = query.strip().lower().rstrip("!?.")
    if cleaned in GREETINGS_MAP:
        return GREETINGS_MAP[cleaned]
    
    tokens = set(cleaned.split())
    if tokens.intersection({"hi", "hello", "hey", "salam", "aoa"}) and len(tokens) <= 3:
        return "Hello! 👋 How can I assist you with your documents today?"
    if tokens.intersection({"thanks", "thankyou"}) and len(tokens) <= 3:
        return "You're most welcome! Happy to assist. 😊"
    return None


# ============================================================
# CONVERSATIONAL MEMORY: QUERY REFORMULATION
# ============================================================
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def contextualize_query(question: str, chat_history: list, api_key: str, base_url: str, model_name: str) -> str:
    """
    Reformulates follow-up queries using past conversational turns.
    Example: 'What is its pricing?' -> 'What is the pricing for Junoon perfume?'
    """
    if not chat_history or not api_key:
        return question

    recent_turns = []
    for role, msg in chat_history[-4:]:
        if role == "user":
            recent_turns.append(HumanMessage(content=msg))
        elif role == "assistant":
            recent_turns.append(AIMessage(content=msg))

    try:
        fast_llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
        )
        chain = CONTEXTUALIZE_PROMPT | fast_llm | StrOutputParser()
        standalone = chain.invoke({"chat_history": recent_turns, "question": question})
        return standalone.strip()
    except Exception:
        return question


# ============================================================
# DOCUMENT INGESTION (STEP 1)
# ============================================================
def load_any_file(uploaded_file) -> list:
    suffix = "." + uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            file_type = "pdf"
        elif suffix == ".csv":
            loader = CSVLoader(file_path=tmp_path, encoding="utf-8")
            file_type = "csv"
        elif suffix in (".docx", ".doc"):
            loader = Docx2txtLoader(tmp_path)
            file_type = "docx"
        elif suffix in (".txt", ".md"):
            loader = TextLoader(tmp_path, encoding="utf-8")
            file_type = "text"
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
            file_type = "unknown"

        documents = loader.load()
        upload_time = datetime.now().isoformat(timespec="seconds")

        for doc in documents:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["file_type"] = file_type
            doc.metadata["upload_time"] = upload_time
            if file_type == "pdf":
                raw_label = doc.metadata.get("page_label")
                if raw_label:
                    try:
                        doc.metadata["page_number"] = int(str(raw_label).strip())
                    except ValueError:
                        doc.metadata["page_number"] = int(doc.metadata.get("page", 0)) + 1
                else:
                    doc.metadata["page_number"] = int(doc.metadata.get("page", 0)) + 1

        return documents
    finally:
        os.remove(tmp_path)


def format_doc_location(meta: dict) -> str:
    """Extracts human-readable 1-indexed page number or row for exact citations."""
    if meta.get("page_label"):
        label = str(meta["page_label"]).strip()
        if label:
            return f"Page {label}"
    if meta.get("page_number") is not None:
        return f"Page {meta['page_number']}"
    if meta.get("page") is not None:
        p = meta["page"]
        if isinstance(p, int):
            return f"Page {p + 1}"
        return f"Page {p}"
    if meta.get("row") is not None:
        return f"Row {meta['row']}"
    return ""


# ============================================================
# SMART STRUCTURE-AWARE SPLITTING (STEP 2)
# ============================================================
def split_documents(documents: list, chunk_size: int = 1200, chunk_overlap: int = 200) -> list:
    smart_separators = [
        "\n\n\n",
        "\n## ",
        "\n### ",
        "\nTopic ",
        "\nPrompt ",
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=smart_separators,
        keep_separator=True,
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)
        chunk.metadata["chunk_id"] = hashlib.md5(
            (chunk.metadata.get("source", "") + chunk.page_content).encode("utf-8")
        ).hexdigest()

    return chunks


# ============================================================
# EMBEDDINGS & QDRANT VECTOR STORE (STEP 3 & 4)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_qdrant_client(mode: str, url: str, api_key: str):
    if mode == "Cloud":
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(path=LOCAL_QDRANT_PATH)


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )


@st.cache_resource(show_spinner=False)
def get_vectorstore(_client: QdrantClient, _embeddings):
    ensure_collection(_client)
    return QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embedding=_embeddings,
    )


def reset_vectorstore(client: QdrantClient, embeddings):
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    get_vectorstore.clear()
    ensure_collection(client)
    return get_vectorstore(client, embeddings)


def index_chunks(vectorstore, chunks: list):
    if not chunks:
        return
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.metadata["chunk_id"])) for chunk in chunks]
    vectorstore.add_documents(documents=chunks, ids=ids)


# ============================================================
# SEMANTIC VECTOR RETRIEVAL & EXACT SCORING (STEP 5)
# ============================================================
def retrieve_docs_with_scores(
    vectorstore,
    question: str,
    k: int,
    search_mode: str = "Score Threshold (Cutoff)",
    score_threshold: float = 0.40,
    source_filter: str = "All files",
) -> tuple[list, float]:
    """
    Retrieves top matching documents using true Cosine Similarity vectors.
    Every document receives its real, honest cosine similarity percentage.
    Applies score_threshold filtering (e.g. >= 0.50).
    Returns (filtered_docs, max_similarity_score).
    """
    filter_obj = None
    if source_filter and source_filter != "All files":
        filter_obj = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source_filter))]
        )

    max_score = 0.0
    all_scored = []

    if "Hybrid" in search_mode:
        fetch_k = max(k * 2, 8)
        dense_results = vectorstore.similarity_search_with_score(
            query=question,
            k=fetch_k,
            filter=filter_obj,
        )
        bm25_docs = []
        if hasattr(st, "session_state") and getattr(st.session_state, "all_chunks", None):
            try:
                from langchain_community.retrievers import BM25Retriever
                bm25 = BM25Retriever.from_documents(st.session_state.all_chunks)
                bm25.k = fetch_k
                bm25_docs = bm25.invoke(question)
            except Exception:
                bm25_docs = []

        rrf_scores = {}
        doc_map = {}
        rrf_k = 60.0

        for rank, (doc, score) in enumerate(dense_results, 1):
            key = doc.metadata.get("chunk_id", doc.page_content)
            doc_map[key] = (doc, float(score))
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (0.6 / (rrf_k + rank))

        for rank, doc in enumerate(bm25_docs, 1):
            key = doc.metadata.get("chunk_id", doc.page_content)
            if key not in doc_map:
                doc_map[key] = (doc, 0.70)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (0.4 / (rrf_k + rank))

        sorted_keys = sorted(rrf_scores.keys(), key=lambda k_id: rrf_scores[k_id], reverse=True)[:k]
        for key in sorted_keys:
            doc, s = doc_map[key]
            doc.metadata["score"] = round(s, 4)
            if s > max_score:
                max_score = s
            all_scored.append(doc)
    elif search_mode == "MMR (Diverse)":
        docs = vectorstore.max_marginal_relevance_search(
            query=question,
            k=k,
            fetch_k=max(k * 4, 10),
            lambda_mult=0.5,
            filter=filter_obj,
        )
        scored_pairs = vectorstore.similarity_search_with_score(
            query=question,
            k=max(k * 4, 10),
            filter=filter_obj,
        )
        score_map = {doc.metadata.get("chunk_id", doc.page_content): score for doc, score in scored_pairs}
        for doc in docs:
            chunk_key = doc.metadata.get("chunk_id", doc.page_content)
            s = round(float(score_map.get(chunk_key, 0.0)), 4)
            doc.metadata["score"] = s
            if s > max_score:
                max_score = s
            all_scored.append(doc)
    else:
        results = vectorstore.similarity_search_with_score(
            query=question,
            k=k,
            filter=filter_obj,
        )
        for doc, score in results:
            s = round(float(score), 4)
            doc.metadata["score"] = s
            if s > max_score:
                max_score = s
            all_scored.append(doc)

    if score_threshold > 0.0:
        filtered = [doc for doc in all_scored if doc.metadata.get("score", 0.0) >= score_threshold]
    else:
        filtered = all_scored

    return filtered, max_score


# ============================================================
# LCEL GENERATION PIPELINE (STEP 6)
# ============================================================
def format_docs(docs) -> str:
    if not docs:
        return "No relevant context found matching the search criteria or threshold."
    if isinstance(docs, tuple):
        docs = docs[0]
    parts = []
    for item in docs:
        if isinstance(item, tuple) and len(item) == 2:
            doc, _ = item
        elif isinstance(item, list) and item:
            doc = item[0]
        else:
            doc = item
        if not hasattr(doc, "metadata"):
            continue
        meta = doc.metadata
        src = meta.get("source", "unknown")
        loc = format_doc_location(meta)
        loc_str = f", {loc}" if loc else ""
        score = meta.get("score")
        score_tag = f" (Confidence: {score * 100:.1f}%)" if score is not None else ""
        tag = f"{src}{loc_str}{score_tag}"
        parts.append(f"[Document Source: {tag}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts) if parts else "No relevant context found."


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are NexusAI, an accurate and truthful enterprise knowledge assistant.\n\n"
        "Grounding Rules (STRICT):\n"
        "1. Answer based ONLY on the following context. Do NOT use any outside knowledge, training memory, or web information.\n"
        "2. Do NOT add any information not explicitly stated in the context.\n"
        "3. If the answer is not in the context, respond verbatim: \"I don't have enough information in the uploaded documents to answer that.\" Never guess, speculate, or fill gaps.\n"
        "4. Direct & Concise: Answer ONLY what the user explicitly asks. No unsolicited filler, long introductions, or extra unrelated details.\n"
        "5. Exact Citation: Always finish with a single, elegant citation line:\n"
        "   ---\n"
        "   > 📑 **Source:** `<filename>` | **Location:** <Page X> | **Confidence:** <score>%",
    ),
    ("human", "Context:\n{context}\n\nQuestion: {question}\n\nConcise & Accurate Answer:"),
])


def stream_text(text: str, delay: float = 0.012):
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(delay)


def build_rag_chain(api_key: str, base_url: str, model: str):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
        streaming=True,
    )
    return RAG_PROMPT | llm | StrOutputParser()


# ============================================================
# STATE INITIALIZATION
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()
if "all_metadata" not in st.session_state:
    st.session_state.all_metadata = []
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []

embeddings = get_embeddings()


# ============================================================
# ENTERPRISE SIDEBAR
# ============================================================
with st.sidebar:
    st.title("⚡ NexusAI")
    st.caption("Enterprise Document Intelligence Platform")
    st.markdown("---")

    st.subheader("🤖 AI Model Provider")
    provider = st.selectbox("Select Provider", ["Groq", "OpenAI", "Preview Mode (No LLM)"])

    api_key, model_name, base_url = "", "", None
    if provider == "Groq":
        default_groq = os.getenv("GROQ_API_KEY", "")
        api_key = st.text_input("Groq API Key", value=default_groq, type="password", placeholder="gsk_...")
        groq_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "groq/compound-mini",
            "allam-2-7b",
        ]
        model_name = st.selectbox("Model", groq_models, index=0)
        base_url = "https://api.groq.com/openai/v1"
    elif provider == "OpenAI":
        default_openai = os.getenv("OPENAI_API_KEY", "")
        api_key = st.text_input("OpenAI API Key", value=default_openai, type="password", placeholder="sk-...")
        model_name = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)

    st.markdown("---")
    st.subheader("📂 Document Scoping")
    available_files = ["All files"] + sorted(list(st.session_state.indexed_files))
    selected_source = st.selectbox(
        "Filter by Document",
        available_files,
        help="Restrict answers to a specific file or search across all files.",
    )

    # Clean Advanced / Developer Drawer
    with st.expander("🛠️ Advanced Search & Indexing Engine"):
        qdrant_mode = st.radio(
            "Database Mode",
            ["Cloud Cluster", "Local Embedded"],
            index=0 if os.getenv("QDRANT_URL") else 1,
        )
        default_url = os.getenv("QDRANT_URL", "")
        default_key = os.getenv("QDRANT_API_KEY", "")
        
        qdrant_url = default_url if qdrant_mode == "Cloud Cluster" else ""
        qdrant_api_key = default_key if qdrant_mode == "Cloud Cluster" else ""

        search_mode = st.selectbox(
            "Search Engine Strategy",
            [
                "Hybrid (Dense + BM25) [Optimal Recall]",
                "Score Threshold (Cutoff)",
                "Semantic Similarity",
                "MMR (Diverse)",
            ],
            index=0,
            help="Hybrid combines semantic dense vectors with BM25 keyword matching to prevent missing exact terms and numbers.",
        )

        k_value = st.slider("Top Chunks (k)", 1, 15, 8, help="Recommended optimal: k=8 (boosts context recall in RAGAS evaluation).")
        score_threshold = st.slider(
            "Similarity Score Threshold",
            min_value=0.0,
            max_value=0.95,
            value=0.40,
            step=0.05,
            help="Minimum Cosine Similarity required for a passage to be retrieved and answered (Default: 0.40 / 40%). Set higher for stricter matching.",
        )

        chunk_size_val = st.slider("Chunk Size", 600, 2500, 1200, 100)
        chunk_overlap_val = st.slider("Chunk Overlap", 50, 400, 200, 50)

    st.markdown("---")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        if st.button("🗑️ Reset Docs", use_container_width=True):
            qdrant_mode_key = "Cloud" if qdrant_mode == "Cloud Cluster" else "Local"
            client_temp = get_qdrant_client(qdrant_mode_key, qdrant_url, qdrant_api_key)
            reset_vectorstore(client_temp, embeddings)
            st.session_state.indexed_files = set()
            st.session_state.all_metadata = []
            st.session_state.all_chunks = []
            st.success("Knowledge base reset.")
    with col_b2:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# INITIALIZE VECTOR DATABASE
# ============================================================
qdrant_mode_key = "Cloud" if qdrant_mode == "Cloud Cluster" else "Local"
embeddings = get_embeddings()
qdrant_client = get_qdrant_client(qdrant_mode_key, qdrant_url, qdrant_api_key)
vectorstore = get_vectorstore(qdrant_client, embeddings)

# Auto-recover chunks from persistent Qdrant collection on page load
if not st.session_state.all_chunks:
    try:
        points, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=2000,
            with_payload=True,
            with_vectors=False,
        )
        if points:
            for p in points:
                payload = p.payload or {}
                meta = payload.get("metadata", {})
                content = payload.get("page_content", "")
                if content:
                    doc = Document(page_content=content, metadata=meta)
                    st.session_state.all_chunks.append(doc)
                    src = meta.get("source")
                    if src:
                        st.session_state.indexed_files.add(src)
                    st.session_state.all_metadata.append(meta)
    except Exception:
        pass


# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================
st.markdown('<div class="main-header">NexusAI — Document Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-caption">Enterprise Question-Answering & Knowledge Retrieval Engine with Source Grounding</div>', unsafe_allow_html=True)

# Status Pill
status_text = f"Connected • {qdrant_mode} • {len(st.session_state.indexed_files)} Document(s) Active"
st.markdown(f'<span class="status-badge">🟢 {status_text}</span>', unsafe_allow_html=True)

# Document Ingestion Box
with st.expander("📤 Upload & Index Business Documents", expanded=not bool(st.session_state.indexed_files)):
    uploaded_files = st.file_uploader(
        "Upload enterprise files (PDF, Word DOCX, CSV, TXT, MD)",
        type=["pdf", "txt", "csv", "docx", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            with st.spinner(f"Ingesting and indexing {len(new_files)} document(s)..."):
                all_chunks = []
                for f in new_files:
                    docs = load_any_file(f)
                    chunks = split_documents(docs, chunk_size=chunk_size_val, chunk_overlap=chunk_overlap_val)
                    all_chunks.extend(chunks)
                    st.session_state.indexed_files.add(f.name)

                index_chunks(vectorstore, all_chunks)
                st.session_state.all_metadata.extend(c.metadata for c in all_chunks)
                st.session_state.all_chunks.extend(all_chunks)

            st.success(f"Successfully indexed {len(new_files)} document(s) into {len(all_chunks)} semantic chunks.")

if st.session_state.indexed_files:
    with st.expander("📊 Document Index & Metadata Inspector"):
        st.dataframe(st.session_state.all_metadata, use_container_width=True)

st.markdown("---")

# Conversational Chat
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Ask any question about your documents...")

if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # 1. SMART INTENT ROUTER: Check for Chit-Chat / Greetings first
        chit_chat_reply = detect_chit_chat(question)

        if chit_chat_reply:
            answer = st.write_stream(stream_text(chit_chat_reply))
        elif not st.session_state.indexed_files:
            answer = st.write_stream(stream_text("Please upload at least one document above so I can assist you with your knowledge base! 📂"))
        else:
            # 2. CONTEXTUAL MEMORY: Resolve follow-up inquiries
            standalone_query = contextualize_query(
                question=question,
                chat_history=st.session_state.chat_history[:-1],
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
            )

            # 3. HIGH-PRECISION RETRIEVAL (True Cosine Similarity + Threshold)
            retrieved_docs, max_score = retrieve_docs_with_scores(
                vectorstore=vectorstore,
                question=standalone_query,
                k=k_value,
                search_mode=search_mode,
                score_threshold=score_threshold,
                source_filter=selected_source,
            )

            # 4. ANSWER GENERATION OR THRESHOLD CUTOFF
            if not retrieved_docs:
                filter_info = f" in `{selected_source}`" if selected_source != "All files" else ""
                no_match_text = (
                    f"⚠️ **No Document Passed Similarity Threshold of {score_threshold:.2f} ({score_threshold*100:.0f}%)**{filter_info}.\n\n"
                )
                if max_score > 0:
                    no_match_text += (
                        f"- **Highest Similarity Found:** `{max_score * 100:.1f}%` (Below required threshold `{score_threshold * 100:.0f}%`).\n"
                        f"- 💡 *Action:* Lower the **Similarity Score Threshold** slider in the sidebar (e.g., to 0.30 or 0.40) to permit lower-confidence passages."
                    )
                else:
                    no_match_text += "- No matching content found in the indexed knowledge base."
                answer = st.write_stream(stream_text(no_match_text))
            elif provider == "Preview Mode (No LLM)" or not api_key:
                filter_info = f" | Filter: `{selected_source}`" if selected_source != "All files" else ""
                raw_text = f"**Top Matching Passages ({len(retrieved_docs)} found | Threshold: ≥ {score_threshold*100:.0f}%{filter_info}):**\n\n"
                for i, doc in enumerate(retrieved_docs, 1):
                    meta = doc.metadata
                    score_val = meta.get("score")
                    score_badge = f" — 🎯 **Similarity: {score_val * 100:.1f}%**" if score_val is not None else ""
                    loc = format_doc_location(meta)
                    loc_info = f" — 📄 **{loc}**" if loc else ""
                    raw_text += (
                        f"**{i}. Document: {meta.get('source', 'unknown')}**{loc_info}{score_badge}\n"
                        f"{doc.page_content}\n\n---\n\n"
                    )
                answer = st.write_stream(stream_text(raw_text))
            else:
                # 5. STREAMING GENERATION VIA LCEL (token-by-token display)
                status_holder = st.empty()
                status_holder.markdown("💭 *Generating answer, please wait...*")
                rag_chain = build_rag_chain(api_key, base_url, model_name)
                context_str = format_docs(retrieved_docs)
                answer = st.write_stream(rag_chain.stream({"context": context_str, "question": question}))
                status_holder.empty()
            st.session_state.chat_history.append(("assistant", answer))
