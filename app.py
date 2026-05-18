"""
app.py — Streamlit UI  (pure frontend, zero ML/NLP logic)
──────────────────────────────────────────────────────────
All backend work is delegated to chatbot.py.

Run:  streamlit run app.py
"""

import os, time, shutil
import streamlit as st

# ── Page config — must come before any other st.* call ────────────────────
st.set_page_config(
    page_title="PDF Chatbot · BERT",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import backend (no Streamlit inside chatbot.py) ────────────────────────
from chatbot import (
    process_pdf_bytes,
    answer_question,
    cleanup_chroma_dir,
    EMBED_MODEL,
    QA_MODEL,
)


# ══════════════════════════════════════════════════════════════════════════
#  Streamlit cache wrappers
#  (st.cache_resource must live in the UI layer, not in the backend)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def cached_process_pdf(file_bytes: bytes, filename: str):
    """Thin cached wrapper around the backend PDF pipeline."""
    return process_pdf_bytes(file_bytes, filename)


# ══════════════════════════════════════════════════════════════════════════
#  CSS — all styling lives here, not in the backend
# ══════════════════════════════════════════════════════════════════════════

def inject_css() -> None:
    st.markdown("""
<style>
/* ── Layout ── */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3e; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
    border-radius: 14px; padding: 28px 36px; margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(21,101,192,.35);
    display: flex; align-items: center; gap: 18px;
}
.app-header h1 { margin: 0; font-size: 2rem; color: #fff; font-weight: 700; }
.app-header p  { margin: 4px 0 0; color: #90caf9; font-size: .95rem; }

/* ── Chat bubbles ── */
.msg-user {
    background: linear-gradient(135deg, #1565c0, #1976d2);
    color: #fff; border-radius: 18px 18px 4px 18px;
    padding: 14px 18px; margin: 8px 0; max-width: 80%;
    margin-left: auto; box-shadow: 0 2px 8px rgba(21,101,192,.3);
}
.msg-bot {
    background: #1e2535; border: 1px solid #2a3550;
    color: #e8eaf6; border-radius: 18px 18px 18px 4px;
    padding: 14px 18px; margin: 8px 0; max-width: 85%;
    box-shadow: 0 2px 8px rgba(0,0,0,.25);
}
.msg-bot .answer { font-size: 1.05rem; line-height: 1.6; }
.msg-meta { display: flex; gap: 12px; margin-top: 10px; flex-wrap: wrap; }

/* ── Badges ── */
.badge {
    background: #0d1b2a; border: 1px solid #1565c0;
    color: #90caf9; border-radius: 20px;
    padding: 3px 12px; font-size: .78rem; font-weight: 600;
}
.badge.green { border-color: #2e7d32; color: #a5d6a7; }

/* ── Sidebar info cards ── */
.info-card {
    background: #1e2535; border: 1px solid #2a3550;
    border-radius: 10px; padding: 14px 16px; margin-bottom: 12px;
}
.info-card h4 {
    margin: 0 0 8px; color: #90caf9;
    font-size: .85rem; text-transform: uppercase; letter-spacing: .06em;
}
.info-card p { margin: 0; color: #cfd8dc; font-size: .88rem; line-height: 1.5; }

/* ── Widgets ── */
[data-testid="stFileUploader"]    { background: #1e2535 !important; border-radius: 10px !important; }
[data-testid="stChatInput"] textarea {
    background: #1e2535 !important; color: #e8eaf6 !important;
    border: 1px solid #2a3550 !important; border-radius: 12px !important;
}

/* ── Thinking animation ── */
.thinking {
    display: flex; align-items: center; gap: 10px;
    color: #90caf9; font-size: .9rem; padding: 10px 0;
}
.dot {
    width: 8px; height: 8px; border-radius: 50%; background: #1565c0;
    animation: bounce .9s infinite; display: inline-block;
}
.dot:nth-child(2) { animation-delay: .15s; }
.dot:nth-child(3) { animation-delay: .30s; }
@keyframes bounce {
    0%, 80%, 100% { transform: translateY(0);  }
    40%           { transform: translateY(-8px); }
}

/* ── Context preview box ── */
.context-box {
    background: #12181f; border-left: 3px solid #1565c0;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    color: #b0bec5; font-size: .82rem; line-height: 1.6;
    margin-top: 8px; white-space: pre-wrap;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Reusable UI components
# ══════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    st.markdown("""
<div class="app-header">
  <span style="font-size:2.4rem">📄</span>
  <div>
    <h1>PDF Chatbot</h1>
    <p>Powered by Google BERT · ChromaDB · 100% local · No API key needed</p>
  </div>
</div>
""", unsafe_allow_html=True)


def render_sidebar() -> tuple:
    """
    Draw sidebar and return user-selected options.
    Returns: (uploaded_file, top_k, show_context)
    """
    with st.sidebar:
        st.markdown("## 📄 PDF Chatbot")
        st.markdown("---")

        uploaded = st.file_uploader(
            "Upload a PDF", type=["pdf"],
            help="Your file never leaves your machine — all processing is local.",
        )

        st.markdown("---")
        top_k = st.slider(
            "Chunks to retrieve", min_value=2, max_value=10, value=5,
            help="More chunks = richer context, but slightly slower.",
        )
        show_ctx = st.toggle("Show retrieved context", value=False)

        st.markdown("---")
        st.markdown(f"""
<div class="info-card">
<h4>Models</h4>
<p>
🔵 <b>Embeddings</b><br>{EMBED_MODEL}<br><br>
🟢 <b>QA Reader</b><br>{QA_MODEL}
</p>
</div>
<div class="info-card">
<h4>Vector Store</h4>
<p>🟠 <b>ChromaDB</b><br>In-process · persisted locally<br>Cosine similarity search</p>
</div>
<div class="info-card">
<h4>How it works</h4>
<p>
1. PDF → 400-char chunks<br>
2. BERT embeds chunks → ChromaDB<br>
3. Question → top-K chunk retrieval<br>
4. BERT SQuAD reader extracts answer
</p>
</div>
""", unsafe_allow_html=True)

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return uploaded, top_k, show_ctx


def render_user_bubble(text: str) -> None:
    st.markdown(
        f'<div class="msg-user">{text}</div>',
        unsafe_allow_html=True,
    )


def render_bot_bubble(data: dict, show_ctx: bool) -> None:
    conf      = data["score"]
    pages     = data["pages"]
    conf_cls  = "green" if conf >= 70 else ""
    pages_str = ", ".join(f"p.{p}" for p in pages) if pages else "—"

    html = f"""
<div class="msg-bot">
  <div class="answer">{data['answer']}</div>
  <div class="msg-meta">
    <span class="badge {conf_cls}">Confidence: {conf}%</span>
    <span class="badge">📄 {pages_str}</span>
  </div>
"""
    if show_ctx and data.get("context"):
        preview = data["context"][0][:300].replace("<", "&lt;").replace(">", "&gt;")
        html += f'<div class="context-box">{preview}…</div>'

    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_thinking() -> st.empty:
    ph = st.empty()
    ph.markdown("""
<div class="thinking">
  <span class="dot"></span>
  <span class="dot"></span>
  <span class="dot"></span>
  &nbsp;BERT is searching ChromaDB…
</div>""", unsafe_allow_html=True)
    return ph


def render_chat_history(show_ctx: bool) -> None:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user_bubble(msg["content"])
        else:
            render_bot_bubble(msg["data"], show_ctx)


# ══════════════════════════════════════════════════════════════════════════
#  Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════

def init_session() -> None:
    defaults = {
        "messages":   [],
        "db":         None,
        "pdf_name":   None,
        "chroma_dir": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ══════════════════════════════════════════════════════════════════════════
#  Main application
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    inject_css()
    init_session()

    render_header()
    uploaded, top_k, show_ctx = render_sidebar()

    # ── Handle new PDF upload ──────────────────────────────────────────────
    if uploaded and uploaded.name != st.session_state.pdf_name:
        # Free previous ChromaDB persist directory
        cleanup_chroma_dir(st.session_state.chroma_dir)

        st.session_state.messages  = []
        st.session_state.pdf_name  = uploaded.name

        with st.spinner(f"Processing **{uploaded.name}** with BERT + ChromaDB…"):
            t0 = time.time()
            db, n_pages, n_chunks, chroma_dir = cached_process_pdf(
                uploaded.read(), uploaded.name
            )
            elapsed = round(time.time() - t0, 1)

        st.session_state.db         = db
        st.session_state.chroma_dir = chroma_dir

        st.success(
            f"✅ **{uploaded.name}** indexed — "
            f"{n_pages} pages · {n_chunks} chunks · {elapsed}s"
        )

    # ── Render existing chat history ───────────────────────────────────────
    render_chat_history(show_ctx)

    # ── Chat input ─────────────────────────────────────────────────────────
    if not uploaded:
        st.info("👈 Upload a PDF from the sidebar to get started.")
        return

    question = st.chat_input("Ask anything about your PDF…")
    if not question:
        return

    if st.session_state.db is None:
        st.warning("PDF is still indexing — please wait a moment.")
        return

    # Show user bubble immediately
    render_user_bubble(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Thinking indicator while backend runs
    thinking_ph = render_thinking()

    # Delegate to backend — no ML code here
    data = answer_question(question, st.session_state.db, top_k=top_k)

    thinking_ph.empty()
    render_bot_bubble(data, show_ctx)

    st.session_state.messages.append(
        {"role": "assistant", "content": data["answer"], "data": data}
    )


# ── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
