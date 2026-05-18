"""
chatbot.py — Core backend logic (no UI, no Streamlit)
──────────────────────────────────────────────────────
Responsibilities:
  - Load BERT embedding model
  - Load BERT QA model
  - Load & chunk PDF files
  - Build and query ChromaDB vector store
  - Return structured answer dicts

Import this module from app.py (or any other frontend).
"""

import os, tempfile, uuid, shutil
from typing import List, Dict, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from transformers import pipeline as hf_pipeline

# ── Model identifiers ──────────────────────────────────────────────────────
EMBED_MODEL = "google-bert/bert-base-uncased"   # semantic search embeddings
QA_MODEL    = "deepset/bert-base-cased-squad2"  # extractive QA (SQuAD2)

# ── Module-level singletons (loaded once per process) ──────────────────────
_embeddings = None
_qa_pipeline = None


# ══════════════════════════════════════════════════════════════════════════
#  Model loaders
# ══════════════════════════════════════════════════════════════════════════

def get_embeddings() -> HuggingFaceEmbeddings:
    """Return (and lazily load) the BERT embedding model."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_qa_pipeline():
    """Return (and lazily load) the BERT SQuAD2 QA pipeline."""
    global _qa_pipeline
    if _qa_pipeline is None:
        _qa_pipeline = hf_pipeline(
            "question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device=-1,           # CPU; set to 0 for GPU
            max_answer_len=120,
            handle_impossible_answer=True,
        )
    return _qa_pipeline


# ══════════════════════════════════════════════════════════════════════════
#  PDF processing
# ══════════════════════════════════════════════════════════════════════════

def load_and_chunk_pdf(pdf_path: str) -> Tuple[List[Document], int]:
    """
    Read a PDF from disk, split into overlapping chunks.

    Args:
        pdf_path: Absolute path to the PDF file.

    Returns:
        (chunks, n_pages) where chunks is a list of LangChain Documents.
    """
    pages = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    return chunks, len(pages)


def build_chroma_index(
    chunks: List[Document],
) -> Tuple[Chroma, str]:
    """
    Embed chunks with BERT and store them in a new ChromaDB collection.

    Args:
        chunks: List of Document objects to embed.

    Returns:
        (Chroma vectorstore, persist_directory path)
    """
    persist_dir = tempfile.mkdtemp(prefix="chroma_")
    collection  = f"pdf_{uuid.uuid4().hex[:8]}"

    db = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=collection,
        persist_directory=persist_dir,
    )
    return db, persist_dir


def process_pdf_bytes(
    file_bytes: bytes,
    filename: str,
) -> Tuple[Chroma, int, int, str]:
    """
    Full pipeline: raw PDF bytes → ChromaDB index.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        filename:   Original filename (used only for labelling).

    Returns:
        (Chroma db, n_pages, n_chunks, persist_directory)
    """
    # Write bytes to a temp file so PyPDFLoader can read it
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        chunks, n_pages = load_and_chunk_pdf(tmp_path)
        db, persist_dir = build_chroma_index(chunks)
        return db, n_pages, len(chunks), persist_dir
    finally:
        os.unlink(tmp_path)


def cleanup_chroma_dir(path: str) -> None:
    """Remove a ChromaDB persist directory to free disk space."""
    if path and os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════
#  Question answering
# ══════════════════════════════════════════════════════════════════════════

def answer_question(
    question: str,
    db: Chroma,
    top_k: int = 5,
) -> Dict:
    """
    Retrieve relevant chunks from ChromaDB, then run BERT QA reader.

    Args:
        question: User's natural-language question.
        db:       ChromaDB vectorstore built from the PDF.
        top_k:    Number of chunks to retrieve (default 5).

    Returns:
        {
          "answer":  str,          # extracted answer span
          "score":   float,        # confidence 0–100
          "pages":   List[int],    # 1-indexed source page numbers
          "context": List[str],    # raw retrieved chunk texts
        }
    """
    # Retrieve top-k chunks by cosine similarity
    docs_scores = db.similarity_search_with_relevance_scores(question, k=top_k)
    docs_scores.sort(key=lambda x: x[1], reverse=True)  # highest relevance first

    parts: List[str] = []
    pages: List[int] = []
    for doc, _ in docs_scores:
        parts.append(doc.page_content.strip())
        pg = doc.metadata.get("page")
        if isinstance(pg, int):
            pages.append(pg + 1)           # convert 0-indexed to 1-indexed

    # Truncate context to stay within BERT's 512-token window
    context = "\n\n".join(parts)[:3000]

    # Run BERT extractive QA
    result = get_qa_pipeline()(question=question, context=context)
    answer = result.get("answer", "").strip()
    score  = result.get("score", 0.0)

    if not answer or score < 0.01:
        answer = (
            "I couldn't find a confident answer in the document. "
            "Try rephrasing your question."
        )

    return {
        "answer":  answer,
        "score":   round(score * 100, 1),
        "pages":   sorted(set(pages)),
        "context": parts,
    }
