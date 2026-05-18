# 📄 PDF Chatbot — RAG and LangChain

**A fully local, private, and fast PDF chatbot** powered by **Google BERT** (embeddings + extractive QA) and **ChromaDB**. No API keys, no internet required after installation.

Built with **LangChain**, **Streamlit**, and Hugging Face Transformers.

---

## ✨ Features

- **100% Local** — Everything runs on your machine
- **No API costs** or keys required
- **Fast BERT-based** semantic search + extractive QA
- **Beautiful Streamlit UI** with chat history and context preview
- **Persistent vector store** using ChromaDB
- **Source page tracking** with confidence scores
- **Clean chunking** with overlapping text segments

---

## 🛠 Tech Stack

| Component           | Technology                          |
|---------------------|-------------------------------------|
| UI                  | Streamlit                           |
| PDF Parsing         | PyPDFLoader                         |
| Text Splitting      | RecursiveCharacterTextSplitter      |
| Embeddings          | `google-bert/bert-base-uncased`     |
| Vector Store        | ChromaDB                            |
| QA Reader           | `deepset/bert-base-cased-squad2`    |
| Framework           | LangChain                           |

---

## 🚀 Installation

### 1. Clone the repository
### 2. Create a virtual environment (recommended)
python -m venv venv

**Activate the virtual environment**
Linux / macOS:

source venv/bin/activate
Windows:
venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

Note: First run may take time as it downloads ~400MB of BERT models.

## 🎯 How to Run

```bash
streamlit run app.py
```

1. Upload a PDF from the sidebar
2. Wait for indexing
3. Start asking questions!

## 📁 Project Structure
pdf-chatbot-bert/
├── app.py # Streamlit UI
├── chatbot.py # Core backend logic
├── requirements.txt # Dependencies
├── README.md
└── chroma_/ # Auto-generated (temporary)

## 🔍 How It Works

1. PDF → Text extraction
2. Split into overlapping chunks (~400 chars)
3. Generate embeddings using BERT
4. Store in ChromaDB vector database
5. User question → Semantic search (top-k chunks)
6. BERT QA model extracts precise answer


## 🎨 UI Features

- Modern dark theme
- Confidence score with color coding
- Source page references
- Optional context preview
- Session chat history
- One-click clear chat

## 🧠 Models
- **Embedding Model**: `google-bert/bert-base-uncased`
- **QA Model**: `deepset/bert-base-cased-squad2`

---

## 📌 Limitations
- Extractive QA only (cannot generate creative answers)
- Limited by BERT's 512 token context
- First model download is slow

---

## 🛠️ Future Enhancements
- Multi-document support
- Better reranking
- Larger / faster embedding models
- Docker support
- Persistent chat history
