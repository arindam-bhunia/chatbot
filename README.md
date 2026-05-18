# 📄 PDF Chatbot — Google BERT + ChromaDB

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

# Activate the virtual environment
# Linux / macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# Note: First run may take time as it downloads ~400MB of BERT models.




