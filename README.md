# Q-A-Chatbot-with-RAG


## 🛠 Features

- Upload and load multiple PDF documents
- Split documents into chunks for better semantic search
- Embed documents using **HuggingFace Sentence Transformers**
- Store embeddings in **FAISS vector store** for fast similarity search
- Integrate with **Groq LLM** (`llama-3.1-8b-instant`) for generating answers
- Streamlit web interface for interactive Q&A
- Optional display of source documents (context) used in answers
- Fully local embedding support (no external API costs)
- `.env` file support for API keys (hidden from GitHub)

---

## 💻 Tech Stack

- **Python 3.10+**
- **LangChain**
- **LangChain-HuggingFace** embeddings
- **FAISS** for vector storage
- **Streamlit** for UI
- **Groq API** for LLM