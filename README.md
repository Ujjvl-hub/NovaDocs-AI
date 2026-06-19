# 📚 NovaDocs AI

NovaDocs AI is a Retrieval-Augmented Generation (RAG) application that allows users to upload one or more PDF documents and interact with them through natural language conversations.

The application uses semantic search and vector embeddings to retrieve the most relevant information from uploaded documents and generates context-aware responses using Mistral AI.

---

## 🚀 Features

* 📄 Upload and process multiple PDF documents
* 🔍 Semantic search using Hugging Face embeddings
* 🧠 Retrieval-Augmented Generation (RAG)
* 💬 Conversational chat interface
* 📚 Multi-document knowledge base
* ⚡ Fast retrieval using ChromaDB
* 🎯 Maximum Marginal Relevance (MMR) retrieval
* 📄 Source chunk visualization
* 📥 Download chat history
* 🗑️ Clear chat functionality
* 📊 Retrieval statistics display

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Backend

* Python
* LangChain

### Vector Database

* ChromaDB

### Embedding Model

* sentence-transformers/all-MiniLM-L6-v2

### Large Language Model

* Mistral AI (mistral-small-latest)

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

---

## 📂 Project Structure

```text
NovaDocs-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── chroma_db/      # Generated at runtime
├── uploads/        # User uploads
└── .venv/          # Virtual environment
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/NovaDocs-AI.git
cd NovaDocs-AI
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_api_key_here
```

You can obtain your API key from Mistral AI.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 📖 How It Works

1. User uploads one or more PDF documents.
2. Documents are loaded using PyPDFLoader.
3. Text is split into chunks using RecursiveCharacterTextSplitter.
4. Chunks are converted into vector embeddings.
5. Embeddings are stored in ChromaDB.
6. User asks a question.
7. Relevant chunks are retrieved using MMR search.
8. Retrieved context is sent to Mistral AI.
9. AI generates an answer grounded in the uploaded documents.
10. Source chunks are displayed for transparency.

---


## 🎯 Future Improvements

* Support for DOCX and TXT files
* Conversation memory
* Hybrid search (keyword + semantic)
* PDF summarization
* User authentication
* Cloud deployment
* Citation-based responses
* Chat export in PDF format

---

## 💡 Sample Questions

* What are the key topics discussed in the document?
* Summarize chapter 3.
* What skills are mentioned in the resume?
* Explain the networking concepts covered in the PDF.
* List all important dates mentioned.

---

## 👨‍💻 Author

Ujjwal Kumar

* LinkedIn: https://www.linkedin.com/in/ujjwal-kumar-8a4b66310/
* GitHub: https://github.com/Ujjvl-hub

---

## ⭐ If you found this project useful, consider giving it a star!
