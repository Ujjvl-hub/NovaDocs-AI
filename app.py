
import os
import shutil
import tempfile
import time

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

load_dotenv()

CHROMA_DIR = "chroma_db"

st.set_page_config(
    page_title="NovaDocs AI",
    page_icon="📚",
    layout="wide"
)

# -------------------------------------------------
# CACHE MODELS
# -------------------------------------------------

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest"
    )


# -------------------------------------------------
# PROMPT
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful AI assistant.

            Use ONLY the provided context.

            If the answer is not present in the context,
            reply exactly:

            I could not find the answer.
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Question:
            {question}
            """
        )
    ]
)

# -------------------------------------------------
# VECTORSTORE CREATION
# -------------------------------------------------

def build_vectorstore_from_pdfs(uploaded_files):

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    all_docs = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:

            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            all_docs.extend(docs)

        except Exception as e:

            st.error(
                f"Error reading {uploaded_file.name}: {e}"
            )

        finally:

            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_docs)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_DIR
    )

    return vectorstore, len(all_docs), len(chunks)


def load_existing_vectorstore():

    embedding_model = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_model
    )

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "book_name" not in st.session_state:
    st.session_state.book_name = None

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.header("📤 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"📚 Selected {len(uploaded_files)} file(s)"
        )

        if st.button(
            "Process PDFs",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Reading PDFs and creating embeddings..."
            ):

                vectorstore, pages, chunks = (
                    build_vectorstore_from_pdfs(
                        uploaded_files
                    )
                )

                st.session_state.vectorstore = (
                    vectorstore
                )

                st.session_state.book_name = (
                    f"{len(uploaded_files)} PDFs"
                )

                st.session_state.messages = []

            st.success(
                f"Indexed {pages} pages into {chunks} chunks"
            )

    if (
        st.session_state.vectorstore is None
        and os.path.exists(CHROMA_DIR)
    ):

        if st.button(
            "Load Existing Index",
            use_container_width=True
        ):

            st.session_state.vectorstore = (
                load_existing_vectorstore()
            )

            st.session_state.book_name = (
                "Previously Indexed PDFs"
            )

    st.divider()

    if st.session_state.book_name:

        st.info(
            f"📖 Active Collection:\n\n{st.session_state.book_name}"
        )

    else:

        st.warning(
            "No PDFs uploaded yet."
        )

    with st.expander(
        "⚙ Retrieval Settings"
    ):

        k = st.slider(
            "Chunks Returned (k)",
            2,
            10,
            4
        )

        fetch_k = st.slider(
            "Fetch Pool Size",
            k,
            20,
            max(10, k)
        )

        lambda_mult = st.slider(
            "MMR Diversity",
            0.0,
            1.0,
            0.5
        )

    if st.button(
        "🗑 Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    if st.session_state.messages:

        chat_history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.messages
            ]
        )

        st.download_button(
            label="⬇ Download Chat",
            data=chat_history,
            file_name="chat_history.txt"
        )

# -------------------------------------------------
# MAIN UI
# -------------------------------------------------

st.title("📚 NovaDocs AI")

st.caption(
    "Upload PDFs and chat with them using RAG + Mistral AI"
)

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

query = st.chat_input(
    "Ask a question about your PDFs..."
)

if query:

    if st.session_state.vectorstore is None:

        st.error(
            "Please upload and process PDFs first."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        with st.chat_message("user"):

            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                retriever = (
                    st.session_state.vectorstore
                    .as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": k,
                            "fetch_k": fetch_k,
                            "lambda_mult": lambda_mult
                        }
                    )
                )

                start = time.time()

                docs = retriever.invoke(query)

                end = time.time()

                context = "\n\n".join(
                    doc.page_content[:1000]
                    for doc in docs
                )

                final_prompt = prompt.invoke(
                    {
                        "context": context,
                        "question": query
                    }
                )

                llm = get_llm()

                response = llm.invoke(
                    final_prompt
                )

                st.markdown(
                    response.content
                )

                st.caption(
                    f"⚡ Retrieved {len(docs)} chunks in {end-start:.2f}s"
                )

                with st.expander(
                    "📄 Sources Used"
                ):

                    for i, doc in enumerate(
                        docs,
                        start=1
                    ):

                        page = doc.metadata.get(
                            "page",
                            "N/A"
                        )

                        st.markdown(
                            f"### Chunk {i} (Page {page})"
                        )

                        st.write(
                            doc.page_content
                        )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response.content
            }
        )

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.divider()

st.caption(
    "Built with Streamlit • LangChain • ChromaDB • HuggingFace • Mistral AI"
)

