import os
import tempfile
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


# Load API keys from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set in the .env file.")
    st.stop()


# Page settings
st.set_page_config(
    page_title="Multi-PDF RAG",
    page_icon="📚"
)

st.title("📚 Multi-PDF RAG App")
st.write("Upload multiple PDFs and ask questions about them.")


# Store the vector database in session state
if "store" not in st.session_state:
    st.session_state.store = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []


# Upload multiple PDFs
files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)


# Index the uploaded PDFs
if files:

    if st.button("Index PDFs"):

        chunks = []

        with st.spinner("Reading and indexing PDFs..."):

            for file in files:

                # Save uploaded PDF temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as temp_file:

                    temp_file.write(file.getbuffer())
                    pdf_path = temp_file.name

                # Load PDF
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                # Add original filename to metadata
                for document in documents:
                    document.metadata["source"] = file.name

                # Split PDF into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=150
                )

                pdf_chunks = splitter.split_documents(documents)

                chunks.extend(pdf_chunks)

                # Remove temporary file
                os.remove(pdf_path)

            # Create OpenAI embeddings
            embeddings = OpenAIEmbeddings(
                api_key=OPENAI_API_KEY
            )

            # Create ONE FAISS vector store for all PDFs
            st.session_state.store = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.session_state.chunks = chunks

        st.success(f"Indexed {len(chunks)} chunks.")


# Ask questions
if st.session_state.store:

    st.subheader("Ask a question")

    question = st.text_input(
        "Enter your question:"
    )

    if st.button("Ask") and question:

        # Retrieve matching chunks
        retriever = st.session_state.store.as_retriever(
            search_kwargs={"k": 4}
        )

        matching_docs = retriever.invoke(question)

        # Create context from retrieved documents
        context = "\n\n".join(
            [
                f"Source: {doc.metadata.get('source', 'Unknown')}, "
                f"Page: {doc.metadata.get('page', 0) + 1}\n"
                f"{doc.page_content}"
                for doc in matching_docs
            ]
        )

        # OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=OPENAI_API_KEY
        )

        # Prompt
        prompt = f"""
You are a helpful assistant answering questions from uploaded PDF documents.

Use ONLY the information provided in the context below.

If the answer is not available in the context, say:
"I could not find this information in the uploaded PDFs."

Context:
{context}

Question:
{question}

Give a clear and simple answer.
"""

        # Generate answer
        response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        # Show sources
        st.subheader("Sources")

        sources = []

        for doc in matching_docs:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                0
            ) + 1

            source_text = f"{source} — Page {page}"

            if source_text not in sources:
                sources.append(source_text)

        for source in sources:
            st.write(f"📄 {source}")