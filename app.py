import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")



prompt = ChatPromptTemplate.from_template(
    "Use the following pieces of context to answer the question at the end. "
    "If you don't know the answer, just say that you don't know, don't try to make up an answer. "
    "\n\n{context}\n\nQuestion: {question}\nAnswer:"
)

def create_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.documents = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        st.session_state.docs = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.docs, st.session_state.embeddings)


user_prompt = st.text_input("Enter your question about the research papers:")
if st.button("Get Answer"):
    create_embeddings()

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
    retriever = st.session_state.vector_store.as_retriever()

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}  # retrieves docs
        | prompt  # formats input to prompt
        | llm     # sends to LLM
    )

    import time
    start = time.process_time()

    response = rag_chain.invoke(user_prompt)

    end = time.process_time() - start
    st.write(f"⏱️ Time taken: {end:.2f} seconds")
    st.subheader("Answer:")
    st.write(response.content)

