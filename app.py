import streamlit as st
import os
import logging
import sys

# Append parent directory to sys.path to allow importing rag_model.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Import functions from the rag_model script
from rag_model import ingest_documents, load_vector_db, create_rag_chain, answer_question, FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME, PAPER_DETAILS, PDF_DIR, LLM_MODEL

# Set up logging for the Streamlit app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Basic setup of streamlit page
st.set_page_config(page_title="Research Paper RAG System", layout="wide")
st.title("📚 Research Paper Q&A with RAG")
st.markdown("Ask questions about the provided machine learning research papers and get answers powered by a Retrieval-Augmented Generation (RAG) system.")

# Setting up the RAG system including document ingestion if needed
@st.cache_resource
def setup_rag_system():

    # Document ingestion if FAISS vector db is not present
    if not os.path.exists(FAISS_INDEX_PATH):
        with st.spinner("FAISS index not found. Ingesting documents (this may take a few minutes)..."):
            logging.info("Attempting to ingest documents as FAISS index is not found.")
            try:
                ingest_documents()
                st.success("Documents ingested and FAISS index created successfully!")
            except Exception as e:
                st.error(f"Error during document ingestion: {e}.")
                logging.error(f"Error during document ingestion: {e}")
                st.stop() # Stop the app if ingestion fails
    
    # Setting up the RAG pipeline
    with st.spinner("Loading RAG system components..."):
        try:
            vector_db = load_vector_db(FAISS_INDEX_PATH, EMBEDDING_MODEL_NAME)
            rag_chain, retriever = create_rag_chain(vector_db)
            st.success("RAG system ready!")
            return rag_chain, retriever
        except Exception as e:
            st.error(f"Error loading RAG system: {e}.")
            logging.error(f"Error loading RAG system: {e}")
            st.stop() # Stop the app if loading fails

rag_chain, retriever = setup_rag_system()

# Display available papers
st.sidebar.header("Available Research Papers")
for title in PAPER_DETAILS.keys():
    st.sidebar.markdown(f"- {title}")

# --- Q&A Interface ---

st.header("Ask a Question")
user_question = st.text_input("Enter your question here:")

if user_question:
    st.write(f"Thinking about: *{user_question}*...")
    with st.spinner("Generating answer..."):
        try:
            # Pass the retriever for manual retrieval and logging purposes as explained in the rag_model.py
            answer = answer_question(rag_chain, retriever, user_question)
            st.markdown("---")
            st.subheader("Answer:")
            st.info(answer)
        except Exception as e:
            st.error(f"An error occurred while generating the answer: {e}")
            logging.error(f"Error during Q&A: {e}")

st.markdown("---")
st.write("Project by Mevan") # Placeholder for your information