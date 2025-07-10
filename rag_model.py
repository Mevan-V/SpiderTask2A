# --- Imports ---

import os
import requests
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import pickle
import logging
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_groq import ChatGroq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---

PAPER_DETAILS = {
    "Attention Is All You Need": "1706.03762",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding": "1810.04805",
    "GPT-3: Language Models are Few-Shot Learners": "2005.14165",
    "Contrastive Language-Image Pretraining with Knowledge Graphs": "2210.08901",
    "LLaMA: Open and Efficient Foundation Language Models": "2302.13971"
}
PDF_DIR = "research_papers"
FAISS_INDEX_PATH = "faiss_index.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # A good lightweight model for embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure the PDF directory exists
os.makedirs(PDF_DIR, exist_ok=True)

# Downloads research papers from arXiv if they don't already exist locally
def download_papers(paper_details, download_dir):
    logging.info("Starting paper download...")

    for title, arxiv_id in paper_details.items():
        pdf_filename = os.path.join(download_dir, f"{title.replace(':', '').replace('/', '_')}.pdf")

        if os.path.exists(pdf_filename):
            logging.info(f"'{title}' already exists. Skipping download.")
            continue

        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        try:
            response = requests.get(arxiv_url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(pdf_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"Successfully downloaded '{title}' to {pdf_filename}")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading '{title}' from {arxiv_url}: {e}")

    logging.info("Paper download complete.")

# Parses text from PDF files
def parse_pdfs(pdf_dir):
    logging.info("Starting PDF parsing...")
    documents = []

    for filename in os.listdir(pdf_dir):

        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() if page.extract_text() else ""

                if text:
                    # LangChain Document object
                    documents.append({"page_content": text, "metadata": {"source": filename}})
                    logging.info(f"Parsed text from: {filename}")
                else:
                    logging.warning(f"No text extracted from: {filename}")

            except Exception as e:
                logging.error(f"Error parsing {filename}: {e}")

    logging.info("PDF parsing complete.")
    return documents

# Splits long documents into smaller and overlapping text chunks
def chunk_documents(documents, chunk_size, chunk_overlap):
    logging.info(f"Starting document chunking with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    # Convert custom dicts to LangChain Document objects for text_splitter
    langchain_docs = []
    for doc in documents:
        from langchain.docstore.document import Document # Import here to avoid circular dependency if placed at top

        lc_doc = Document(page_content=doc['page_content'], metadata=doc['metadata'])
        langchain_docs.append(lc_doc)

    chunks = text_splitter.split_documents(langchain_docs)

    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

# Creates a FAISS vector database from the text chunks
def create_vector_db(chunks, embedding_model_name, faiss_index_path):
    logging.info(f"Initializing embeddings with model: {embedding_model_name}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    logging.info("Creating FAISS vector database...")
    db = FAISS.from_documents(chunks, embeddings)
    logging.info("FAISS vector database created.")

    logging.info(f"Persisting FAISS index to {faiss_index_path}...")
    with open(faiss_index_path, "wb") as f:
        pickle.dump(db, f)
    logging.info("FAISS index persisted successfully.")

    return db

# Loads the FAISS vector database
def load_vector_db(faiss_index_path, embedding_model_name):
    logging.info(f"Loading FAISS index from {faiss_index_path}...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    with open(faiss_index_path, "rb") as f:
        db = pickle.load(f)
    logging.info("FAISS index loaded successfully.")

    return db

# Full documents pipeline (download, parse, chunk, embed & store)
def ingest_documents():

    download_papers(PAPER_DETAILS, PDF_DIR)
    raw_documents = parse_pdfs(PDF_DIR)
    chunks = chunk_documents(raw_documents, CHUNK_SIZE, CHUNK_OVERLAP)
    vector_db = create_vector_db(chunks, EMBEDDING_MODEL_NAME, FAISS_INDEX_PATH)

    return vector_db

# Can run this script prematurely to download and store the vector db
if __name__ == '__main__':
    if not os.path.exists(FAISS_INDEX_PATH):
        print("FAISS index not found. Ingesting documents...")
        ingest_documents()
    else:
        print("FAISS index already exists. Skipping ingestion.")

# --- RAG Pipeline Configuration ---

# Running online
LLM_MODEL = "llama3-8b-8192"
# LLM_MODEL = "llama2"

# Creates the desired RAG pipeline using LCEL
def create_rag_chain(vector_db, groq_api_key):
    logging.info(f"Initializing LLM with model: {LLM_MODEL}...")

    try:
        llm = ChatGroq(model="llama3-8b-8192", groq_api_key=groq_api_key)
    except Exception as e:
        raise Exception(f"LLM initialization failed. Reason: {e}")

    logging.info("Creating retriever from vector database...")
    retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

    template = """You are a helpful and knowledgeable assistant.
    Based on the provided context, answer the following question. Please only stick to the given context. 
    If there is no relevent context, then ask the user to ask questions related to the given research papers only!.

    Context: {context}
    
    Question: {question}
    
    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # Define the RAG chain using LCEL
    rag_chain = (
        {"context": itemgetter("context"), "question": itemgetter("question")}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("RAG chain created.")
    return rag_chain, retriever

# The question and answer process using the RAG pipeline
def answer_question(rag_chain, retriever, question):

    # Explicitly retrieving the related the documents
    retrieved_docs = retriever.invoke(question)
    logging.info(f"\n--- Retrieved Documents for Query: '{question}' ---")

    # Extracting the required content from the retrieved docs
    retrieved_content = []
    for i, doc in enumerate(retrieved_docs):
        logging.info(f"Document {i+1} (Source: {doc.metadata.get('source', 'N/A')}, Start Index: {doc.metadata.get('start_index', 'N/A')}):")
        logging.info(f"Preview: {doc.page_content[:200]}...") # Log first 200 chars
        retrieved_content.append(doc.page_content)

    # Now invoke the RAG chain
    answer = rag_chain.invoke({"context": "\n\n".join(retrieved_content), "question": question})
    logging.info(f"Generated answer: {answer}")

    return answer