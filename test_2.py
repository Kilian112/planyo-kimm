import os
import glob
import logging
import PyPDF2
from ollama_client import OllamaClient
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize OpenAI and Pinecone API keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=3072,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=pinecone_env)
    )
pinecone_index = pc.Index(pinecone_index_name)

# Ollama setup
ollama_client = OllamaClient()
ollama_embedding_model = "embedding_model_name"

# Function to clean text
def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    logging.info(f"Extracting text from {file_path}")
    text_data = []
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_data.append((i + 1, clean_text(text.strip())))
        logging.info(f"Extracted {len(text_data)} pages from {file_path}")
    except Exception as e:
        logging.error(f"Failed to extract text from {file_path}: {e}")
    return text_data

# Function to generate embeddings
def query_ollama_embeddings(texts):
    try:
        embeddings = []
        for text in texts:
            embedding = ollama_client.embed(text, ollama_embedding_model)
            embeddings.append(embedding)
        return embeddings
    except Exception as e:
        logging.error(f"Exception during Ollama API request: {e}")
        return None

# Function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(vectors, namespace):
    try:
        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        logging.info(f"Upserted {len(vectors)} vectors to Pinecone under namespace '{namespace}'")
    except Exception as e:
        logging.error(f"Error upserting to Pinecone: {e}")

# Main function to process all documents in knowledge_base_docs
def process_knowledge_base_docs():
    # Check if the directory exists
    knowledge_base_path = "./knowledge_base_docs/*.pdf"
    if not os.path.exists('./knowledge_base_docs'):
        logging.error("Directory 'knowledge_base_docs' does not exist.")
        return
    
    # List all PDF files in the directory
    files = glob.glob(knowledge_base_path)
    if not files:
        logging.warning("No PDF files found in 'knowledge_base_docs'.")
        return

    logging.info(f"Found {len(files)} PDF files in 'knowledge_base_docs': {files}")

    # Process each file
    for file_path in files:
        filename = os.path.basename(file_path)
        namespace = f"knowledge_base_{filename}"

        logging.info(f"Processing file: {filename} in namespace: {namespace}")
        
        # Extract text
        text_data = extract_text_from_pdf(file_path)
        if not text_data:
            logging.warning(f"No text extracted from {filename}. Skipping.")
            continue

        # Prepare vectors for Pinecone
        vectors = []
        for page_number, page_text in text_data:
            embeddings = query_ollama_embeddings([page_text])
            if embeddings:
                vector_id = f"{filename}_page_{page_number}"
                vector = {"id": vector_id, "values": embeddings[0], "metadata": {
                    "document_id": vector_id,
                    "filename": filename,
                    "page_number": page_number,
                    "paragraph": page_text,
                    "source_type": "knowledge_base",
                    "namespace": namespace
                }}
                vectors.append(vector)
        
        # Upsert to Pinecone
        if vectors:
            upsert_embeddings_to_pinecone(vectors, namespace)
        else:
            logging.warning(f"No vectors generated for {filename}. Skipping upsert.")

# Run the test
if __name__ == "__main__":
    process_knowledge_base_docs()

