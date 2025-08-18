import os
import glob
import logging
import unicodedata
import re
import PyPDF2
from datetime import datetime
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from ollama_client import OllamaClient

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Pinecone API keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_ENV")

# Ollama setup
ollama_client = OllamaClient()
ollama_embedding_model = "embedding_model_name"
ollama_llm = "llm_model_name"

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,  # Adjust to match model embedding dimensions
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=pinecone_env)
    )
pinecone_index = pc.Index(pinecone_index_name)

# Mapping of filenames to URLs
filename_to_url = {
    "18.07.2007 - VIII ZR 285_06.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=6e3f9aaf3a1e5c007680dd54d995d612&nr=40921&anz=1&pos=0&Blank=1.pdf",
    "17.12.2014 - VIII ZR 87_13.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=5cd8f9af8b0512cb5c3a995e65403abb&nr=70069&anz=1&pos=0&Blank=1.pdf",
    "17.12.2014 - VIII ZR 88_13.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=de14277d67adc269b46c3d2beaa6d2d4&nr=70070&anz=1&pos=0&Blank=1.pdf",
    "17.12.2014 - VIII ZR 89_13.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=b81ad0dc1ddb5fe46c999f91263925c0&nr=70071&anz=1&pos=0&Blank=1.pdf",
    "16.05.2023 - VIII ZR 106_21.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=4c206d10762c458c9fae14f4ea360056&nr=133960&anz=1&pos=0&Blank=1.pdf",
    "23.11.2022 - VIII ZR 59_21.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=23451a84deb7e3561478ab26ded5e323&nr=132187&anz=1&pos=0&Blank=1.pdf",
    "20.07.2022 - VIII ZR 361_21.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=79c9bda06cab8eb1dc4e9509d4b9eb8d&nr=130938&anz=1&pos=0&Blank=1.pdf",
    "21.02.2023 - VIII ZR 106_21.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=9bf6d13e2eaa471cbfab746c92193ca7&nr=134061&anz=1&pos=0&Blank=1.pdf",
    "17.06.2020 - VIII ZR 81_19.pdf": "https://juris.bundesgerichtshof.de/cgi-bin/rechtsprechung/document.py?Gericht=bgh&Art=en&sid=e971f4d0276560780414b03c1994eb40&nr=108738&anz=1&pos=0&Blank=1.pdf",
    "01.05.2024 - Analysen_und_Empfehlungen.pdf": "https://www.bbsr.bund.de/BBSR/DE/veroeffentlichungen/bbsr-online/2024/bbsr-online-87-2024.html",
    "01.06.2018 - Berücksichtigung_des_Nutzerverhaltens.pdf": "https://www.bbsr.bund.de/BBSR/DE/veroeffentlichungen/bbsr-online/2019/bbsr-online-04-2019.html",
    "01.08.2016 - Neue_Ansichten_auf_die_Wohnungsmieten.pdf": "https://www.bbsr.bund.de/BBSR/DE/veroeffentlichungen/analysen-kompakt/2016/ak-08-2016.html",
    "01.08.2021 - BGH-VIII ZR 8119.pdf": "https://www.juris.de/jportal/nav/produkte/werk/wohnungswirtschaft-und-mietrecht-(wum).jsp",
    "01.11.2016 - Mietrecht_und_energetische_Sanierung.pdf": "https://www.bbsr.bund.de/BBSR/DE/veroeffentlichungen/sonderveroeffentlichungen/2016/mietrecht-energetische-sanierung-eu.html",
    "03.06.2024 - Mieterhöhung_nach_Modernisierungsmaßnahmen.pdf": "https://beck-online.beck.de/?vpath=bibdata%2Fkomm%2FMuekoBGB_Band3%2FBGB%2Fcont%2FMuekoBGB.BGB.P557.T0.htm",
    "12.01.2021 - Die_aktuelle_Rechtsprechung.pdf": "https://www.beck-shop.de/boerstinghaus-kuendigungs-handbuch/product/16036127",
    "23.04.2024 - Energetische_Ausstattungsmerkmale.pdf": "https://www.bbsr.bund.de/BBSR/DE/veroeffentlichungen/analysen-kompakt/2024/ak-05-2024.html",
    "26.06.2012 - Kosten_energierelevanter.pdf": "https://www.bbsr.bund.de/BBSR/DE/veroeffentlichungen/ministerien/bmvbs/bmvbs-online/2012/ON072012.html",
    "12.07.2019 - David_Funktionales_Kostensplitting_Dissertation_HCU_2019.pdf": "https://repos.hcu-hamburg.de/bitstream/hcu/505/1/David_Funktionales_Kostensplitting_Dissertation_HCU_2019.pdf"
}

# Function to sanitize filenames for namespaces
def sanitize_filename(filename):
    # Remove accents and non-ASCII characters
    filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    # Replace unsupported characters with underscores
    filename = re.sub(r'[^A-Za-z0-9_.-]', '_', filename)
    return filename

# Function to clean text
def clean_text(text):
    # Ensure text is in UTF-8 format
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

# Function to chunk text for large files
def chunk_text(text_list, max_tokens=500):
    # Split text into smaller chunks to fit token limit
    chunks = []
    current_chunk = []
    current_size = 0

    for page_number, page_text in text_list:
        # Approximate token size based on text length
        token_size = len(page_text) // 4

        if current_size + token_size > max_tokens:
            chunks.append(current_chunk)
            current_chunk = []
            current_size = 0

        current_chunk.append((page_number, page_text))
        current_size += token_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks

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

# Function to extract court decision date from filename
def extract_date_from_filename(filename):
    try:
        # Assume the date format is "DD.MM.YYYY -"
        date_str = filename.split(" - ")[0]
        return datetime.strptime(date_str, "%d.%m.%Y").isoformat()
    except Exception as e:
        logging.error(f"Failed to parse date from filename '{filename}': {e}")
        return None

# Function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(vectors, namespace):
    try:
        # Perform upsert operation to Pinecone
        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        logging.info(f"Upserted {len(vectors)} vectors to Pinecone under namespace '{namespace}'")
    except Exception as e:
        logging.error(f"Error upserting to Pinecone: {e}")

# Main function to process all documents in the knowledge base
def process_knowledge_base_docs():
    # Define the knowledge base document path
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

    # Fetch existing namespaces
    existing_namespaces = set(pinecone_index.describe_index_stats()["namespaces"].keys())
    unified_vectors = []
    unified_namespace = "knowledge_base"

    # Process each file
    for file_path in files:
        filename = os.path.basename(file_path)
        sanitized_filename = sanitize_filename(filename)
        namespace = f"knowledge_base_{sanitized_filename}"

        # Extract court decision date
        court_decision_date = extract_date_from_filename(filename)
        if not court_decision_date:
            logging.warning(f"Skipping file {filename} due to missing or invalid date.")
            continue

        # Skip if already indexed
        if namespace in existing_namespaces:
            logging.info(f"File {filename} is already indexed in Pinecone. Skipping.")
            continue

        logging.info(f"Processing file: {filename} in namespace: {namespace}")

        # Extract text from the PDF
        text_data = extract_text_from_pdf(file_path)
        if not text_data:
            logging.warning(f"No text extracted from {filename}. Skipping.")
            continue

        # Chunk and process text
        text_chunks = chunk_text(text_data, max_tokens=500)

        for chunk in text_chunks:
            chunk_vectors = []
            for page_number, page_text in chunk:
                embeddings = query_ollama_embeddings([page_text])
                if embeddings:
                    vector_id = f"{sanitized_filename}_page_{page_number}"
                    url = filename_to_url.get(filename, "")  # Default to an empty string if no URL is found
                    vector = {
                        "id": vector_id,
                        "values": embeddings[0],
                        "metadata": {
                            "document_id": vector_id,
                            "filename": sanitized_filename,
                            "page_number": page_number,
                            "paragraph": page_text,
                            "source_type": "knowledge_base",
                            "namespace": namespace,
                            "court_decision_date": court_decision_date,
                            "url": url if url else "",  # Ensure the 'url' is a valid string or an empty string
                        },
                    }
                    chunk_vectors.append(vector)
                    unified_vectors.append(vector)
            # Add to unified namespace

            # Upsert individual chunks to Pinecone
            if chunk_vectors:
                upsert_embeddings_to_pinecone(chunk_vectors, namespace)

    # Upsert all documents to unified namespace
    if unified_vectors:
        for i in range(0, len(unified_vectors), 100):  # Split into batches of 100
            batch = unified_vectors[i : i + 100]
            upsert_embeddings_to_pinecone(batch, unified_namespace)

        logging.info(f"Unified namespace '{unified_namespace}' updated with all documents.")

# Run the script
if __name__ == "__main__":
    process_knowledge_base_docs()
