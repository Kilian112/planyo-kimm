import os
import re
import glob
import time
import json
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename
import logging
from dotenv import load_dotenv
from datetime import datetime
import PyPDF2
import docx
import requests
import redis
import secrets
import hashlib
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from flask import session
from collections import OrderedDict
from ollama_client import OllamaClient
import openai

# Load environment variables from .env file
load_dotenv()

# Initialize Flask App
app = Flask(__name__)

# CORS Configuration
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:4200",
            "http://127.0.0.1:5500",
            "https://uni-bielefeld.de",
            "https://hci-website-haw-hamburg-70c49395aec5266f13540c340688b0e3e2b2efe.pages.ub.uni-bielefeld.de",
            "http://164.92.195.181",
            "https://164.92.195.181",
            "https://hci-website-mieter-ea9c6f.pages.ub.uni-bielefeld.de"
        ],
        "supports_credentials": True,
        "methods": ["POST", "GET", "OPTIONS"],
        "allow_headers": ["Content-Type", "x-api-key", "Authorization"]
    }
})

app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(16))

# Set file upload size limit to 100MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Retrieve Pinecone configuration from environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_env = os.getenv("PINECONE_ENV")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)

# Check if index exists, otherwise create it
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region=pinecone_env)
    )

# Get the index instance
pinecone_index = pc.Index(pinecone_index_name)

# Initialize Redis
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB_INDEX", 0)),
    password=os.getenv("REDIS_PASSWORD"),
    decode_responses=True
)

# Initialize Ollama Client
ollama_client = OllamaClient()
ollama_llm = "fzkun/deepseek-r1-medical:8b" # or other deployed llm
ollama_embedding_model = "kronos483/MedEmbed-large-v0.1:latest" # or other deployed llm

# Initialize OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Main route
@app.route('/')
def home():
    return render_template("index.html")

# Function to clean text
def clean_text(text):
    try:
        return text.encode("utf-8", "ignore").decode("utf-8")
    except Exception as e:
        logging.error(f"Error cleaning text: {e}")
        return text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = [(i + 1, clean_text(page.extract_text().strip())) for i, page in enumerate(reader.pages) if page.extract_text()]
        logging.debug(f"Extracted text from PDF: {text[:500]}...")
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {e}")
        return None

# Function to extract text from DOCX
def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = [p.text for p in doc.paragraphs if p.text.strip()]
        return [(i + 1, paragraph) for i, paragraph in enumerate(text)]
    except Exception as e:
        logging.error(f"Failed to extract text from DOCX: {e}")
        return None

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
    # try:
    #     response = openai.Embedding.create(input=texts, model="text-embedding-3-large")
    #     embeddings = [record["embedding"] for record in response['data']]
    #     return embeddings
    # except Exception as e:
    #     logging.error(f"Exception during OpenAI API request: {e}")
    #     return None

# Function to store document metadata in Redis
def store_document_metadata(document_id, filename, paragraph_number, paragraph, source_type, namespace):
    metadata_key = f"metadata:{document_id}"
    redis_client.hmset(metadata_key, {
        "document_id": document_id,
        "filename": filename,
        "paragraph_number": paragraph_number,  # Changed from page_number, to paragraph number for the URLs only
        "paragraph": paragraph,
        "source_type": source_type,
        "namespace": namespace,
        "uploaded_at": datetime.utcnow().isoformat()
    })

# Function to store embeddings in Pinecone
def store_embeddings_in_pinecone(vectors, namespace):
    logging.debug(f"===================================={vectors}...")
    try:
        pinecone_index.upsert(vectors=vectors, namespace=namespace)
        logging.debug(f"Stored {len(vectors)} vectors in Pinecone with namespace '{namespace}'")
    except Exception as e:
        logging.error(f"Error storing vectors in Pinecone: {e}")

# Function to store conversation history in Redis
def store_conversation_in_redis(session_id, prompt, response):
    """Store the conversation history in Redis."""
    conversation_key = f"conversation:{session_id}"
    conversation_history = redis_client.get(conversation_key) or ""
    updated_history = f"{conversation_history}\nUser: {prompt}\nAssistant: {response}"
    redis_client.set(conversation_key, updated_history, ex=172800)

# Function to retrieve conversation history from Redis
def get_conversation_history(session_id):
    """Retrieve the conversation history from Redis."""
    conversation_key = f"conversation:{session_id}"
    return redis_client.get(conversation_key) or ""

# Function to format response with inline metadata
def format_response_with_metadata(text, metadata_list):
    """
    Append inline metadata references to paragraphs in the text response.

    Args:
        text (str): The generated AI response text.
        metadata_list (list): List of metadata dictionaries for each relevant match.

    Returns:
        str: Text with inline metadata references embedded.
    """
    paragraphs = text.split("\n\n")
    formatted_paragraphs = []

    for idx, paragraph in enumerate(paragraphs):
        if idx < len(metadata_list):
            metadata = metadata_list[idx].get('metadata', {})
            filename = metadata.get('filename', 'Unknown File')
            page_number = metadata.get('page_number', 'Unknown Page')
            url = metadata.get('url', '#')  # Use '#' as a fallback if URL is missing

            # Create a clickable reference with the filename and page number
            ref_tag = f'<a href="{url}" target="_blank">[{filename}, Page {page_number}]</a>'
            formatted_paragraph = f"{paragraph} {ref_tag}"
        else:
            formatted_paragraph = paragraph  # No metadata for this paragraph
        formatted_paragraphs.append(formatted_paragraph)

    return "\n\n".join(formatted_paragraphs)


# Function to query Ollama API for chat completions to include history
def query_ollama_api(prompt, context, metadata, history=None):
    """Query Ollama API for chat completions with contextual memory and inline metadata."""
    try:
        # Combine conversation history with the current context
        if history:
            combined_context = f"{history}\n\nContext: {context}\nQuestion: {prompt}"
        else:
            combined_context = f"Context: {context}\nQuestion: {prompt}"


        generated_text = ollama_client.send_prompt(
            combined_context,
            ollama_llm,
            "You are a helpful assistant. Provide inline metadata references."
        )

        # Format the response with inline metadata
        formatted_text = format_response_with_metadata(generated_text, metadata)

        return formatted_text

    except Exception as e:
        logging.error(f"Exception during Ollama API request: {e}")
        return f"Error: Exception occurred during API request ({str(e)})"


# Route for URL upload (Sanitize filenames and return titles before uploading URL).
def sanitize_title(url):
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')  # Remove 'www.'
    path = parsed.path.strip('/').replace('/', '_') or 'index'  # Default to 'index' if no path
    title = f"{domain}_{path[:20]}"  # Limit path length
    # Remove accents, non-ASCII, and unsupported characters
    title = re.sub(r'[^\x00-\x7F]+', '', title)  # Remove non-ASCII
    title = re.sub(r'[^a-zA-Z0-9_]', '_', title)  # Replace unsupported with underscores
    return title




@app.route('/upload-url', methods=['POST'])
def upload_url():
    url = request.json.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip()]
        text_data = [(i + 1, p) for i, p in enumerate(paragraphs)]
        title = sanitize_title(url)
        namespace = f"url_{hashlib.sha256(url.encode()).hexdigest()[:10]}"
        vectors = []
        for paragraph_number, paragraph in text_data:
            embeddings = query_ollama_embeddings([paragraph])
            if embeddings:
                vector_id = f"{namespace}_para_{paragraph_number}"
                metadata = {
                    "document_id": vector_id,
                    "filename": title,
                    "paragraph_number": paragraph_number,  # Changed from page_number, to paragraph number for the URLs only
                    "paragraph": paragraph,
                    "source_type": "url",
                    "namespace": namespace,
                    "url": url
                }
                vector = {"id": vector_id, "values": embeddings[0], "metadata": metadata}
                vectors.append(vector)
                store_document_metadata(vector_id, title, paragraph_number, paragraph, "url", namespace)
        store_embeddings_in_pinecone(vectors, namespace=namespace)
        redis_client.hset("url_namespaces", namespace, title)
        return jsonify({"message": "URL uploaded successfully!", "namespace": namespace, "title": title, "url": url}), 200
    except Exception as e:
        logging.error(f"Error processing URL {url}: {e}")
        return jsonify({"error": "Failed to process URL"}), 500


# Route for document upload
@app.route('/upload-document', methods=['POST'])
def upload_document():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file part"}), 400

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext not in ['.pdf', '.docx']:
        return jsonify({"error": "Unsupported file format"}), 400

    text_data = None
    if file_ext == '.pdf':
        text_data = extract_text_from_pdf(file)
    elif file_ext == '.docx':
        text_data = extract_text_from_docx(file)

    if not text_data:
        return jsonify({"error": f"Failed to extract text from {filename}"}), 500

    vectors = []
    for page_number, page_text in text_data:
        embeddings = query_ollama_embeddings([page_text])
        if embeddings:
            vector_id = f"{filename}_page_{page_number}"
            metadata = {
                "document_id": vector_id,
                "filename": filename,
                "page_number": page_number,
                "paragraph": page_text,
                "source_type": "user_upload",
                "namespace": filename
            }
            vector = {"id": vector_id, "values": embeddings[0], "metadata": metadata}
            vectors.append(vector)
            store_document_metadata(vector_id, filename, page_number, page_text, "user_upload", filename)

    store_embeddings_in_pinecone(vectors, namespace=filename)

    return jsonify({"message": "Document uploaded and indexed successfully!", "namespace": filename}), 200


# Route for retrieving namespaces
@app.route('/get_namespaces', methods=['GET'])
def get_namespaces():
    try:
        stats = pinecone_index.describe_index_stats()
        active_namespaces = {ns for ns, info in stats["namespaces"].items() if info["vector_count"] > 0}
        # Only "knowledge_base" appears under Knowledge Base
        knowledge_base_files = ["knowledge_base"] if "knowledge_base" in active_namespaces else []
        # User uploads exclude anything with "knowledge_base" in the name
        user_uploads = [ns for ns in active_namespaces if "knowledge_base" not in ns]
        url_namespaces = redis_client.hgetall("url_namespaces")
        user_uploads_display = [
            {"namespace": ns, "display": url_namespaces.get(ns, ns)}
            for ns in user_uploads
        ]
        return jsonify({
            "knowledge_base_files": knowledge_base_files,
            "user_uploads": user_uploads_display
        }), 200
    except Exception as e:
        logging.error(f"Error fetching namespaces: {e}")
        return jsonify({"error": "Failed to retrieve namespaces."}), 500
    

# Route for querying documents
@app.route('/query_stream', methods=['GET'])
def query_stream():
    query_text = request.args.get('query', '')
    namespace = request.args.get('namespace', '').strip()
    logging.debug(f"Querying with namespace: {namespace}")
    query_embeddings = query_ollama_embeddings([query_text])
    query_response = pinecone_index.query(
        vector=query_embeddings[0],
        top_k=10,
        include_metadata=True,
        namespace=namespace
    )
    logging.debug(f"Pinecone matches: {query_response.get('matches', [])}")
    if not query_response.get("matches"):
        return Response("data: No relevant documents found.\n\n", content_type='text/event-stream')

  
    if not query_response.get("matches"):
        return Response("data: No relevant documents found.\n\n", content_type='text/event-stream')

    # Parse and sort matches by court decision date (descending order)
    def parse_date(metadata):
        raw_date = metadata.get('court_decision_date', '1970-01-01')
        clean_date = raw_date.split("T")[0]  # Remove T00:00:00 if present
        return datetime.strptime(clean_date, '%Y-%m-%d')

    sorted_matches = sorted(
        query_response['matches'],
        key=lambda x: parse_date(x['metadata']),
        reverse=True
    )

    # Build the context from matching documents
    context = "\n\n".join([match['metadata']['paragraph'] for match in sorted_matches if 'paragraph' in match['metadata']])

    # Get session ID and retrieve conversation history
    session_id = session.get('session_id', secrets.token_hex(16))
    session['session_id'] = session_id
    conversation_history = get_conversation_history(session_id)

    # Retrieve the AI-generated response based on context and history
    response_text = query_ollama_api(query_text, context, sorted_matches, history=conversation_history)

    # Store the conversation in Redis
    store_conversation_in_redis(session_id, query_text, response_text)

    # Generator function for streaming data to client
    def generate():
        for line in response_text.split("\n"):
            yield f"data: {line}\n\n"
            time.sleep(0.01)
        yield "data: <br><b>Metadata [or Reference sources]:</b><br><hr>\n"
        for idx, match in enumerate(sorted_matches):
            metadata = match.get('metadata', {})
            filename = metadata.get('filename', 'N/A')
            source_type = metadata.get('source_type', 'unknown')
            court_decision_date = metadata.get('court_decision_date', 'Unknown')
            if source_type == "url":
                paragraph_number = metadata.get('paragraph_number', 'N/A')
                metadata_output = (
                    f'{idx+1}. Filename: <a href="{metadata.get("url", "#")}" target="_blank">{filename}</a>,<br>'
                    f'Paragraph: {paragraph_number},<br>'
                    f'Court Decision Date: {court_decision_date}<br>\n'
                )
            else:
                page_number = metadata.get('page_number', 'N/A')
                metadata_output = (
                    f'{idx+1}. Filename: <a href="{metadata.get("url", "#")}" target="_blank">{filename}</a>,<br>'
                    f'Page: {page_number},<br>'
                    f'Court Decision Date: {court_decision_date}<br>\n'
                )
            yield f"data: {metadata_output}\n\n"
            time.sleep(0.01)
        yield "data: [DONE]\n\n"
    return Response(stream_with_context(generate()), content_type='text/event-stream', headers={'Connection': 'keep-alive'})



# Function to initialize Pinecone with all documents in the knowledge base
def initialize_knowledge_base():
    namespace = "knowledge_base"  # Unified namespace
    try:
        existing_namespaces = set(pinecone_index.describe_index_stats()["namespaces"].keys())
        if namespace in existing_namespaces:
            logging.info("knowledge_base already indexed. Skipping.")
            return
    except Exception as e:
        logging.error(f"Error checking namespaces: {e}")
        return

    knowledge_base_docs = glob.glob("./knowledge_base_docs/*.pdf")
    if not knowledge_base_docs:
        logging.warning("No PDF files found in 'knowledge_base_docs'.")
        return

    vectors = []
    for doc_path in knowledge_base_docs:
        filename = os.path.basename(doc_path)
        try:
            with open(doc_path, "rb") as file:
                text_data = extract_text_from_pdf(file)
            if not text_data:
                continue
            for page_number, page_text in text_data:
                embeddings = query_ollama_embeddings([page_text])
                if embeddings:
                    vector_id = f"{filename}_page_{page_number}"
                    metadata = {
                        "document_id": vector_id,
                        "filename": filename,
                        "page_number": page_number,
                        "paragraph": page_text,
                        "source_type": "knowledge_base",
                        "namespace": namespace
                    }
                    vectors.append({"id": vector_id, "values": embeddings[0], "metadata": metadata})
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
    if vectors:
        store_embeddings_in_pinecone(vectors, namespace)
        logging.info("knowledge_base initialized.")


# Decorator for API Key Authentication
def require_api_key(f):
    from functools import wraps
    from flask import abort

    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        if api_key != os.getenv("API_KEY"):
            abort(401)
        return f(*args, **kwargs)
    return decorated_function

# API query endpoint with metadata
# First allow for handling preflight requests for /api/query
@app.route('/api/query', methods=['OPTIONS'])
def handle_preflight():
    origin = request.headers.get('Origin', '')

    allowed_origins = [
        "http://localhost:4200",
        "http://127.0.0.1:5500",
        "https://uni-bielefeld.de",
        "https://hci-website-haw-hamburg-70c49395aec5266f13540c340688b0e3e2b2efe.pages.ub.uni-bielefeld.de",
        "http://164.92.195.181",
        "https://hci-website-mieter-ea9c6f.pages.ub.uni-bielefeld.de"
    ]

    if origin == "null":
        print("⚠️ Warning: Request had a 'null' origin. Allowing it.")
        origin = "http://164.92.195.181"  # Assign a fallback origin

    if origin in allowed_origins:
        print(f"✅ Preflight Allowed: {origin}")
        response = jsonify({"message": "Preflight request accepted."})
        response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Credentials", "true")
        response.headers.add("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, x-api-key, Authorization")
        return response, 200
    else:
        print(f"❌ Preflight Denied: {origin}")
        return jsonify({"error": "Origin not allowed"}), 403

# Clear session at the beginning of the request to avoid stale responses
@app.route('/api/query', methods=['POST'])
@require_api_key
def api_query():
    """Handle API queries with structured JSON response as per client specifications."""
    try:
        # Allow requests without Origin header (e.g. from curl, Postman)
        # 🔍 Debugging: Print the received origin
        print(f"🔍 Received Origin: {request.headers.get('Origin')}")

        origin = request.headers.get('Origin', 'http://localhost:4200')  # Default to localhost if missing
        allowed_origins = [
            "http://localhost:4200",
            "http://127.0.0.1:5500",
            "https://uni-bielefeld.de",
            "https://hci-website-haw-hamburg-70c49395aec5266f13540c340688b0e3e2b2efe.pages.ub.uni-bielefeld.de",
            "http://164.92.195.181",
            "https://164.92.195.181", 
            "https://hci-website-mieter-ea9c6f.pages.ub.uni-bielefeld.de"
        ]

        # Check if origin is allowed
        # 🔍 Debugging: Check if Flask is recognizing the origin correctly
        if origin and origin not in allowed_origins:
            print(f"❌ Origin {origin} is NOT allowed!")  # Print in logs
            return jsonify({"error": "Origin not allowed"}), 403
        else:
            print(f"✅ Origin {origin} is allowed!")

        # Reset session to avoid stale responses
        session.clear()

        # Parse the incoming request
        data = request.json
        query_text = data.get('query')
        namespace = data.get('namespace')

        # Validate required inputs
        if not query_text or not namespace:
            return jsonify({"error": "Missing 'query' or 'namespace' in request"}), 400

        # Ensure a consistent session ID for the user
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
        session_id = session['session_id']

        # Retrieve conversation history from Redis
        conversation_history_key = f"conversation:{session_id}"
        conversation_history = redis_client.get(conversation_history_key)
        if conversation_history:
            conversation_history = json.loads(conversation_history)
        else:
            conversation_history = []

        # Add the current user query to the conversation history
        conversation_history.append({"role": "user", "content": query_text})

        # Generate embeddings for the query
        query_embeddings = query_ollama_embeddings([query_text])

        # Query Pinecone for the most relevant context
        query_response = pinecone_index.query(
            vector=query_embeddings[0],
            top_k=10,
            include_metadata=True,
            namespace=namespace
        )

        # Validate that matches are found
        if not query_response.get("matches"):
            return jsonify({"response": "No relevant documents found."}), 404

        # Extract matched context paragraphs
        document_context = "\n\n".join([
            match['metadata'].get('paragraph', '') for match in query_response['matches']
        ])

        # Query the AI model with structured conversation history
        raw_response_text = query_ollama_api(
            prompt=query_text,
            context=document_context,
            metadata=query_response['matches'],
            history=conversation_history  
        )

        # Clean response text by safely removing embedded URLs without truncation
        clean_text = re.sub(r'<a href=".*?">.*?</a>', '', raw_response_text).strip()

        # Extract metadata from the first match
        first_match = query_response['matches'][0]['metadata'] if query_response['matches'] else {}
        anchor_url = first_match.get('url', '')
        anchor_label = f"{first_match.get('filename', 'Unknown File')}, Page {first_match.get('page_number', 'Unknown Page')}"

        # Structuring AI response exactly as requested by the client
        ai_response = {
            "text": clean_text,  
            "anchor_url": anchor_url,
            "anchor_label": anchor_label
        }

        # Structuring metadata exactly as requested by the client
        metadata_details = [
            {
                "file_name": match['metadata'].get('filename', 'N/A'),
                "page_number": match['metadata'].get('page_number', 'N/A'),
                "court_decision_date": match['metadata'].get('court_decision_date', 'Unknown')
            }
            for match in query_response['matches']
        ]

        # Ensure the JSON fields order using collections.OrderedDict
        ordered_response = OrderedDict([
            ("AI Response", OrderedDict([
                ("text", ai_response["text"]),
                ("anchor_url", ai_response["anchor_url"]),
                ("anchor_label", ai_response["anchor_label"])
            ])),
            ("Metadata [or Reference sources]", [
                OrderedDict([
                    ("file_name", item["file_name"]),
                    ("page_number", item["page_number"]),
                    ("court_decision_date", item["court_decision_date"])
                ]) for item in metadata_details
            ])
        ])

        # Store the conversation in Redis
        conversation_history.append({"role": "assistant", "content": clean_text})
        redis_client.set(conversation_history_key, json.dumps(conversation_history))

        # Return the response as structured JSON with UTF-8 encoding to prevent truncation
        return app.response_class(
            response=json.dumps(ordered_response, ensure_ascii=False, indent=4),
            mimetype="application/json; charset=utf-8"
        )

    except Exception as e:
        print(f"🔥 Exception: {str(e)}")  # Log the actual exception
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

# Add a health check route to (/health) to monitor if the app is responsive
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

# Run the app
if __name__ == '__main__':
    initialize_knowledge_base()
    logging.basicConfig(level=logging.DEBUG)
    
    # Detect which port to run on, defaulting to 8001
    port = int(os.getenv("APP_PORT", 8001))
    app.run(host='0.0.0.0', port=port, debug=True)