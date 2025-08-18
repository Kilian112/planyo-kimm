import os
from dotenv import load_dotenv
import requests
from pinecone import Pinecone
from ollama_client import OllamaClient

# Ollama setup
ollama_client = OllamaClient()
ollama_embedding_model = "embedding_model_name"
ollama_llm = "llm_model_name"

# Load environment variables from .env file
load_dotenv()

# Set up API keys and environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Check if all required environment variables are set
if not pinecone_api_key:
    raise ValueError("Pinecone API key not set. Please set the PINECONE_API_KEY environment variable.")

if not pinecone_env:
    raise ValueError("Pinecone environment not set. Please set the PINECONE_ENV environment variable.")

if not pinecone_index_name:
    raise ValueError("Pinecone index name not set. Please set the PINECONE_INDEX_NAME environment variable.")

# Initialize Pinecone client
pinecone = Pinecone(api_key=pinecone_api_key)

# Step 1: Retrieve Namespaces
print("Retrieving namespaces from Pinecone...")

try:
    index = pinecone.Index(pinecone_index_name)
    namespaces = index.describe_index_stats()["namespaces"]
    print("Namespaces available in Pinecone:")
    for ns in namespaces.keys():
        print(f" - {ns}")
except Exception as e:
    print("An error occurred while retrieving namespaces:", e)
    exit()  # Exit if namespaces cannot be retrieved

# Step 2: Prompt user to select a namespace and query it
namespace_to_test = input("Enter the namespace you want to query (as listed above): ")
query_text = input("Enter the query text: ")

# Generate embeddings for the query
print(f"\nGenerating embeddings for query: '{query_text}'")

try:
    query_embedding = ollama_client.embed(query_text, ollama_embedding_model)
    print("Query embedding generated successfully.")
except Exception as e:
    print("Error generating query embedding:", e)
    exit()

# Query Pinecone with the generated embedding
print(f"\nQuerying Pinecone in namespace '{namespace_to_test}'...")

try:
    query_response = index.query(vector=query_embedding, top_k=5, include_metadata=True, namespace=namespace_to_test)
    print("Query Response:")
    
    # Extract the paragraphs from the query response
    context = "\n\n".join(match["metadata"]["paragraph"] for match in query_response["matches"])

    if not context:
        print("No relevant documents found.")
    else:
        print("Context extracted from Pinecone query response:")
        print(context)

        # Step 3: Send context to Ollama for inference
        print("\nSending context to Ollama for inference...")
        
        try:
            inference_result = ollama_client.send_prompt(
                f"Context: {context}\nQuestion: {query_text}",
                ollama_llm,
                "You are a helpful assistant.")

        except Exception as e:
            print("An error occurred while fetching inference from Ollama:", e)

except Exception as e:
    print("Error querying Pinecone:", e)
