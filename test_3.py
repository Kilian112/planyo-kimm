import os
import requests
import logging
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from ollama_client import OllamaClient

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# API configurations
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# Ollama setup
ollama_client = OllamaClient()
ollama_embedding_model = "embedding_model_name"
ollama_llm = "llm_model_name"

# Function to retrieve namespaces
def get_pinecone_namespaces():
    """Retrieve available namespaces in Pinecone."""
    try:
        index_stats = index.describe_index_stats()
        namespaces = index_stats.get("namespaces", {}).keys()
        return list(namespaces)
    except Exception as e:
        logging.error(f"Error retrieving namespaces: {e}")
        return []

# Function to generate embeddings using OpenAI
def query_ollama_embeddings(text):
    try:
        return ollama_client.embed(text, ollama_embedding_model)
    except Exception as e:
        logging.error(f"Exception during Ollama API request: {e}")
        return None

# Function to query Pinecone
def query_pinecone(embedding, namespace, top_k=5):
    """Query Pinecone for the most relevant documents."""
    try:
        response = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )
        return response.get("matches", [])
    except Exception as e:
        logging.error(f"Error querying Pinecone: {e}")
        return []

# Function to query Ollama API for chat completions to include history
def query_ollama_api(prompt, context, history=None):
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

        return generated_text

    except Exception as e:
        logging.error(f"Exception during Ollama API request: {e}")
        return f"Error: Exception occurred during API request ({str(e)})"


# Main interactive function
def main():
    # Step 1: Retrieve available namespaces
    namespaces = get_pinecone_namespaces()
    if not namespaces:
        print("No namespaces available in Pinecone.")
        return

    # Step 2: Display and let the user select a namespace
    print("Available namespaces:")
    for i, namespace in enumerate(namespaces, start=1):
        print(f"{i}. {namespace}")
    selected_namespace = None

    while not selected_namespace:
        try:
            choice = int(input("Select a namespace by number: ")) - 1
            if 0 <= choice < len(namespaces):
                selected_namespace = namespaces[choice]
                print(f"Selected namespace: {selected_namespace}")
            else:
                print("Invalid choice. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Step 3: Accept a query from the user
    query_text = input("Enter your query: ")
    if not query_text.strip():
        print("Query cannot be empty.")
        return

    # Step 4: Generate embeddings for the query
    embedding = query_ollama_embeddings(query_text)
    if not embedding:
        print("Failed to generate embeddings.")
        return

    # Step 5: Query Pinecone for relevant documents
    matches = query_pinecone(embedding, selected_namespace)
    if not matches:
        print("No relevant documents found in the selected namespace.")
        return

    # Step 6: Extract context for the query
    context = "\n\n".join([
        f"Document {i+1}:\n{match['metadata'].get('paragraph', '')}"
        for i, match in enumerate(matches)
    ])

    print("\nRetrieved Context:\n")
    print(context)

    # Step 7: Query Ollama for inference
    inference = query_ollama_api(context, query_text)
    if inference:
        print("\nGenerated AI Inference:\n")
        print(inference)
    else:
        print("Failed to generate AI inference.")

if __name__ == "__main__":
    main()
