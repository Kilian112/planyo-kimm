from dotenv import load_dotenv
from pinecone import Pinecone
import os

# Load environment variables
load_dotenv()
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Delete all namespaces starting with "knowledge_base"
for ns in index.describe_index_stats()["namespaces"]:
    if ns.startswith("knowledge_base"):
        index.delete(delete_all=True, namespace=ns)
        print(f"Deleted namespace: {ns}")

print("All specified namespaces have been deleted.")
