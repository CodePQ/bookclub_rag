import os
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

def get_pinecone_index(index_name="bookclub-rag"):
    """
    Connects to the Pinecone index using the API key from .env.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("Error: PINECONE_API_KEY not found in .env file.")
        return None
        
    pc = Pinecone(api_key=api_key)
    
    # Check if index exists (optional but good for debugging)
    if index_name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Error: Index '{index_name}' not found. Please create it in the Pinecone dashboard.")
        return None
        
    return pc.Index(index_name)

def upsert_to_pinecone(index, chunks, embeddings, namespace="default"):
    """
    Uploads vectors and their original text (metadata) to Pinecone.
    """
    vectors_to_upsert = []
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # We need a unique ID for every chunk (e.g., chunk_0, chunk_1)
        vector_id = f"chunk_{i}"
        
        # We store the original text in the metadata dictionary
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {"text": chunk}
        })
    
    # Send the data to Pinecone in blocks
    index.upsert(vectors=vectors_to_upsert, namespace=namespace)
    print(f"Successfully uploaded {len(vectors_to_upsert)} vectors to Pinecone namespace '{namespace}'.")

if __name__ == "__main__":
    # Test the connection
    print("Testing connection to Pinecone...")
    index = get_pinecone_index()
    if index:
        stats = index.describe_index_stats()
        print(f"Connected! Index stats: {stats}")
