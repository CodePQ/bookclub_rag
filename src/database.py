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

def upsert_to_pinecone(index, chunks, embeddings, book_title, namespace="default"):
    """
    Uploads vectors and their original text (metadata) to Pinecone.
    We now include the 'title' of the book so we can filter by it later.
    """
    vectors_to_upsert = []
    
    # Create a safe ID prefix using the book title
    safe_title = book_title.replace(" ", "_").lower()
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Unique ID combining book title and chunk number
        vector_id = f"{safe_title}_chunk_{i}"
        
        # We store the original text AND the book title in metadata
        vectors_to_upsert.append({
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "text": chunk,
                "title": book_title
            }
        })
    
    # Send the data to Pinecone
    index.upsert(vectors=vectors_to_upsert, namespace=namespace)
    print(f"Successfully uploaded {len(vectors_to_upsert)} vectors for '{book_title}' to namespace '{namespace}'.")

def get_all_book_titles(index, namespace="default"):
    """
    Returns a list of all unique book titles found in the index metadata.
    Dynamically detects index dimension to avoid errors.
    """
    try:
        # Get index stats to find the dimension
        stats = index.describe_index_stats()
        dimension = stats.get('dimension', 768) # Fallback to 768 if needed
        
        # Query with a dummy vector of the CORRECT dimension
        results = index.query(
            vector=[0.0] * dimension, 
            top_k=1000, 
            include_metadata=True,
            namespace=namespace
        )
        
        titles = set()
        for match in results["matches"]:
            if "metadata" in match and "title" in match["metadata"]:
                titles.add(match["metadata"]["title"])
                
        return sorted(list(titles))
    except Exception as e:
        print(f"Error fetching book titles from Pinecone: {e}")
        return []

if __name__ == "__main__":
    # Test the connection and library list
    print("Testing connection to Pinecone...")
    index = get_pinecone_index()
    if index:
        stats = index.describe_index_stats()
        print(f"Connected! Index stats: {stats}")
        
        # Test the library list
        titles = get_all_book_titles(index)
        print(f"Books in your library: {titles}")

