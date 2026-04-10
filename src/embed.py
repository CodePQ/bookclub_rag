import ollama

def get_embedding(text, model="nomic-embed-text"):
    """
    Takes a string of text and returns its vector representation (embedding).
    """
    try:
        response = ollama.embeddings(model=model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"ERROR: Ollama failed to generate embedding for text: '{text[:50]}...'")
        print(f"Details: {e}")
        return None

def embed_chunks(chunks, model="nomic-embed-text"):
    """
    Takes a list of text chunks and returns a list of embeddings.
    """
    embeddings = []
    print(f"Generating embeddings for {len(chunks)} chunks using {model}...")
    
    for i, chunk in enumerate(chunks):
        vector = get_embedding(chunk, model)
        if vector:
            embeddings.append(vector)
            # Show progress every 10 chunks
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"Completed {i + 1}/{len(chunks)}")
        else:
            print(f"CRITICAL: Failed to embed chunk {i}. Pipeline may be incomplete.")
        
    return embeddings

if __name__ == "__main__":
    # Test with a simple string
    test_text = "Sherlock Holmes is a great detective."
    vector = get_embedding(test_text)
    if vector:
        print(f"Successfully generated vector of length {len(vector)}")
        print(f"First 5 numbers: {vector[:5]}")
