from ingest import load_pdf, chunk_text
from embed import embed_chunks
from database import get_pinecone_index, upsert_to_pinecone

def main():
    # 1. Setup paths and index
    PDF_PATH = "data/the-adventure-of-the-cheap-flat.pdf"
    BOOK_TITLE = "The Adventure of the Cheap Flat"
    INDEX_NAME = "bookclub-rag"
    
    print(f"--- Starting Data Induction for '{BOOK_TITLE}' ---")
    
    # 2. Ingest: Load and Chunk
    text = load_pdf(PDF_PATH)
    if not text:
        return
        
    chunks = chunk_text(text)
    
    # 3. Embed: Generate Vectors via Ollama
    embeddings = embed_chunks(chunks)
    
    # 4. Database: Upload to Pinecone
    index = get_pinecone_index(INDEX_NAME)
    if index:
        upsert_to_pinecone(index, chunks, embeddings, BOOK_TITLE)
        print(f"--- Final Upload Complete for '{BOOK_TITLE}'! ---")

if __name__ == "__main__":
    main()
