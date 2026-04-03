import ollama
from embed import get_embedding
from database import get_pinecone_index

def retrieve_context(index, query_vector, top_k=3):
    """
    Finds the most relevant text chunks from Pinecone.
    """
    results = index.query(
        vector=query_vector, 
        top_k=top_k, 
        include_metadata=True,
        namespace="default" # Must match what we used in database.py
    )
    
    # DEBUG: See what Pinecone found
    print(f"Pinecone found {len(results['matches'])} matches.")
    for i, match in enumerate(results["matches"]):
        print(f" - Match {i+1} (Score: {match['score']:.4f})")
    
    # Extract the original text from metadata
    context_chunks = [match["metadata"]["text"] for match in results["matches"] if "text" in match["metadata"]]
    return context_chunks

def generate_answer(question, context_chunks, model="llama3"):
    """
    Uses Ollama to generate an answer basing it ONLY on retrieved context.
    """
    # Join the retrieved chunks into one context block
    context_text = "\n---\n".join(context_chunks)
    
    # Create the 'system' instructions for the LLM
    system_prompt = (
        "You are a helpful reading assistant. Use ONLY the provided context "
        "below to answer the question. If the answer isn't in the context, "
        "simply say you don't know."
    )
    
    user_prompt = f"Question: {question}\n\nContext:\n{context_text}"
    
    # Send to Ollama
    response = ollama.chat(model=model, messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ])
    
    return response['message']['content']

def query_rag(question):
    """
    Complete RAG loop: Embed -> Retrieve -> Generate.
    """
    index = get_pinecone_index()
    if not index:
        return None
        
    print(f"\nAnalyzing question: '{question}'...")
    
    # 1. Embed question
    query_vector = get_embedding(question)
    
    # 2. Retrieve relevant parts of the book
    context = retrieve_context(index, query_vector)
    
    # 3. Generate answer
    print("Synthesizing answer from retrieved chunks...")
    answer = generate_answer(question, context)
    return answer

if __name__ == "__main__":
    print("Welcome to your RAG Book Assistant! (Type 'exit' or 'quit' to stop)")
    
    while True:
        user_question = input("\nAsk a question about the book: ").strip()
        
        if user_question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
            
        if not user_question:
            continue
            
        try:
            ans = query_rag(user_question)
            print("\n--- ANSWER ---")
            print(ans)
            print("--------------")
        except Exception as e:
            print(f"Error during query: {e}")
