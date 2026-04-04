import ollama
from embed import get_embedding
from database import get_pinecone_index

def retrieve_context(index, query_vector, book_title=None, top_k=5):
    """
    Finds the most relevant text chunks from Pinecone.
    If book_title is provided, it only searches that specific book.
    """
    # Create the filter dictionary if a title is provided
    filter_dict = {"title": book_title} if book_title else None

    results = index.query(
        vector=query_vector, 
        top_k=top_k, 
        include_metadata=True,
        namespace="default",
        filter=filter_dict # Apply the metadata filter!
    )
    
    # DEBUG: See what Pinecone found
    print(f"Pinecone found {len(results['matches'])} matches.")
    for i, match in enumerate(results["matches"]):
        print(f" - Match {i+1} (Score: {match['score']:.4f})")
    
    # Extract the original text from metadata
    context_chunks = [match["metadata"]["text"] for match in results["matches"] if "text" in match["metadata"]]
    return context_chunks

def generate_answer(question, context_chunks, chat_history=None, model="llama3"):
    """
    Uses Ollama to generate an answer. Supports sliding window memory.
    """
    context_text = "\n---\n".join(context_chunks)
    
    # 1. System Prompt
    system_prompt = (
        "You are a helpful reading assistant. Use ONLY the provided context "
        "below to answer the question. If the answer isn't in the context, "
        "simply say you don't know. Be conversational but concise."
    )
    
    # 2. Build the messages list for Ollama
    messages = [{'role': 'system', 'content': system_prompt}]
    
    # 3. Add Chat History (Sliding Window: Last 6 messages / 3 rounds)
    if chat_history:
        messages.extend(chat_history[-6:])
        
    # 4. Add current context and question
    user_input = f"Context:\n{context_text}\n\nQuestion: {question}"
    messages.append({'role': 'user', 'content': user_input})
    
    # Send to Ollama
    response = ollama.chat(model=model, messages=messages)
    return response['message']['content']

def query_rag(question, book_title=None, chat_history=None):
    """
    Complete RAG loop: Embed -> Retrieve -> Generate.
    """
    index = get_pinecone_index()
    if not index:
        return None
        
    # 1. Embed question
    query_vector = get_embedding(question)
    
    # 2. Retrieve relevant parts
    context = retrieve_context(index, query_vector, book_title=book_title)
    
    # 3. Generate answer (passing history)
    answer = generate_answer(question, context, chat_history=chat_history)
    return answer, context # Returning context for citations!

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
