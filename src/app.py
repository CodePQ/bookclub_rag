import streamlit as st
import os
import ollama
from ingest import load_pdf, chunk_text
from embed import embed_chunks, get_embedding
from database import get_pinecone_index, upsert_to_pinecone, get_all_book_titles
from query import retrieve_context, generate_answer, query_rag

st.set_page_config(page_title="BookHub RAG", page_icon="📚", layout="wide")
st.title("📚 BookHub: Your AI Library")

# Sidebar for Uploading
with st.sidebar:
    st.header("Service Status")
    # Quick health checks
    try:
        ollama.list()
        st.success("✅ Ollama: Running")
    except:
        st.error("❌ Ollama: Not Found")
        
    index = get_pinecone_index()
    if index:
        st.success("✅ Pinecone: Connected")
    else:
        st.error("❌ Pinecone: Error")

    st.divider()
    st.header("Add to Library")
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")
    book_title = st.text_input("Enter Book Title")
    
    if st.button("Ingest Book") and uploaded_file and book_title:
        # Check services before ingesting
        if not index:
            st.error("Pinecone connection lost. Check your .env.")
        else:
            with st.spinner(f"Ingesting '{book_title}'..."):
                # Save temporary file
                if not os.path.exists("data"):
                    os.makedirs("data")
                temp_path = f"data/{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Run RAG Pipeline
                text = load_pdf(temp_path)
                if text:
                    chunks = chunk_text(text)
                    embeddings = embed_chunks(chunks)
                    if not embeddings:
                        st.error("Failed to generate embeddings. Is Ollama running?")
                    else:
                        upsert_to_pinecone(index, chunks, embeddings, book_title)
                        st.success(f"'{book_title}' added to your library!")
                        st.rerun() # Refresh to show new book in selectbox
                else:
                    st.error("Failed to extract text from PDF.")

# Main Chat Interface
index = get_pinecone_index()
if index:
    books = get_all_book_titles(index)
    selected_book = st.selectbox("Select a book to chat with:", ["All Books"] + books)
    
    # Simple Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the book..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use our new RAG logic with MEMORY (passing chat_history)
                filter_title = selected_book if selected_book != "All Books" else None
                
                # query_rag now returns BOTH the answer and the context chunks
                response, context = query_rag(
                    prompt, 
                    book_title=filter_title, 
                    chat_history=st.session_state.messages
                )
                
                if response:
                    st.markdown(response)
                    
                    # Add Source Citations below the answer
                    with st.expander("🔍 Show Source Evidence"):
                        for i, chunk in enumerate(context):
                            st.info(f"Chunk {i+1}:\n\n{chunk}")
                            
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    st.markdown("I couldn't find any relevant information.")
else:
    st.error("Could not connect to Pinecone. Please check your .env file.")
