import streamlit as st
import google.generativeai as genai
import time
import os
from backend import ProductionRAGPipeline

st.set_page_config(page_title="Enterprise RAG System", layout="centered")

@st.cache_resource
def load_backend():
    return ProductionRAGPipeline()

rag_backend = load_backend()

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password", help="Get this from Google AI Studio")
    
    st.divider()
    st.header("📄 Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDFs (≥200 pages)", type="pdf", accept_multiple_files=True)
    
    if st.button("Run Ingestion Pipeline"):
        if not uploaded_files:
            st.error("Please upload files first.")
        else:
            with st.spinner("Extracting, Chunking, and Embedding..."):
                for file in uploaded_files:
                    file_path = f"./{file.name}"
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                    rag_backend.ingest_pdf(file_path)
                st.success("✅ Vectors Synced to Qdrant!")

# --- Main Chat UI ---
st.title("📚 Private Knowledge RAG")
st.caption("Hybrid Search (HNSW ANN + Cross-Encoder) with Google Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Query the database..."):
    if not api_key:
        st.error("⚠️ API Key required. Please configure it in the sidebar.")
        st.stop()
        
    genai.configure(api_key=api_key)
    generation_model = genai.GenerativeModel('gemini-2.5-flash')

    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Executing Vector Search & Semantic Reranking..."):
            retrieval_data = rag_backend.query_pipeline(user_query)
            context = retrieval_data["context_window"]
            sources = retrieval_data["sources"]
            search_latency = retrieval_data["latency_seconds"]

            if not sources:
                response_text = "No relevant context found in the database."
            else:
                prompt = f"""
                You are a highly accurate analytical assistant. Use ONLY the provided context to answer the user's question. 
                If the answer is not contained in the context, say "I don't have enough information."
                Question: {user_query}
                Context Documents: {context}
                """
                
                start_gen = time.time()
                llm_response = generation_model.generate_content(prompt)
                gen_latency = time.time() - start_gen
                response_text = llm_response.text

            total_time = round(search_latency + gen_latency, 2)
            st.markdown(response_text)
            
            if sources:
                st.divider()
                st.caption(f"⏱️ **Pipeline Latency:** {total_time}s (Retrieval: {search_latency}s | LLM: {round(gen_latency, 2)}s)")
                st.caption("**Verified Citations:**")
                # Deduplicate sources for clean UI
                unique_sources = {f"{src['filename']} (Page {src['page']})" for src in sources}
                for src in unique_sources:
                    st.caption(f"- 📄 `{src}`")

    st.session_state.messages.append({"role": "assistant", "content": response_text})
