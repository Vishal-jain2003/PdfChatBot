import streamlit as st
import os
from pdf_processor import extract_text_from_pdf, create_vector_store, save_vector_store
from supabase_utils import init_supabase, store_pdf_metadata
from rag_pipeline import answer_question

st.title("PDF QA Chatbot")

# Initialize session state
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = None
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded file
    os.makedirs("uploads", exist_ok=True)
    pdf_path = os.path.join("uploads", uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.pdf_name = uploaded_file.name
    
    # Process PDF
    with st.spinner("Processing PDF..."):
        text = extract_text_from_pdf(pdf_path)
        if text:
            index, chunks, embeddings = create_vector_store(text)
            save_vector_store(index)
            
            # Store in Supabase
            supabase = init_supabase()
            store_pdf_metadata(supabase, uploaded_file.name, embeddings, chunks)
            st.session_state.vector_store_ready = True
            st.success("PDF processed and stored successfully!")
        else:
            st.error("Failed to extract text from PDF.")

# Question input
if st.session_state.vector_store_ready:
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Generating answer..."):
            answer = answer_question(question, st.session_state.pdf_name)
            st.write("**Answer**:")
            st.write(answer)