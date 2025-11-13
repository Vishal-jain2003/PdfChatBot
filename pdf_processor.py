import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text() or ''
        return text
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ''

def create_vector_store(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks and create a FAISS vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    return index, chunks, embeddings

def save_vector_store(index, filename='faiss_index.bin'):
    """Save FAISS index to disk."""
    faiss.write_index(index, filename)

def load_vector_store(filename='faiss_index.bin'):
    """Load FAISS index from disk."""
    if os.path.exists(filename):
        return faiss.read_index(filename)
    return None