from supabase import create_client, Client
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

def init_supabase():
    """Initialize Supabase client."""
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    return create_client(url, key)

def create_table_if_not_exists(supabase: Client):
    """Check if the pdf_metadata table exists."""
    try:
        # Try to query the table to check if it exists
        supabase.table('pdf_metadata').select('*').limit(1).execute()
    except Exception as e:
        raise Exception(
            "The 'pdf_metadata' table does not exist. Please create it in your Supabase dashboard using the following SQL:\n\n"
            "-- Enable the vector extension\n"
            "CREATE EXTENSION IF NOT EXISTS vector;\n\n"
            "-- Create the table\n"
            "CREATE TABLE pdf_metadata (\n"
            "    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,\n"
            "    pdf_name TEXT,\n"
            "    embedding VECTOR(384),\n"
            "    text_chunk TEXT\n"
            ");"
        ) from e

def store_pdf_metadata(supabase: Client, pdf_name, embeddings, chunks):
    """Store PDF metadata and embeddings in Supabase."""
    create_table_if_not_exists(supabase)
    for i, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
        supabase.table('pdf_metadata').insert({
            'pdf_name': pdf_name,
            'embedding': embedding.tolist(),
            'text_chunk': chunk
        }).execute()

def query_similar_chunks(supabase: Client, query_embedding, pdf_name, limit=5):
    """Query Supabase for similar text chunks based on embedding."""
    # Fallback: fetch embeddings for the given pdf and compute similarities locally
    response = supabase.table('pdf_metadata')\
        .select('text_chunk, embedding')\
        .eq('pdf_name', pdf_name)\
        .execute()

    rows = response.data or []
    if not rows:
        return []

    # Build embedding matrix
    embeddings = []
    for row in rows:
        emb = row.get('embedding')
        # Handle different possible returned formats
        if isinstance(emb, list):
            embeddings.append(np.array(emb, dtype=float))
        elif isinstance(emb, str):
            try:
                import ast
                embeddings.append(np.array(ast.literal_eval(emb), dtype=float))
            except Exception:
                # If parsing fails, skip this row
                continue
        else:
            try:
                embeddings.append(np.array(emb, dtype=float))
            except Exception:
                continue

    if len(embeddings) == 0:
        return []

    emb_matrix = np.vstack(embeddings)
    q = np.array(query_embedding, dtype=float)

    # Compute cosine similarity where possible, fallback to negative L2 distance
    try:
        emb_norms = np.linalg.norm(emb_matrix, axis=1)
        q_norm = np.linalg.norm(q)
        if q_norm == 0 or np.any(emb_norms == 0):
            raise Exception("zero norm")
        sims = (emb_matrix @ q) / (emb_norms * q_norm)
        # higher is better
        top_idx = np.argsort(-sims)[:limit]
    except Exception:
        # fallback to L2 distance (smaller is better)
        dists = np.linalg.norm(emb_matrix - q, axis=1)
        top_idx = np.argsort(dists)[:limit]

    # Map back to rows. Note: embeddings list may be shorter than rows if some rows skipped.
    selected_chunks = []
    for i in top_idx:
        # Find the corresponding row (assume order preserved)
        try:
            selected_chunks.append(rows[i]['text_chunk'])
        except Exception:
            continue

    return selected_chunks