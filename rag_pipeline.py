from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from supabase_utils import init_supabase, query_similar_chunks
import os
from dotenv import load_dotenv

load_dotenv()

def init_llm():
    """Initialize Google Gemini API (via LangChain)."""
    api_key = os.getenv('GOOGLE_API_KEY')
    return GoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
        temperature=0.7
    )

def get_query_embedding(query):
    """Generate embedding for the query."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode([query])[0]

def answer_question(query, pdf_name):
    """Run the RAG pipeline to answer a question."""
    # Initialize Supabase and LLM
    supabase = init_supabase()
    llm = init_llm()
    
    # Get query embedding
    query_embedding = get_query_embedding(query)
    
    # Retrieve relevant chunks
    relevant_chunks = query_similar_chunks(supabase, query_embedding, pdf_name)
    context = "\n".join(relevant_chunks)
    
    # Define prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Context: {context}\n\nQuestion: {question}\nAnswer:"
    )
    
    # Generate answer
    prompt = prompt_template.format(context=context, question=query)

    # The GoogleGenerativeAI wrapper may expose different call methods depending on
    # the installed langchain/google-genai versions. Try several common call patterns
    # and return the generated text.
    def _call_llm(local_llm, text_prompt: str) -> str:
        # try predict
        if hasattr(local_llm, 'predict'):
            try:
                return local_llm.predict(text_prompt)
            except Exception:
                pass

        # try generate (LangChain LLM API)
        if hasattr(local_llm, 'generate'):
            try:
                result = local_llm.generate([text_prompt])
                # result.generations is typically a list[list[Generation]]
                gens = getattr(result, 'generations', None)
                if gens:
                    return gens[0][0].text
            except Exception:
                pass

        # try __call__ via async/sync helper names
        if hasattr(local_llm, '__call__'):
            try:
                return local_llm.__call__(text_prompt)
            except Exception:
                pass

        # last resort: attempt using invoke
        if hasattr(local_llm, 'invoke'):
            try:
                return local_llm.invoke(text_prompt)
            except Exception:
                pass

        raise RuntimeError('Unable to call the LLM - no supported method found')

    answer = _call_llm(llm, prompt)
    return answer