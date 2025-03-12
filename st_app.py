import os
import faiss
import streamlit as st
import nest_asyncio
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_parse import LlamaParse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
nest_asyncio.apply()

# Set Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.title("📊 AI-Powered Data Query Engine")
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload an XLSX file", type=["xlsx"])

if uploaded_file:
    st.sidebar.success("File uploaded successfully!")
    file_path = f"./temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and Parse XLSX File
    parser = LlamaParse(
        api_key=os.getenv("LAMA_API_KEY"),
        parsing_instruction="Extract all sheet names and their contents in a structured format",
        result_type="markdown"
    )
    file_extractor = {".xlsx": parser}
    documents = SimpleDirectoryReader(input_files=[file_path], file_extractor=file_extractor).load_data()
    
    # Set LLM (Gemini)
    llm = Gemini(model="models/gemini-1.5-flash", api_key=GOOGLE_API_KEY)
    Settings.llm = llm

    # Set Embedding Model (Gemini)
    embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=GOOGLE_API_KEY)
    Settings.embed_model = embed_model

    # FAISS Vector Store
    dimensions = 768
    faiss_index = faiss.IndexFlatL2(dimensions)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    # Build Index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    index.storage_context.persist()

    # Query Engine
    query_engine = index.as_query_engine(similarity_top_k=5)

    def get_response(query):
        """Fetch AI-generated response for a query."""
        response = query_engine.query(query)
        return response.response

    # User Query Interface
    st.header("💬 Ask Your Query")
    user_query = st.text_input("Enter your query:")

    if st.button("Get Response"):
        if user_query:
            with st.spinner("Processing..."):
                response = get_response(user_query)
            st.success("Response Generated")
            st.write(response)
        else:
            st.warning("Please enter a query.")
