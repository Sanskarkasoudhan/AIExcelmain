import os
import streamlit as st
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize LlamaParse
parser = LlamaParse(
    api_key=os.getenv("LAMA_API_KEY"),  
    parsing_instruction="Extract all sheet names and their contents in a structured format",
    result_type="markdown"
)
file_extractor = {".xlsx": parser}

# Qdrant Config
QDRANT_URL = os.getenv('QDRANT_URL') 
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

# Load Qdrant Client
qdrantClient = QdrantClient(
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY
)

vector_store = QdrantVectorStore(client=qdrantClient, collection_name="Misc_data")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# LLM Config
llm = Gemini(model="models/gemini-1.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))  
Settings.llm = llm
Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en")

# Streamlit UI
st.set_page_config(page_title="Excel AI Assistant", layout="wide")
st.title("📊 Excel AI Assistant")
st.subheader("Upload an Excel file and ask questions about its data.")

# Upload file
uploaded_file = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    with open("temp.xlsx", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    # Load and process data
    documents = SimpleDirectoryReader(input_files=['temp.xlsx'], file_extractor=file_extractor).load_data()
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = db_index.as_query_engine()

    # Query input
    query = st.text_input("Ask a question about the dataset:")
    if st.button("Get Response") and query:
        with st.spinner("Processing..."):
            response = query_engine.query(query).response
            st.success("Response:")
            st.write(response)

st.markdown("---")
st.markdown("🤖 **Built with LlamaIndex, Gemini AI & Qdrant**")
