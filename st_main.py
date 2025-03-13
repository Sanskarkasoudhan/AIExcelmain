import os
import streamlit as st
import pandas as pd
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
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
file_extractor = {".xlsx": parser, ".csv": parser}

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

# Streamlit UI Configuration
st.set_page_config(page_title="📊 Excel & CSV AI Assistant", layout="wide")
st.title("📊 Excel & CSV AI Assistant")
st.subheader("Upload your dataset (Excel/CSV), explore the data, and get insights using AI.")

# File Upload
uploaded_files = st.file_uploader("📂 Upload Excel or CSV files (Multiple allowed)", type=["xlsx", "csv"], accept_multiple_files=True)

if uploaded_files:
    dataset_info = []
    document_list = []

    # Process Uploaded Files
    for file in uploaded_files:
        temp_path = os.path.join("temp_data", file.name)
        os.makedirs("temp_data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Read CSV or Excel File
        if file.name.endswith(".csv"):
            df = pd.read_csv(temp_path)
            file_type = "CSV"
            sheets = ["Single CSV File"]
        else:
            xls = pd.ExcelFile(temp_path)
            file_type = "Excel"
            sheets = xls.sheet_names

        # Store Dataset Info
        dataset_info.append({
            "File Name": file.name,
            "File Type": file_type,
            "Sheets": sheets,
            "Rows": df.shape[0] if file_type == "CSV" else None,
            "Columns": list(df.columns) if file_type == "CSV" else None
        })

        # Add to Document List for LlamaParse
        document_list.append(temp_path)

    # Display Dataset Information
    st.success("📂 Files uploaded successfully! Here’s an overview:")

    for info in dataset_info:
        st.write(f"📁 **{info['File Name']}** ({info['File Type']})")
        st.write(f"📜 **Sheets**: {', '.join(info['Sheets'])}")
        if info["Rows"] is not None:
            st.write(f"📊 **Rows**: {info['Rows']} | **Columns**: {len(info['Columns'])}")
            st.write(f"📝 **Column Names**: {', '.join(info['Columns'])}")

    # Preview Data
    st.subheader("📋 Dataset Preview")
    for file in uploaded_files:
        temp_path = os.path.join("temp_data", file.name)
        if file.name.endswith(".csv"):
            df = pd.read_csv(temp_path)
            st.write(f"📌 **{file.name} (First 5 Rows)**")
            st.dataframe(df.head())
        else:
            xls = pd.ExcelFile(temp_path)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                st.write(f"📌 **{file.name} - {sheet} (First 5 Rows)**")
                st.dataframe(df.head())

    # Load Data into Vector Index
    documents = SimpleDirectoryReader(input_files=document_list, file_extractor=file_extractor).load_data()
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = db_index.as_query_engine()

    # Query Input
    query = st.text_input("🔍 Ask a question about the dataset:")

    if st.button("Get Response") and query:
        with st.spinner("🔍 Processing..."):
            response = query_engine.query(query)
            explanation_prompt = f"Explain the response with deeper insights: {response.response}"
            explanation = query_engine.query(explanation_prompt).response

            st.success("💡 AI Response:")
            st.write(response.response)

            st.markdown("📌 **Additional Insights:**")
            st.info(explanation)

st.markdown("---")
st.markdown("🤖 **Built with LlamaIndex, Gemini AI & Qdrant**")
