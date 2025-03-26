#final code for the streamlit app deployment

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
from fuzzywuzzy import fuzz

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

# Streamlit UI Configuration
st.set_page_config(page_title="Excel AI Assistant", layout="wide")
st.title("📊 Excel AI Assistant")
st.subheader("Upload an Excel file or folder, view dataset details, and ask queries.")

# File Upload
uploaded_files = st.file_uploader("Upload an Excel file or multiple files in a folder", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    dataset_info = []
    document_list = []
    dataframes = {}

    # Process Uploaded Files
    for file in uploaded_files:
        temp_path = os.path.join("temp_data", file.name)
        os.makedirs("temp_data", exist_ok=True)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        # Read Excel File to Extract Sheet Names and Data
        xls = pd.ExcelFile(temp_path)
        sheet_info = {"File Name": file.name, "Sheets": xls.sheet_names}
        dataset_info.append(sheet_info)
        
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            dataframes[(file.name, sheet)] = df
        
        document_list.append(temp_path)

    # Display Dataset Information
    st.success("Files uploaded successfully! Below are the dataset details:")
    for info in dataset_info:
        st.write(f"📂 **{info['File Name']}** - Sheets: {', '.join(info['Sheets'])}")
        
    # Display first 5 rows of each sheet
    for (file_name, sheet_name), df in dataframes.items():
        st.markdown(f"### 📄 {file_name} - Sheet: {sheet_name}")
        st.dataframe(df.head())
    
    # Load Data into Vector Index
    documents = SimpleDirectoryReader(input_files=document_list, file_extractor=file_extractor).load_data()
    VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = db_index.as_query_engine()

    # Query Input
    query = st.text_input("🔍 Ask a question about the dataset:")
    
    if st.button("Get Response") and query:
        with st.spinner("Processing..."):
            response = query_engine.query(query)
            explanation_prompt = f"Explain the response in more detail: {response.response}"
            explanation = query_engine.query(explanation_prompt).response
            
            # Identify relevant sheet using fuzzy matching
            best_match = (None, None, 0)
            for (file_name, sheet_name), df in dataframes.items():
                combined_text = ' '.join(df.astype(str).values.flatten())
                match_score = fuzz.partial_ratio(query, combined_text)
                if match_score > best_match[2]:
                    best_match = (file_name, sheet_name, match_score)
            
            file_name, sheet_name, _ = best_match if best_match[2] > 60 else ("All Files", ', '.join([s for _, s in dataframes.keys()]), 0)
            
            st.success("💡 AI Response:")
            st.write(response.response)

            st.markdown("📌 **Explanation:**")
            st.info(explanation)
            
            st.markdown("📍 **Source Dataset:**")
            st.write(f"File: {file_name}, Sheets: {sheet_name}")

st.markdown("---")
st.markdown("🤖 **Built with LlamaIndex, Gemini AI & Qdrant**")
