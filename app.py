import os
import nest_asyncio
import faiss
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_parse import LlamaParse
from dotenv import load_dotenv

load_dotenv()
nest_asyncio.apply()

# Set Google API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Load and Parse XLSX File
parser = LlamaParse(
    api_key=os.getenv("LAMA_API_KEY"),
    parsing_instruction="Extract all sheet names and their contents in a structured format",
    result_type="markdown"
)

file_extractor = {".xlsx": parser}
documents = SimpleDirectoryReader(input_files=["Samplefile.xlsx"], file_extractor=file_extractor).load_data()

# Set LLM (Gemini)
llm = Gemini(model="models/gemini-1.5-flash", api_key=os.getenv('GOOGLE_API_KEY'))
Settings.llm = llm

# Set Embedding Model (Gemini)
embed_model = GeminiEmbedding(model_name="models/embedding-001", api_key=os.getenv('GOOGLE_API_KEY'))
Settings.embed_model = embed_model

# FAISS Vector Store
dimensions = 768
faiss_index = faiss.IndexFlatL2(dimensions)
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Build Index
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Save Index to Disk
index.storage_context.persist()

# Query Engine
query_engine = index.as_query_engine(similarity_top_k=5)

def get_response(query):
    """Fetch AI-generated response for a query."""
    response = query_engine.query(query)
    return response.response

# CLI Interface
if __name__ == "__main__":
    print("\nAI Query CLI: Type your query or 'exit' to quit.\n")
    while True:
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting...")
            break
        response = get_response(query)
        print(f"Response: {response}\n")
