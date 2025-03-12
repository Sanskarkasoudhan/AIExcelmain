import os
from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
import nest_asyncio
from dotenv import load_dotenv


nest_asyncio.apply()
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
# GROQ_API_KEY = "" # add your GROQ API key here
# # OPENAI_API_KEY = "" # add your OPENAI API key here

parser = LlamaParse(
    api_key=os.getenv("LAMA_API_KEY"),    #LAMA CLOUD API KEY
    parsing_instruction = """ you are parsing the uploaded file. Extract all sheet names and their contents in a structured format""",
    result_type="markdown"
)

file_extractor = {".xlsx": parser}
documents = SimpleDirectoryReader(input_files=['Samplefile.xlsx'], file_extractor=file_extractor).load_data()


llm = Gemini(model="models/gemini-1.5-flash", api_key =os.getenv("GOOGLE_API_KEY")) # 

# Configuring LLM and embeddings
Settings.llm = llm

embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en")

Settings.embed_model = embed_model



# Qdrant Vectorstore
QDRANT_URL =os.getenv('QDRANT_URL') #Qdrant URL
QDRANT_API_KEY =os.getenv('QDRANT_API_KEY')#Qdrant API Key

qdrantClient = QdrantClient(
        url=QDRANT_URL,
        prefer_grpc=True,
        api_key=QDRANT_API_KEY)

vector_store = QdrantVectorStore(client=qdrantClient, collection_name="Misc_data")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

VectorStoreIndex.from_documents(documents, storage_context=storage_context)
#vector_store = QdrantVectorStore(client=qdrantClient, collection_name="Misc_data")
db_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = db_index.as_query_engine()


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