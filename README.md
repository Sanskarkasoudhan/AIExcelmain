# Excel AI Assistant

## 📌 Overview
Excel AI Assistant is a Streamlit-based application that enables users to upload Excel (XLSX) files, extract structured data, and query the dataset using natural language. The system leverages LlamaIndex, Gemini AI, FAISS, and Qdrant for efficient data retrieval and AI-powered insights.

## 🚀 Features
- 📂 **Upload multiple Excel files** and extract sheet names and contents.
- 🔍 **AI-powered querying** using LlamaIndex and Gemini AI.
- 📊 **Vector search with FAISS & Qdrant** for efficient retrieval.
- 📌 **Fuzzy matching for relevant sheets** to improve query accuracy.
- 📝 **Detailed AI explanations** based on user queries.
- 🛠️ **Persistent storage of embeddings** using FAISS and Qdrant.
- 🌐 **Streamlit-based web UI** for an interactive experience.

## 🏗️ Tech Stack
- **LlamaIndex** (for indexing and querying data)
- **Google Gemini AI** (for LLM-based responses and embeddings)
- **FAISS** (for efficient similarity search)
- **Qdrant** (for scalable vector search)
- **Streamlit** (for frontend UI)
- **LlamaParse** (for structured data extraction from XLSX)
- **FuzzyWuzzy** (for fuzzy matching of relevant sheets)
- **Pandas** (for data handling and visualization)

## 🔧 Setup Instructions
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/Sanskarkasoudhan/excel-ai-assistant.git
cd excel-ai-assistant
```

### 2️⃣ Install Dependencies
Make sure you have Python installed, then install the required dependencies using:
```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up Environment Variables
Create a `.env` file in the project root and add the following:
```sh
GOOGLE_API_KEY=your_google_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
LAMA_API_KEY=your_llama_parse_api_key
```

### 4️⃣ Run the Streamlit App
```sh
streamlit run app.py
```

## 📌 Usage
1. **Upload an Excel file** (or multiple files).
2. **View dataset details**, including sheet names and previews.
3. **Ask queries** related to the dataset.
4. **Get AI-generated responses** with explanations and source data references.

## 🛠️ Deployment
For deploying the application, you can use services like:
- **Streamlit Cloud**
- **Docker** (Containerize and deploy using Docker)
- **Heroku** / **Google Cloud Run**

## 📜 Requirements.txt
```txt
fastembed
llama-index
llama-index-embeddings-fastembed
llama-index-embeddings-gemini
llama-index-llms-gemini
llama-index-vector-stores-qdrant
llama-parse
python-dotenv
qdrant_client
streamlit
openpyxl
pandas
fuzzywuzzy
```

## 🤖 Contributing
Feel free to contribute to this project by submitting issues or pull requests.

## 📄 License
MIT License

## 💡 Acknowledgments
- OpenAI for LLM capabilities.
- LlamaIndex for indexing and retrieval.
- FAISS & Qdrant for vector search.
- Streamlit for UI development.
- Pandas for data processing.

---
🚀 **Excel AI Assistant - Bringing AI to Your Data!**

