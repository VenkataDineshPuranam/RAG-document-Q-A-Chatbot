### RAG Document Q&A with Groq and OpenAI Embeddings

A Streamlit application for document question-answering using RAG, Groq LLM, and OpenAI embeddings.

### Features
- PDF document processing 
- OpenAI embeddings
- FAISS vector search
- Streamlit interface
- Response time tracking

### Installation
```bash
pip install streamlit langchain-groq langchain-openai langchain-community python-dotenv faiss-cpu PyPDF2
```

### Setup
1. Create .env file:
```
GROQ_API_KEY=your_key
OPENAI_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
```

2. Place PDFs in `research_papers/` directory

### Usage
```bash
streamlit run main.py
```
1. Click "Document Embeddings"
2. Enter query
3. View response and relevant document segments

### Requirements
- Python 3.8+
- Groq API key
- OpenAI API key
- LangChain API key

### Structure
```
research_papers/     # PDF documents
├── main.py         # Streamlit app
├── .env            # API keys
└── README.md       # Documentation
```
