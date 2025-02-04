import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv

load_dotenv()

## Load Groq API

# os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "RAG Document Q&A"

groq_api_key = os.environ["GROQ_API_KEY"] 
openapi_key = os.environ["OPENAI_API_KEY"]
lang_chain_key = os.environ["LANGCHAIN_API_KEY"]

llm = ChatGroq(groq_api_key = groq_api_key, model_name='gemma2-9b-it' )

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only. Here is the context:
    please provide the most accurate response based on the question
    <Context>
    {context}
    <Context>
    Question: {input}
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") # Data Ingestion Step
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        #embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("RAG Document Q&A with Groq and OpenAI Embeddings")

user_prompt = st.text_input("Enter your query from the documents from research papers")

if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector DB is created")
    
import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start_time = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    response_time = time.time()
    st.write("Response time: ", response_time - start_time)
    st.write(response['answer'])
    
    ## With a Streamlit expander
    with st.expander("Document Similarity Search"):
        # Find relevant chunks
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("--------------------------------")
