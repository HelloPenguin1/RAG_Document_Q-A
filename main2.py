import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings


from dotenv import load_dotenv
load_dotenv()

##load the groq api
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')

llm = ChatGroq(groq_api_key=groq_api_key, model_name= "Llama3-8b-8192")


prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant and you have to answer the given question based on the context provided.

    <context>
    {context}
    <context>

    Question: {input}
    """
)

def create_vector_embeddings():
    state = st.session_state
    if "vectors" not in state:
        ## Data Ingestion
        state.loader = PyPDFDirectoryLoader('./medical')
        state.docs = state.loader.load()

        ##Embedding
        state.text_splitter =  SemanticChunker(OpenAIEmbeddings()) 
        state.final_docs = state.text_splitter.split_documents(state.docs)

        #vector storing
        state.vectorstore = FAISS.from_documents(state.final_docs, OpenAIEmbeddings())


st.title("Medical Paper RAG Q&A")


if st.button("Prepare Document for Q&A"):
    create_vector_embeddings()
    st.success("Vector Database is ready for use!")

user_query= st.text_input("Enter your query:")

if user_query:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever =  st.session_state.vectorstore.as_retriever()
    
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    response = retrieval_chain.invoke({"input": user_query})

    st.write(response['answer'])


