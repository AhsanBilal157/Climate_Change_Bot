import streamlit as st
import os
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()
# Load the Google API key from .env
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Directory for Chroma persistence
PERSISTENT_DIR = "chroma_db"
os.makedirs(PERSISTENT_DIR, exist_ok=True)  # Create the directory if it doesn't exist

def charomaDB_initializor():
    # Define Chroma database directory
    chroma_db = Chroma(
    collection_name="my_documents",
    persist_directory=PERSISTENT_DIR,
    embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY )
    )
    retriever = chroma_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

def chain_initializor():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    template = ChatPromptTemplate.from_template("""
    
    You are an expert climate change policies assistant who helps people with their queries. Always answer politely and accurately. 
    Consider the following context:
    {context}

    Based on this context, provide a direct and helpful answer to the following question:
    {input}
    
    Always follow these guidelines:
    1. Answer queries in a way that feels like you are sharing your own expertise, without mentioning the documents or stating that the answer is based on the provided text.
    2. If there is insufficient information in the context to answer the question, say: "I think your query doesn't relate to climate change policies, or we don't have sufficient data to answer your query. Please try with something else."
    3. Ensure responses are concise, clear, and easy to understand, avoiding any technical jargon unless explicitly requested and simple.
    4. Maintain a polite and professional tone, even when the question is casual or conversational.
    5. If the question or context suggests presenting information in a structured manner (e.g., bullet points or a table), or if explicitly requested by the user, use the appropriate format for clarity.


    """)

    chain = create_stuff_documents_chain(llm, template)
    return chain

def chatbot_initializor(input):
    retriever = charomaDB_initializor()
    chain = chain_initializor()
    qa_chain = create_retrieval_chain(retriever, chain)
    
    # query = "who is john cena"
    retrieved_docs = retriever.get_relevant_documents(input)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
    try:
        response = qa_chain.invoke({"input": input, "context": retrieved_text})
    except:
        response = {"answer":print("We are working on the issues")}
    return response['answer']


# Initialize Streamlit header
st.header("🌍 Demystifying Climate Change Policies")
st.text("")
input = st.text_input("🍃💚 Ask me anything about climate change policies and get clear, expert answers to your questions! 🌱🏞️")
st.text("")
if st.button("Submit"):
    if input:
        answer = chatbot_initializor(input)
        st.text("")
        st.subheader("The Response is")
        st.text("")
        st.write(answer)
    else:
        st.warning("Please enter a query.")


