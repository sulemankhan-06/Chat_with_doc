import streamlit as st
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from load_dotenv import load_dotenv
import os
load_dotenv()

MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')


# Set page configuration
st.set_page_config(page_title="Chat with PDF", layout="wide")

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key = MISTRAL_API_KEY)

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatGroq(model ="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def main():
    st.header("💬 Chat with your PDF")
    
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Upload PDF files
    pdf_files = st.file_uploader(
        "Upload your PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if pdf_files:
        if st.button("Process PDFs"):
            with st.spinner("Processing PDFs..."):
                # Get PDF text
                raw_text = get_pdf_text(pdf_files)
                
                # Get text chunks
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store
                vectorstore = get_vectorstore(text_chunks)
                
                # Create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                
                st.success("PDFs processed successfully!")
    
    # Chat interface
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about your PDFs:")
        
        if user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({
                    'question': user_question
                })
                st.session_state.chat_history.append((user_question, response['answer']))
        
        # Display chat history
        for question, answer in st.session_state.chat_history:
            with st.container():
                st.write(f"**You:** {question}")
                st.write(f"**Assistant:** {answer}")
                st.divider()

if __name__ == '__main__':
    main()