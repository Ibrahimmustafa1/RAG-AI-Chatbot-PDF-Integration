import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Loads API key

model = 'models/embedding-001'
chat_model = genai.GenerativeModel('gemini-1.5-flash')

def extract_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for i, page in enumerate(reader.pages):
        if page:
            text += page.extract_text()
    return text

def google_pdf_gemini_embedding(text, type):
    embedding = GoogleGenerativeAIEmbeddings(model=model, task_type=type)
    return embedding

def create_vector_db(texts):
    v_db = FAISS.from_texts(texts, google_pdf_gemini_embedding(texts, "SEMANTIC_SIMILARITY"))
    return v_db

def get_similar_context(v_db, v_user, n):
    if v_user:
        docs = v_db.similarity_search(v_user, k=n)
        return docs

def get_response(query):
    response = chat_model.generate_content(query, stream=True)
    for res in response:
        if res.text:
          yield res.text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# HTML and CSS for styled title
title_html = """
    <style>
    .title {
        font-size: 70px;
        font-weight: 800;
        color: #c13584; /* Gradient color start */
        background: -webkit-linear-gradient(#4c68d7, #ff6464); /* Gradient background */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 50px;
        font-weight: 400;
        color: #333337; /* Subtitle color */
    }
    </style>
    <div class="title">Hello,</div>
    <div class="subtitle">How can I help you today?</div>
    """

st.markdown(title_html, unsafe_allow_html=True)

if "pdf" not in st.session_state:
    st.session_state.pdf = None
if "v_db" not in st.session_state:
    st.session_state.v_db = None
if "texts" not in st.session_state:
    st.session_state.texts = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("Chatbot")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf and st.button("Create Vector Database"):
        with st.spinner("Creating vector database..."):
            texts = text_splitter.split_text(extract_from_pdf(pdf))
            st.session_state.v_db = create_vector_db(texts)
            st.session_state.pdf = pdf
            st.session_state.texts = texts
            st.success("Vector database created successfully!")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

    if st.button("Delete Vector Database"):
        st.session_state.v_db = None
        st.session_state.pdf = None
        st.session_state.texts = None
        st.success("Vector database deleted!")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Enter your message:")

if user_input:
    st.chat_message("user").write(user_input)
    placeholder = st.chat_message("AI").empty()
    similar_text = "You are an expert researcher in generative AI and machine learning."
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.v_db:
        similar_context = get_similar_context(st.session_state.v_db, user_input, 5)
        for doc in similar_context:
            similar_text += doc.page_content

    with st.spinner("Thinking..."):
        stream_res = ""
        conversation_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
        combined_input = f"{conversation_history}\nuser: {user_input}\nAI:"
        combined_input += similar_text

        for response in get_response(combined_input):
            stream_res += response
            placeholder.markdown(stream_res)
        st.session_state.messages.append({"role": "AI", "content": stream_res})
