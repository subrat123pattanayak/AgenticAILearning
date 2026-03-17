import streamlit as st
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os

st.set_page_config(page_title='HR Assistant Portal', layout='centered')
st.title('🧑‍💼 HR Assistant Portal')
st.write('Upload HR Guidelines & Ask Questions')

# Sidebar
with st.sidebar:
    st.header('⚙️ Configuration')
    user_api_key = st.text_input('Enter Groq API Key:', type='password')
    st.info('Upload HR policy files (PDF / TXT)')

    uploaded_files = st.file_uploader(
        "Upload HR Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

# Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# Load Documents
def process_documents(files):
    documents = []

    for file in files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if file.type == "application/pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        documents.extend(loader.load())
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    return vectorstore

# Process Upload
if uploaded_files and st.session_state.vectorstore is None:
    with st.spinner("Processing HR documents..."):
        st.session_state.vectorstore = process_documents(uploaded_files)
    st.success("Documents ready! You can ask questions now.")

# Show Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat Input
if user_query := st.chat_input('Ask HR related questions...'):

    if not user_api_key:
        st.error("Please enter your API Key in the sidebar first!")

    elif st.session_state.vectorstore is None:
        st.error("Please upload HR documents first!")

    else:
        st.session_state.messages.append({'role': 'user', 'content': user_query})
        with st.chat_message('user'):
            st.markdown(user_query)

        llm = ChatGroq(
            temperature=0.3,
            model='llama-3.3-70b-versatile',
            api_key=user_api_key
        )

        with st.spinner('HR Assistant is thinking...'):
            docs = st.session_state.vectorstore.similarity_search(user_query, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
            You are an HR Assistant.
            Answer strictly from the HR policy documents.

            HR Policies:
            {context}

            Question:
            {user_query}
            """

            response = llm.invoke(prompt)
            bot_answer = response.content

        st.session_state.messages.append({'role': 'assistant', 'content': bot_answer})
        with st.chat_message('assistant'):
            st.markdown(bot_answer)
