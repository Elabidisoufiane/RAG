import streamlit as st
import os
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
import tempfile
from typing import Iterator

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'llama3.1'

@st.cache_resource
def initialize_models():
    """Initialize and return the available models."""
    base_url = "http://127.0.0.1:11434"  # Replace with your actual Ollama base URL
    return {
        'llama3.1': {
            'llm': Ollama(model="llama3.1", base_url=base_url),
            'embeddings': OllamaEmbeddings(model="llama3.1", base_url=base_url),
        },
        'llama2': {
            'llm': Ollama(model="llama2", base_url=base_url),
            'embeddings': OllamaEmbeddings(model="llama2", base_url=base_url),
        },
        'mestral': {
            'llm': Ollama(model="mestral", base_url=base_url),
            'embeddings': OllamaEmbeddings(model="mestral", base_url=base_url),
        }
    }

# Load models
all_models = initialize_models()

# App header
st.title("📄 Chat with Your PDF and More!")
st.write("Upload a PDF and ask questions about it or explore general knowledge.")

# Model selection
selected_model_name = st.selectbox(
    "Select the model to use:",
    options=list(all_models.keys()),
    index=list(all_models.keys()).index(st.session_state.selected_model)
)
st.session_state.selected_model = selected_model_name
models = all_models[selected_model_name]

def process_pdf(pdf_file):
    """Process the uploaded PDF and create a vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        pdf_path = tmp_file.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    pdf_chunks = text_splitter.split_documents(pages)

    vector_store = Chroma.from_documents(
        documents=pdf_chunks,
        embedding=models['embeddings'],
        persist_directory="./pdf_chroma_db"
    )
    vector_store.persist()
    os.unlink(pdf_path)  # Clean up temporary file
    return vector_store

# PDF upload
uploaded_file = st.file_uploader("📤 Upload a PDF file", type="pdf")
if uploaded_file and not st.session_state.vector_store:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    st.success("✅ PDF processed successfully!")

def get_response(question) -> Iterator[str]:
    """Generate a response for the given question."""
    try:
        if st.session_state.vector_store:
            retriever = st.session_state.vector_store.as_retriever()
            prompt = PromptTemplate.from_template("""
            Answer the following question based on the provided context.
            If the question cannot be answered from the context, provide a general response.

            Context: {context}
            Question: {input}

            Answer: """)
            combine_docs_chain = create_stuff_documents_chain(
                llm=models['llm'],
                prompt=prompt
            )
            retrieval_chain = create_retrieval_chain(
                retriever,
                combine_docs_chain
            )
            response = retrieval_chain.invoke({"input": question})
            words = response['answer'].split()
            for word in words:
                yield word + " "
        else:
            response = models['llm'].invoke(question)
            words = response.split()
            for word in words:
                yield word + " "
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

def handle_user_input():
    """Handle user input and generate a response."""
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for chunk in get_response(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.user_input = ""  # Clear input

# Chat interface
st.text_input("💬 Ask a question:", key="user_input", on_change=handle_user_input)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Sidebar instructions
with st.sidebar:
    st.markdown("""
    ### 🛠️ How to Use:
    1. Upload a PDF document (optional)
    2. Ask questions about the PDF content
    3. Or ask any general knowledge questions

    The system will:
    - Use PDF content when relevant
    - Fall back to general knowledge otherwise
    """)
