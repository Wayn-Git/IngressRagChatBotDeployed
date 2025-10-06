import streamlit as st
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile


# Load environment variables
load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
OPEN_ROUTER_KEY=os.getenv("OPEN_ROUTER_KEY")

# Page config
st.set_page_config(
    page_title="RAG Q&A System with HuggingFace Embeddings",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = False

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Keys
    st.subheader("API Keys")
    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        value=OPEN_ROUTER_KEY
    )
    pinecone_key = st.text_input(
        "Pinecone API Key",\
        value=PINECONE_API_KEY
    )
    
    # Pinecone settings
    st.subheader("Pinecone Settings")
    index_name = st.text_input("Index Name", value="rag-huggingface")
    
    # Model settings
    st.subheader("Model Settings")
    model_choice = st.selectbox(
        "LLM Model",
        [
            "meta-llama/llama-3.3-70b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "mistralai/mistral-7b-instruct:free"
        ],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.slider("Documents to Retrieve", 1, 10, 3)
    
    st.divider()
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

# Initialize RAG system
if st.button("🚀 Initialize RAG System", use_container_width=True):
    if not openrouter_key or not pinecone_key:
        st.error("Please provide both API keys!")
    else:
        with st.spinner("Initializing RAG system..."):
            try:
                # Set Pinecone API key
                os.environ["PINECONE_API_KEY"] = pinecone_key
                pc = Pinecone(api_key=pinecone_key)

                # Check and create index
                embedding_model = "sentence-transformers/all-mpnet-base-v2"
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model,
                    model_kwargs={"device": "cpu"},  # Force CPU to avoid GPU dependency
                    encode_kwargs={"batch_size": 32}  # Smaller batches to reduce memory
                )
                sample_embedding = embeddings.embed_query("Test embedding")
                embedding_dimension = len(sample_embedding)
                st.write(f"Embedding dimension: {embedding_dimension}")

                if index_name in pc.list_indexes().names():
                    index = pc.Index(index_name)
                    index_stats = index.describe_index_stats()
                    if index_stats['dimension'] != embedding_dimension:
                        st.error(f"Index {index_name} has dimension {index_stats['dimension']}, expected {embedding_dimension}. Please delete the index in Pinecone and try again.")
                    else:
                        st.info(f"Connected to existing index: {index_name}")
                else:
                    pc.create_index(
                        name=index_name,
                        dimension=embedding_dimension,
                        metric="cosine",
                        spec=ServerlessSpec(cl2oud="aws", region="us-east-1")
                    )
                    st.info(f"Created new index: {index_name}")

                # Process uploaded files if any
                if uploaded_files:
                    documents = []
                    for uploaded_file in uploaded_files:
                        if uploaded_file.size == 0:
                            st.error(f"File {uploaded_file.name} is empty!")
                            continue
                        if not uploaded_file.name.endswith(('.pdf', '.txt')):
                            st.error(f"Unsupported file type: {uploaded_file.name}")
                            continue
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        try:
                            if uploaded_file.name.endswith('.pdf'):
                                loader = PyPDFLoader(tmp_path)
                            else:
                                loader = TextLoader(tmp_path)

                            loaded_docs = loader.load()
                            if not loaded_docs:
                                st.error(f"No documents loaded from {uploaded_file.name}")
                                continue
                            documents.extend(loaded_docs)
                        finally:
                            os.unlink(tmp_path)

                    if not documents:
                        st.error("No valid documents loaded. Please upload valid files.")
                    else:
                        # Debug: Inspect loaded documents
                        for doc in documents:
                            if not isinstance(doc, Document):
                                st.error(f"Loaded document is not of type Document: {type(doc)}")
                            else:
                                st.write(f"Loaded doc content: {doc.page_content[:100]}")

                        # Split documents with smaller chunk size
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=300,  # Smaller chunks for less memory
                            chunk_overlap=30
                        )
                        split_docs = text_splitter.split_documents(documents)

                        # Debug: Inspect split documents
                        for doc in split_docs:
                            if not isinstance(doc, Document):
                                st.error(f"Split document is not of type Document: {type(doc)}")
                            else:
                                st.write(f"Split doc content: {doc.page_content[:100]}")

                        # Add to vectorstore
                        vectorstore = PineconeVectorStore.from_documents(
                            documents=split_docs,
                            embedding=embeddings,
                            index_name=index_name
                        )
                        st.info(f"Added {len(split_docs)} chunks to {index_name}")

                else:
                    # Connect to existing vectorstore
                    vectorstore = PineconeVectorStore.from_existing_index(
                        embedding=embeddings,
                        index_name=index_name
                    )
                    st.info(f"Connected to existing index: {index_name}")

                # Setup LLM
                llm = ChatOpenAI(
                    api_key=openrouter_key,
                    base_url="https://openrouter.ai/api/v1",
                    model=model_choice,
                    temperature=temperature
                )

                # Create retriever and QA chain
                retriever = vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k}
                )

                # Debug: Test retriever
                try:
                    sample_docs = retriever.get_relevant_documents("test query")
                    for doc in sample_docs:
                        if not isinstance(doc, Document):
                            st.error(f"Retrieved document is not of type Document: {type(doc)}")
                        else:
                            st.write(f"Retrieved doc type: {type(doc)}, content: {doc.page_content[:100]}")
                except Exception as e:
                    st.error(f"Retriever test failed: {str(e)}")

                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )

                st.session_state.vectorstore_ready = True
                st.success(f"✅ RAG system initialized and connected to Pinecone index: {index_name}")

            except Exception as e:
                st.error(f"Initialization error: {str(e)}")

# Main content
st.title("🤖 RAG-Powered Q&A System with HuggingFace Embeddings")
st.markdown("Ask questions about your uploaded documents!")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(source[:300] + "..." if len(source) > 300 else source)
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.vectorstore_ready:
        st.error("Please initialize the RAG system first.")
    else:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": prompt})
                    answer = result["result"]
                    sources = [doc.page_content for doc in result["source_documents"]]

                    st.markdown(answer)

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                            st.divider()
                except Exception as e:
                    st.error(f"Query error: {str(e)}")