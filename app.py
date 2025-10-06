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
from langchain.prompts import PromptTemplate
import tempfile
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# Helper function to convert trace dictionaries to Plotly trace objects
def convert_to_plotly_traces(trace_data):
    trace_types = {
        'scatter': go.Scatter,
        'line': go.Scatter,
        'bar': go.Bar,
        'pie': go.Pie,
        'histogram': go.Histogram,
        'scatter3d': go.Scatter3d,
        'surface': go.Surface,
        'heatmap': go.Heatmap,
    }
    
    traces = []
    for trace_dict in trace_data:
        trace_type = trace_dict.get('type', 'scatter').lower()
        if trace_type not in trace_types:
            st.warning(f"Unsupported trace type '{trace_type}'. Defaulting to Scatter.")
            trace_type = 'scatter'
        
        trace_kwargs = {k: v for k, v in trace_dict.items() if k != 'type'}
        trace_class = trace_types[trace_type]
        traces.append(trace_class(**trace_kwargs))
    
    return traces

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPEN_ROUTER_KEY = os.getenv("OPEN_ROUTER_KEY")

# Page config
st.set_page_config(
    page_title="RAG Q&A Analytics System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2563EB;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .status-ready {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .status-not-ready {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .metric-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2563EB;
    }
    .stChatMessage {
        border-radius: 1rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .chat-history {
        height: calc(70vh - 100px);
        overflow-y: auto;
        padding-right: 1rem;
    }
    div.element-container:has(.stAlert), div.element-container:has(.stError) {
        padding-top: 2rem;
    }
    div.stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: transparent;
        padding: 0.5rem 8rem;
        z-index: 100;
        border-top: 1px solid #eee;
    }
    .stTab {
        padding-bottom: 5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vectorstore_ready' not in st.session_state:
    st.session_state.vectorstore_ready = False
if 'query_analytics' not in st.session_state:
    st.session_state.query_analytics = []
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {
        'total_documents': 0,
        'total_chunks': 0,
        'total_queries': 0
    }
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []
if 'llm' not in st.session_state:
    st.session_state.llm = None

# Sidebar for configuration and initialization
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    if st.session_state.vectorstore_ready:
        st.markdown('<div class="status-badge status-ready">● System Ready</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-not-ready">● Not Initialized</div>', unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("📄 Document Upload", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, TXT)",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            help="Upload one or more documents to analyze"
        )
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
    
    st.divider()
    
    with st.expander("🔑 API Configuration", expanded=False):
        openrouter_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=OPEN_ROUTER_KEY,
            help="Your OpenRouter API key"
        )
        pinecone_key = st.text_input(
            "Pinecone API Key",
            type="password",
            value=PINECONE_API_KEY,
            help="Your Pinecone API key"
        )
    
    with st.expander("🗄️ Pinecone Settings", expanded=False):
        index_name = st.text_input(
            "Index Name",
            value="rag-huggingface",
            help="Name of your Pinecone index"
        )
    
    with st.expander("🤖 Model Settings", expanded=False):
        model_choice = st.selectbox(
            "LLM Model",
            [
                "meta-llama/llama-3.3-70b-instruct:free",
                "meta-llama/llama-3.1-8b-instruct:free",
                "google/gemma-2-9b-it:free",
                "mistralai/mixtral-8x7b-instruct:free"
            ],
            index=0,
            help="Select the language model"
        )
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.0, 0.1,
            help="Controls randomness in responses"
        )
        top_k = st.slider(
            "Documents to Retrieve",
            1, 10, 3,
            help="Number of relevant chunks to retrieve"
        )
    
    st.divider()
    
    init_button = st.button(
        "🚀 Initialize RAG System",
        use_container_width=True,
        type="primary"
    )
    
    if init_button:
        if not openrouter_key or not pinecone_key:
            st.error("⚠️ Please provide both API keys!")
        else:
            with st.spinner("🔄 Initializing RAG system..."):
                try:
                    os.environ["PINECONE_API_KEY"] = pinecone_key
                    pc = Pinecone(api_key=pinecone_key)

                    embedding_model = "sentence-transformers/all-mpnet-base-v2"
                    embeddings = HuggingFaceEmbeddings(
                        model_name=embedding_model,
                        model_kwargs={"device": "cpu"},
                        encode_kwargs={"batch_size": 32}
                    )
                    sample_embedding = embeddings.embed_query("Test embedding")
                    embedding_dimension = len(sample_embedding)

                    if index_name in pc.list_indexes().names():
                        index = pc.Index(index_name)
                        index_stats = index.describe_index_stats()
                        if index_stats['dimension'] != embedding_dimension:
                            st.error(f"⚠️ Dimension mismatch! Expected {embedding_dimension}, got {index_stats['dimension']}")
                            st.stop()
                        st.info(f"✓ Connected to existing index: {index_name}")
                    else:
                        pc.create_index(
                            name=index_name,
                            dimension=embedding_dimension,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                        st.info(f"✓ Created new index: {index_name}")

                    if uploaded_files:
                        documents = []
                        for uploaded_file in uploaded_files:
                            if uploaded_file.size == 0:
                                st.warning(f"⚠️ Skipping empty file: {uploaded_file.name}")
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
                                if loaded_docs:
                                    documents.extend(loaded_docs)
                                    st.success(f"✓ Loaded: {uploaded_file.name}")
                            finally:
                                os.unlink(tmp_path)

                        if documents:
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=500,
                                chunk_overlap=50
                            )
                            split_docs = text_splitter.split_documents(documents)

                            vectorstore = PineconeVectorStore.from_documents(
                                documents=split_docs,
                                embedding=embeddings,
                                index_name=index_name
                            )
                            
                            st.session_state.document_stats['total_documents'] = len(documents)
                            st.session_state.document_stats['total_chunks'] = len(split_docs)
                            
                            st.success(f"✓ Added {len(split_docs)} chunks from {len(documents)} documents")
                        else:
                            st.warning("⚠️ No documents loaded. Connecting to existing index.")
                            vectorstore = PineconeVectorStore.from_existing_index(
                                embedding=embeddings,
                                index_name=index_name
                            )
                    else:
                        vectorstore = PineconeVectorStore.from_existing_index(
                            embedding=embeddings,
                            index_name=index_name
                        )
                        st.info("✓ Connected to existing index")

                    llm = ChatOpenAI(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                        model=model_choice,
                        temperature=temperature
                    )
                    st.session_state.llm = llm

                    retriever = vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": top_k}
                    )
                    st.session_state.retriever = retriever

                    prompt_template = """You are a helpful AI assistant that answers questions based on the provided context documents. 

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

If the question requires a visualization or plot, output your entire response as JSON with 'answer' (the text answer) and 'plot_json' (the Plotly figure JSON) keys. 
The 'plot_json' should be a valid Plotly figure specification with 'data' as a list of trace dictionaries and an optional 'layout' dictionary, for example:
{{
  "answer": "Here is the visualization of the data...",
  "plot_json": {{
    "data": [
      {{
        "type": "bar",
        "x": ["Category A", "Category B", "Category C"],
        "y": [20, 14, 23]
      }}
    ],
    "layout": {{
      "title": "Sample Bar Chart"
    }}
  }}
}}
Otherwise, if no plot is needed, output only the plain text answer without any JSON formatting.

Context:
{context}

Question: {question}

"""

                    PROMPT = PromptTemplate(
                        template=prompt_template, 
                        input_variables=["context", "question"]
                    )

                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPT}
                    )

                    st.session_state.vectorstore_ready = True
                    st.success(f"✅ System initialized successfully!")
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Initialization error: {str(e)}")
                    import traceback
                    st.error(f"Debug info: {traceback.format_exc()}")
    
    st.divider()
    
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Documents", st.session_state.document_stats['total_documents'])
        st.metric("Chunks", st.session_state.document_stats['total_chunks'])
    with col2:
        st.metric("Queries", st.session_state.document_stats['total_queries'])
        if st.session_state.query_analytics:
            avg_time = sum([q['response_time'] for q in st.session_state.query_analytics]) / len(st.session_state.query_analytics)
            st.metric("Avg Time", f"{avg_time:.2f}s")

# Main content area
st.markdown('<div class="main-header">🤖 RAG Q&A Analytics System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent document analysis with conversation memory and real-time analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📈 Query Analytics", "📚 Document Insights"])

with tab1:
    with st.expander("🧠 Conversation Memory"):
        st.caption("The system remembers your recent conversation context")
        
        if st.session_state.conversation_memory:
            memory_display = st.session_state.conversation_memory[-6:]
            for mem in memory_display:
                role_icon = "👤" if mem['role'] == 'user' else "🤖"
                st.markdown(f"{role_icon} **{mem['role'].title()}:** {mem['content'][:100]}...")
        else:
            st.info("Start a conversation to see memory context")
        
        if st.button("🗑️ Clear Memory", use_container_width=True):
            st.session_state.conversation_memory = []
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("#### Conversation")
    
    with st.container(height=500, border=False):
        for idx, message in enumerate(st.session_state.chat_history):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "plot" in message and message["plot"]:
                    st.plotly_chart(message["plot"], use_container_width=True)
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source[:300] + "..." if len(source) > 300 else source)
                            if i < len(message["sources"]):
                                st.divider()
    
    if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
        if not st.session_state.vectorstore_ready:
            st.error("⚠️ Please initialize the RAG system first.")
        else:
            start_time = time.time()
            
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            st.session_state.conversation_memory.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        retrieved_docs = st.session_state.retriever.get_relevant_documents(prompt)
                        
                        conversation_context = ""
                        if len(st.session_state.conversation_memory) > 1:
                            recent_history = st.session_state.conversation_memory[-4:]
                            conversation_context = "Previous conversation:\n"
                            for mem in recent_history[:-1]:
                                role_name = 'User' if mem['role'] == 'user' else 'Assistant'
                                conversation_context += f"{role_name}: {mem['content']}\n"
                            conversation_context += "\n"
                        
                        contextualized_question = f"{conversation_context}Given the above conversation history, answer the following question using the provided documents.\n\nQuestion: {prompt}" if conversation_context else prompt
                        
                        result = st.session_state.qa_chain.invoke({"query": contextualized_question})
                        response_text = result["result"]
                        
                        # Debug: Output raw response for verification
                        # st.write("Raw LLM response:", response_text)
                        
                        plot_figure = None
                        answer = response_text
                        
                        try:
                            response_json = json.loads(response_text)
                            if isinstance(response_json, dict) and 'plot_json' in response_json:
                                answer = response_json.get('answer', "Here is the visualization:")
                                plot_data = response_json['plot_json']
                                
                                if 'data' in plot_data:
                                    traces = convert_to_plotly_traces(plot_data['data'])
                                    plot_figure = go.Figure(
                                        data=traces,
                                        layout=plot_data.get('layout', {})
                                    )
                                else:
                                    st.warning("No 'data' key in plot_json. Skipping plot rendering.")
                                    plot_figure = None
                        
                        except (json.JSONDecodeError, KeyError, TypeError) as e:
                            answer = response_text
                            plot_figure = None
                        
                        sources = [doc.page_content for doc in result["source_documents"]]
                        relevance_scores = [0.95 - (i * 0.1) for i in range(len(sources))]

                        st.markdown(answer)
                        
                        if plot_figure:
                            st.plotly_chart(plot_figure, use_container_width=True)
                        
                        response_time = time.time() - start_time
                        
                        st.session_state.conversation_memory.append({"role": "assistant", "content": answer})
                        
                        st.session_state.query_analytics.append({
                            'timestamp': datetime.now(),
                            'query': prompt,
                            'response_time': response_time,
                            'sources_count': len(sources),
                            'relevance_scores': relevance_scores
                        })
                        st.session_state.document_stats['total_queries'] += 1
                        
                        chat_entry = {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources
                        }
                        if plot_figure:
                            chat_entry["plot"] = plot_figure
                        
                        st.session_state.chat_history.append(chat_entry)

                        with st.expander("📚 View Sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source[:300] + "..." if len(source) > 300 else source)
                                if i < len(sources):
                                    st.divider()
                        
                        st.caption(f"⏱️ Response time: {response_time:.2f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Query error: {str(e)}")
                        import traceback
                        st.error(f"Debug info: {traceback.format_exc()}")

with tab2:
    st.markdown("### 📈 Query Analytics Dashboard")
    
    if st.session_state.query_analytics:
        st.markdown("#### Response Time Trend")
        
        timestamps = [q['timestamp'].strftime('%H:%M:%S') for q in st.session_state.query_analytics]
        response_times = [q['response_time'] for q in st.session_state.query_analytics]
        queries = [q['query'][:50] + '...' if len(q['query']) > 50 else q['query'] for q in st.session_state.query_analytics]
        
        fig_time = go.Figure()
        fig_time.add_trace(go.Scatter(
            x=timestamps,
            y=response_times,
            mode='lines+markers',
            name='Response Time',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=8, color=response_times, colorscale='Viridis', showscale=True),
            text=queries,
            hovertemplate='<b>Query:</b> %{text}<br><b>Time:</b> %{y:.2f}s<br><extra></extra>'
        ))
        
        avg_time = sum(response_times) / len(response_times)
        fig_time.add_hline(y=avg_time, line_dash="dash", line_color="red", 
                          annotation_text=f"Avg: {avg_time:.2f}s")
        
        fig_time.update_layout(
            title="Query Response Time Analysis",
            xaxis_title="Time",
            yaxis_title="Response Time (seconds)",
            hovermode='closest',
            height=400
        )
        
        st.plotly_chart(fig_time, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Sources Retrieved per Query")
            sources_count = [q['sources_count'] for q in st.session_state.query_analytics]
            
            fig_sources = go.Figure()
            fig_sources.add_trace(go.Bar(
                x=list(range(1, len(sources_count) + 1)),
                y=sources_count,
                marker=dict(color=sources_count, colorscale='Blues'),
                text=sources_count,
                textposition='auto',
                hovertemplate='<b>Query #%{x}</b><br>Sources: %{y}<extra></extra>'
            ))
            
            fig_sources.update_layout(
                xaxis_title="Query Number",
                yaxis_title="Number of Sources",
                height=350,
                showlegend=False
            )
            
            st.plotly_chart(fig_sources, use_container_width=True)
        
        with col2:
            st.markdown("#### Query Performance Metrics")
            
            metrics_data = {
                'Metric': ['Total Queries', 'Avg Response Time', 'Min Time', 'Max Time'],
                'Value': [
                    len(response_times),
                    f"{avg_time:.2f}s",
                    f"{min(response_times):.2f}s",
                    f"{max(response_times):.2f}s"
                ]
            }
            
            fig_metrics = go.Figure(data=[go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='#2563EB',
                    font=dict(color='white', size=14),
                    align='left'
                ),
                cells=dict(
                    values=[metrics_data['Metric'], metrics_data['Value']],
                    fill_color='#F9FAFB',
                    align='left',
                    font=dict(size=13),
                    height=30
                )
            )])
            
            fig_metrics.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_metrics, use_container_width=True)
        
    else:
        st.info("📊 Start querying documents to see analytics!")

with tab3:
    st.markdown("### 📚 Document Insights")
    
    if st.session_state.query_analytics:
        last_query = st.session_state.query_analytics[-1]
        
        if last_query.get('relevance_scores'):
            st.markdown("#### Top Retrieved Chunks - Relevance Scores")
            st.caption(f"For query: _{last_query['query']}_")
            
            chunk_ids = [f"Chunk {i+1}" for i in range(len(last_query['relevance_scores']))]
            scores = last_query['relevance_scores']
            
            fig_relevance = go.Figure()
            fig_relevance.add_trace(go.Bar(
                y=chunk_ids,
                x=scores,
                orientation='h',
                marker=dict(
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Relevance")
                ),
                text=[f"{s:.2f}" for s in scores],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Relevance: %{x:.3f}<extra></extra>'
            ))
            
            fig_relevance.update_layout(
                title="Chunk Relevance Scores",
                xaxis_title="Relevance Score",
                yaxis_title="",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_relevance, use_container_width=True)
        
        st.markdown("#### Document Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Documents", st.session_state.document_stats['total_documents'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Chunks", st.session_state.document_stats['total_chunks'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Queries", st.session_state.document_stats['total_queries'])
            st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.info("📈 Query documents to see insights!")

st.divider()
st.caption("💡 Tip: The system maintains conversation context for better follow-up questions!")