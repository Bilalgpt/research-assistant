import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# Import RAG components
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(page_title="Research Assistant", page_icon="🔎", layout="wide")
st.title("🔎 Research Assistant")
st.markdown("Search the web, academic papers, and your own documents")

# Sidebar for settings
with st.sidebar:
    st.title("Settings")
    api_key = st.text_input("Enter your Groq API key:", type="password")
    model_name = st.selectbox(
        "Select LLM model:",
        ["Llama3-8b-8192", "Gemma2-9b-It", "Mixtral-8x7b-32768"]
    )
    search_mode = st.radio("Search mode:", ["Web & Academic", "PDF Documents", "All Sources"])

# Initialize session states
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm your research assistant. I can search the web, academic sources, and your documents. How can I help you?"}
    ]

if "pdf_processed" not in st.session_state:
    st.session_state["pdf_processed"] = False

if "store" not in st.session_state:
    st.session_state.store = {}

# Get session history helper function
def get_session_history(session_id):
    """Return chat history for the given session ID"""
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Setup PDF processing if needed
if search_mode in ["PDF Documents", "All Sources"]:
    with st.sidebar:
        st.subheader("PDF Document Settings")
        session_id = st.text_input("Session ID", value="default_session")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files and not st.session_state["pdf_processed"] and api_key:
            with st.spinner("Processing PDFs..."):
                # Initialize HuggingFace embeddings
                os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Process uploaded PDFs
                documents = []
                for uploaded_file in uploaded_files:
                    # Save temporary file
                    temp_pdf = f"./temp_{uploaded_file.name}"
                    with open(temp_pdf, "wb") as file:
                        file.write(uploaded_file.getvalue())
                    
                    # Load and process PDF
                    loader = PyPDFLoader(temp_pdf)
                    docs = loader.load()
                    documents.extend(docs)
                    
                    # Clean up temp file
                    os.remove(temp_pdf)
                
                # Create vector store from documents
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
                splits = text_splitter.split_documents(documents)
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
                st.session_state["retriever"] = vectorstore.as_retriever()
                st.session_state["pdf_processed"] = True
                st.sidebar.success(f"Processed {len(documents)} documents")

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    if api_key:
        # Initialize LLM
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            streaming=True
        )
        
        # Define web search tools
        if search_mode in ["Web & Academic", "All Sources"]:
            arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
            
            wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
            wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
            
            search = DuckDuckGoSearchRun(name="Search")
            tools = [search, arxiv, wiki]
        
        with st.chat_message("assistant"):
            response_container = st.container()
            
            # Process with appropriate search method
            if search_mode == "Web & Academic":
                # Use web search agent
                st_cb = StreamlitCallbackHandler(response_container, expand_new_thoughts=False)
                search_agent = initialize_agent(
                    tools, 
                    llm, 
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True
                )
                response = search_agent.run(prompt, callbacks=[st_cb])
                
            elif search_mode == "PDF Documents" and st.session_state["pdf_processed"]:
                # Use RAG for PDF documents
                # Context-aware question reformulation prompt
                contextualize_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Reformulate the user's question to be standalone, considering the chat history context."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                
                # Create history-aware retriever
                history_aware_retriever = create_history_aware_retriever(
                    llm, 
                    st.session_state["retriever"], 
                    contextualize_prompt
                )
                
                # Answer generation prompt
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the question concisely using the context provided. Maximum three sentences. {context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                
                # Create RAG chain
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                # Create conversational chain with history
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                # Get response with history
                with response_container:
                    with st.spinner("Searching your documents..."):
                        response_obj = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}}
                        )
                        response = response_obj['answer']
                
            elif search_mode == "All Sources" and st.session_state["pdf_processed"]:
                # Combine both approaches
                # First search documents
                contextualize_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Reformulate the user's question to be standalone, considering the chat history context."),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                
                history_aware_retriever = create_history_aware_retriever(
                    llm, 
                    st.session_state["retriever"], 
                    contextualize_prompt
                )
                
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer the question concisely using the context provided. If the context doesn't contain the answer, say so. {context}"),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ])
                
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer"
                )
                
                with response_container:
                    with st.spinner("Searching your documents..."):
                        doc_response = conversational_rag_chain.invoke(
                            {"input": prompt},
                            config={"configurable": {"session_id": session_id}}
                        )
                
                    # If documents don't have the answer, use web search
                    if "don't" in doc_response['answer'] or "doesn't" in doc_response['answer'] or "not" in doc_response['answer']:
                        st.write("I couldn't find enough information in your documents. Let me search the web...")
                        st_cb = StreamlitCallbackHandler(response_container, expand_new_thoughts=False)
                        search_agent = initialize_agent(
                            tools, 
                            llm, 
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                            handle_parsing_errors=True
                        )
                        web_response = search_agent.run(prompt, callbacks=[st_cb])
                        response = f"From your documents: {doc_response['answer']}\n\nFrom web search: {web_response}"
                    else:
                        response = f"Found in your documents: {doc_response['answer']}"
            else:
                response = "Please upload PDF documents to use this search mode."
                
            # Display final response
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.error("Please enter your Groq API key in the sidebar.")
