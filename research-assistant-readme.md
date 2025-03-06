# Enhanced Research Assistant

A versatile Streamlit application that combines web search, academic sources, and PDF document search to create a comprehensive research assistant.

## Features

- **Multi-Source Search**: Search the web, academic repositories, and your own documents from one interface
- **Web Search Tools**: 
  - DuckDuckGo for general web searching
  - Wikipedia for encyclopedia knowledge
  - ArXiv for academic papers and research
- **PDF Document Processing**: 
  - Upload and search through multiple PDF files
  - RAG implementation with semantic search
  - Conversation memory for contextual follow-up questions
- **Flexible Search Modes**:
  - Web & Academic: Search online sources only
  - PDF Documents: Search only through uploaded documents
  - All Sources: Comprehensive search across all available resources
- **Interactive Interface**:
  - Chat-style conversation
  - Real-time agent thought process visualization
  - Streaming responses
- **Model Selection**: Choose from multiple Groq LLM models

## Requirements

- Python 3.8+
- Streamlit
- LangChain
- Groq API key
- HuggingFace API token (for embeddings)
- Internet connection (for web search features)

## Installation

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install streamlit langchain langchain-chroma langchain-groq langchain-huggingface python-dotenv
   ```

3. **Create a `.env` file with your API keys**
   ```
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   ```

## Running the Application

```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.

## Usage Guide

1. **Setup**:
   - Enter your Groq API key in the sidebar
   - Select your preferred LLM model
   - Choose a search mode based on your needs

2. **For PDF Search**:
   - Upload one or more PDF files
   - Enter a session ID (or use the default)
   - Wait for the "Processed X documents" success message

3. **Asking Questions**:
   - Type your query in the chat input box
   - The assistant will search the appropriate sources based on your selected mode
   - For "All Sources" mode, it will first check your documents and then search the web if needed

4. **Tips for Best Results**:
   - Be specific in your questions
   - For document search, use terminology that appears in your PDFs
   - Large PDFs may take longer to process initially
   - Web search works best for factual, current information

## Architecture

The application integrates two main components:

1. **Web Search Agent**:
   - Uses LangChain's agent framework with ZERO_SHOT_REACT_DESCRIPTION
   - Integrates DuckDuckGo, Wikipedia, and ArXiv tools
   - Shows thought process through StreamlitCallbackHandler

2. **RAG PDF Search**:
   - Processes PDFs into chunks with RecursiveCharacterTextSplitter
   - Embeds content using HuggingFace embeddings
   - Stores vectors in Chroma vector database
   - Uses history-aware retrieval to maintain conversation context

## Customization

- Add more search tools by extending the tools list
- Modify chunk size and overlap for different document types
- Adjust system prompts to change response style
- Change embedding models for different languages or domains

## Limitations

- Requires API keys for full functionality
- PDF processing may be memory-intensive for very large documents
- Search results are limited by the quality of the search tools
- Response quality depends on the selected LLM model