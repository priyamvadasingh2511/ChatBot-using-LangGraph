# AI Chatbot using LangGraph

This project is an AI-powered chatbot built using LangGraph, LangChain, Streamlit, and OpenRouter.

The chatbot supports conversational memory, tool calling, Retrieval-Augmented Generation (RAG), and PDF-based question answering using vector search.

## Features

* Conversational AI chatbot
* LangGraph-based workflow orchestration
* Persistent chat memory using SQLite
* Streaming responses
* Tool calling support
* Web search integration
* Stock price lookup tool
* Calculator tool
* PDF upload and question answering
* Retrieval-Augmented Generation (RAG)
* FAISS vector database integration
* Thread-specific document retrieval

## Tech Stack

* Python
* LangGraph
* LangChain
* Streamlit
* OpenRouter API
* FAISS
* SQLite
* HuggingFace Embeddings


## Installation

Clone the repository:

```bash
git clone <repository-url>
cd ChatBot
```

Create and activate virtual environment.

### Mac/Linux

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key
ALPHA_VANTAGE_API_KEY=your_api_key
```

## Run the Application

Start the Streamlit frontend:

```bash
streamlit run Frontend/frontend.py
```

## Supported Capabilities

### Chat Memory

The chatbot uses LangGraph checkpointing with SQLite to maintain persistent conversation history across chat threads.

### Tool Calling

The assistant can use external tools including:

* Web search
* Calculator
* Stock price lookup

### RAG Pipeline

Users can upload PDF documents and ask questions based on the uploaded content.

The application:

* extracts text from PDFs
* splits documents into chunks
* generates embeddings using HuggingFace models
* stores embeddings in FAISS
* retrieves relevant chunks during conversations

### Thread-Based Retrieval

Each chat thread maintains its own retriever and uploaded document context.

## Learning Goals

This project was built to practice:

* LangGraph state management
* AI workflow orchestration
* Conversational memory systems
* Tool calling with LLMs
* Retrieval-Augmented Generation (RAG)
* Vector databases and embeddings
* Stateful chatbot architectures

## Notes

* SQLite database files are excluded using `.gitignore`
* API keys are managed using environment variables
* The project is intended for learning and experimentation with LangGraph and LLM application development
