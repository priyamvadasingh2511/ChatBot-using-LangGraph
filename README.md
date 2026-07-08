# ChatBot-using-LangGraph

A conversational chatbot built with LAngGraph, capable of answering questions using an uploaded PDF (RAG), web search, live stock prices, and basic arithmetic — with persistent chat history across sessions.

## Features

- **PDF Question Answering (RAG)** — upload a PDF per chat thread; the bot chunks, embeds (`sentence-transformers/all-MiniLM-L6-v2`), and indexes it with FAISS, then retrieves relevant passages to answer questions about it.
- **Tool-using agent** — the LLM can call:
  - `rag_tool` — retrieve context from the uploaded PDF for the current thread
  - `search_tool` — general web search via DuckDuckGo
  - `get_stock_price` — live stock quotes via Alpha Vantage
  - `calculator` — basic arithmetic (add/sub/mul/div)
- **Per-thread document isolation** — each chat thread has its own PDF and retriever; documents don't leak across conversations.
- **Persistent chat history** — conversations are checkpointed to SQLite (`chat_history.db`) via `langgraph-checkpoint-sqlite`, so threads survive restarts.
- **Streamlit frontend** — simple chat UI for interacting with the bot and managing threads.

## Architecture

```
Frontend.py   → Streamlit UI: chat interface, PDF upload, thread management
Backend.py    → LangGraph definition: state, tools, chatbot node, graph compilation
eval/         → LangSmith-based evaluation harness (see below)
```

The graph is a simple loop: `chatbot` node → (conditionally) `tools` node → back to `chatbot`, until the model produces a final answer with no further tool calls. State is a message list, threaded per conversation via `thread_id`.

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/priyamvadasingh2511/ChatBot-using-LangGraph.git
   cd ChatBot-using-LangGraph
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the repo root with:
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_key
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_PROJECT=chatbot
   ```
   `LANGCHAIN_*` vars are only needed if you want tracing/evaluation via LangSmith.

3. Run the app:
   ```bash
   streamlit run Frontend.py
   ```

## Model

Uses `openrouter/free` via OpenRouter as the LLM (free tier). Because it's a free/shared model, response quality and latency can vary — the [evaluation harness](#evaluation) below is set up specifically to measure and track this.

## Evaluation

The `eval/` folder contains a small, fixed regression-test suite for the RAG pipeline, built on LangSmith:

- `eval/docs/` — two sample PDFs (a contract and a quarterly report) used as fixed test documents. These are not meant to represent every kind of PDF a user might upload — they're a stable baseline to catch regressions when the pipeline changes.
- `eval/dataset.py` — uploads a hand-written set of question/answer pairs (based on the sample PDFs) to a LangSmith dataset.
- `eval/run_eval.py` — runs the chatbot against that dataset and scores each answer with an LLM-as-judge correctness evaluator, repeating each question 3x to average out variance from the free model.

To run it:
```bash
python eval/dataset.py    # one-time: uploads the golden dataset
python eval/run_eval.py   # runs and scores the chatbot against it
```
Results appear in the LangSmith UI under **Datasets & Experiments → pdf-chatbot-eval-v1**, as a scored experiment you can compare across runs (e.g. before/after changing the prompt, chunking, or model).

## Known limitations

- Retrieval quality depends on the free embedding model and default FAISS similarity search (`k=4`); no re-ranking or hybrid search.
- Tool selection (e.g. whether the LLM calls `rag_tool` vs. answering directly) depends on the LLM correctly following the system prompt — this can be inconsistent with a free-tier model, and is one of the things the eval harness is designed to surface.
- Chat history persistence uses a local SQLite file; not suitable for multi-instance/production deployments as-is.
