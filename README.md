# Eval harness for ChatBot-using-LangGraph

## Files
- `docs/contract.pdf` — sample service agreement (fixed golden doc)
- `docs/report_with_tables.pdf` — sample quarterly report with a table (fixed golden doc)
- `dataset.py` — one-time script: uploads 8 hand-written Q&A pairs (based on the two docs above) to a LangSmith dataset called `pdf-chatbot-eval-v1`
- `run_eval.py` — runs `Backend.py`'s `chat` graph against that dataset and scores answers with an LLM-as-judge correctness evaluator, 3x per question to average out free-model variance

## Confirmed against your actual `Backend.py`
- Import: `from Backend import chat, retreiver_doc`
- `retreiver_doc(file_bytes, thread_id=..., filename=...)` — matches
- `chat.invoke({"messages": [("user", question)]}, config={"configurable": {"thread_id": thread_id}})` — matches
- Answer extraction: `result["messages"][-1].content` — matches

## ⚠️ One thing worth knowing before you run this
Your `chatbot` node doesn't call `rag_tool` directly — the LLM decides whether to call it, based on the system prompt instructing it to use `rag_tool` for PDF questions and pass along the thread_id. This means:
- If the free model ignores the instruction, misformats the tool call, or forgets to pass `thread_id`, `rag_tool` either won't get called or will return the "no document indexed" error — and the eval will correctly score that as wrong, which is useful signal, not a bug in the eval.
- If you see several questions scoring 0 with an answer like "please upload a PDF" even though the doc was loaded, that's telling you the tool-calling reliability of the free model is the actual bottleneck — worth knowing on its own, separate from RAG quality.
- Also note: `CONN = sqlite3.connect("chat_history.db", ...)` persists across runs. Rerunning `run_eval.py` reuses the same `thread_id`s (`eval-contract.pdf`, `eval-report_with_tables.pdf`), so prior conversation history in that thread carries over. If you want a truly clean slate each run, delete `chat_history.db` before evaluating, or make the thread_id unique per run (e.g. append a timestamp).

## Setup

```bash
pip install langsmith langchain langchain-openai
```

Set these env vars (same ones you already use for tracing today — nothing new on the LangSmith side):
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your_key
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com   # default, only needed if self-hosted
export OPENROUTER_API_KEY=your_key   # whatever Backend.py already needs to run
```

## Run it

```bash
# 1. Build/upload the dataset (one-time, or after you edit questions in dataset.py)
python eval/dataset.py

# 2. Run the evaluation
python eval/run_eval.py
```

Then check the LangSmith UI → **Datasets & Experiments** → `pdf-chatbot-eval-v1` → you'll see a new experiment row (`rag-openrouter-free-...`) with per-question correctness scores. Click any row to jump into its full trace, same as your existing tracing setup.

## Extending this
- Add more `(doc, question, answer)` triples to `EXAMPLES` in `dataset.py` as you find real failure cases worth locking in as regression tests.
- To add retrieval-quality scoring (not just final-answer correctness), have `run_bot` also return the `context`/`metadata` your `rag_tool` produces, and add a second evaluator that checks whether the retrieved chunk actually contains the answer.
- Re-run `run_eval.py` any time you change chunking, the prompt, or swap models — compare the new experiment against the previous one in the UI to catch regressions.
