"""
Runs Backend.py's LangGraph `chat` graph against the golden dataset built by
dataset.py, and scores answers with an LLM-as-judge correctness evaluator
via LangSmith's evaluate().

Usage:
    python eval/run_eval.py

Requires (same env vars you already use for tracing, plus these installed):
    pip install langsmith langchain langchain-openai
    LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2=true, LANGCHAIN_ENDPOINT
    OPENAI_API_KEY or OPENROUTER_API_KEY (whatever Backend.py already needs)

Notes on the free-model variance issue:
- `num_repetitions=3` runs each question 3x so a single bad/rate-limited
  generation doesn't look like a permanent regression -- look at the
  aggregated pass rate across repetitions, not any single row.
- `run_bot` has basic retry/backoff since OpenRouter's free tier can 429.
- Retrieval (FAISS) is deterministic; only the generation step varies.
  If you see inconsistent scores, check whether it's the same question
  each time (generation variance) or different questions (something else).
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backend import chat, retreiver_doc  # noqa: E402
from langsmith.evaluation import evaluate  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

# Separate judge model call (kept plain OpenAI-compatible via OpenRouter, same
# creds as Backend.py) used only to SCORE answers -- not part of the chatbot
# being tested.
_judge_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openrouter/free",
)

DATASET_NAME = "pdf-chatbot-eval-v1"
DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

_loaded_docs = set()


def _ensure_doc_loaded(doc_name: str, thread_id: str):
    """Load the sample PDF into the retriever for this thread if not already done."""
    key = (doc_name, thread_id)
    if key in _loaded_docs:
        return
    path = os.path.join(DOCS_DIR, doc_name)
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    retreiver_doc(pdf_bytes, thread_id=thread_id, filename=doc_name)
    _loaded_docs.add(key)


def run_bot(inputs: dict) -> dict:
    """Wrapper matching the signature evaluate() expects: inputs -> outputs dict."""
    doc_name = inputs["doc"]
    question = inputs["question"]
    thread_id = f"eval-{doc_name}"

    _ensure_doc_loaded(doc_name, thread_id)

    last_err = None
    for attempt in range(3):
        try:
            result = chat.invoke(
                {"messages": [("user", question)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            return {"answer": result["messages"][-1].content}
        except Exception as e:  # rate limits / transient errors on the free model
            last_err = e
            time.sleep(2 ** attempt)
    return {"answer": f"[ERROR after retries: {last_err}]"}


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    LLM-as-judge evaluator (new-style LangSmith signature).
    Scores whether the bot's answer is semantically correct vs. the
    reference answer -- NOT exact string match, so paraphrasing from the
    free model doesn't get penalized.
    """
    question = inputs.get("question", "")
    predicted = outputs.get("answer", "")
    reference = reference_outputs.get("answer", "")

    judge_prompt = (
        "You are grading a chatbot's answer against a reference answer.\n\n"
        f"Question: {question}\n"
        f"Reference answer: {reference}\n"
        f"Chatbot answer: {predicted}\n\n"
        "Does the chatbot answer convey the same key facts as the reference "
        "answer (wording may differ)? Respond with exactly one word: "
        "'correct' or 'incorrect'."
    )
    try:
        judgment = _judge_llm.invoke(judge_prompt).content.strip().lower()
    except Exception as e:
        return {"key": "correctness", "score": 0, "comment": f"judge error: {e}"}

    score = 1 if "incorrect" not in judgment and "correct" in judgment else 0
    return {"key": "correctness", "score": score, "comment": judgment}


if __name__ == "__main__":
    results = evaluate(
        run_bot,
        data=DATASET_NAME,
        evaluators=[correctness],
        experiment_prefix="rag-openrouter-free",
        num_repetitions=3,
        metadata={"model": "openrouter/free", "pipeline": "langgraph-rag"},
    )
    print("Eval run complete. View the scored experiment in the LangSmith UI "
          f"under the '{DATASET_NAME}' dataset.")