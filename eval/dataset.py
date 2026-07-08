"""
Builds a small, fixed golden dataset in LangSmith for evaluating the
LangGraph PDF chatbot (Backend.py).

This is NOT meant to cover every PDF a real user might upload -- that's
impossible. It's a small, hand-verified regression suite built on top of
a couple of representative sample docs (a contract + a report with a table)
living in eval/docs/. Re-run this once whenever you add/change questions;
create_examples is additive, so delete the dataset in the UI first if you
want a clean rebuild.

Requires: LANGCHAIN_API_KEY (same one you already use for tracing) and
LANGCHAIN_ENDPOINT set in your environment, plus `pip install langsmith`.
"""

from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

DATASET_NAME = "pdf-chatbot-eval-v1"

# Each example says which sample doc it targets so run_eval.py knows
# which PDF to load into the retriever for that question.
EXAMPLES = [
    {
        "inputs": {"question": "What is the termination clause in this agreement?", "doc": "contract.pdf"},
        "outputs": {"answer": "Either party may terminate for convenience with 30 days written notice; "
                               "Provider may terminate immediately if Client is more than 15 days late on an invoice."},
    },
    {
        "inputs": {"question": "How much is the monthly retainer fee?", "doc": "contract.pdf"},
        "outputs": {"answer": "$6,000 per month, invoiced on the first business day of each month."},
    },
    {
        "inputs": {"question": "What law governs this agreement?", "doc": "contract.pdf"},
        "outputs": {"answer": "The laws of the State of Delaware."},
    },
    {
        "inputs": {"question": "What is the initial term length and does it renew automatically?", "doc": "contract.pdf"},
        "outputs": {"answer": "12 months, and it auto-renews for successive 12-month terms unless either "
                               "party gives 30 days written notice of non-renewal."},
    },
    {
        "inputs": {"question": "What was Q3 2026 total revenue and how much did it grow?", "doc": "report_with_tables.pdf"},
        "outputs": {"answer": "$4.2M, up 13.5-14% quarter-over-quarter from $3.7M in Q2."},
    },
    {
        "inputs": {"question": "Which revenue channel declined in Q3?", "doc": "report_with_tables.pdf"},
        "outputs": {"answer": "Retail Stores, which fell from $1.5M to $1.4M (-6.7%)."},
    },
    {
        "inputs": {"question": "How many employees did the company have at the end of Q3 2026?", "doc": "report_with_tables.pdf"},
        "outputs": {"answer": "142 full-time employees, 9 more than in Q2."},
    },
    {
        "inputs": {"question": "What is the revenue outlook for Q4 2026?", "doc": "report_with_tables.pdf"},
        "outputs": {"answer": "Expected to be in the range of $4.5M to $4.8M."},
    },
]


def build_dataset():
    client = Client()

    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists -- reusing it. "
              f"Delete it in the LangSmith UI first if you want a clean rebuild.")
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description="Golden Q&A set over 2 fixed sample PDFs (contract + quarterly report) "
                         "for regression-testing the LangGraph RAG pipeline in Backend.py.",
        )

    client.create_examples(
        inputs=[e["inputs"] for e in EXAMPLES],
        outputs=[e["outputs"] for e in EXAMPLES],
        dataset_id=dataset.id,
    )
    print(f"Uploaded {len(EXAMPLES)} examples to dataset '{DATASET_NAME}'.")


if __name__ == "__main__":
    build_dataset()