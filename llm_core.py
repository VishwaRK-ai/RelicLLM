"""
Shared logic module for RelicLLM.
Extracted from app.py to be reused by both the Streamlit UI and the FastAPI endpoints.

Exposes:
  - load_llm()          -> (model, tokenizer, device)
  - load_rag_db()       -> Chroma vector store
  - COMPANY_TICKERS     -> dict mapping friendly names to NSE tickers
  - get_stock_data(name) -> str of live market context or ""
  - get_rag_context(query, k=3) -> str of retrieved chunks
  - generate_response(system, user, ...) -> str model output
"""

import torch
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from live_stock_api import get_live_stock_data

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER_PATH = "./bharat_finance_model"
CHROMA_PERSIST_DIR = "./chroma_db"

# Company name -> NSE ticker suffix mapping
COMPANY_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "HDFC": "HDFCBANK.NS",
    "SBI": "SBIN.NS",
    "INFOSYS": "INFY.NS",
}


# ---------------------------------------------------------------------------
# Cached resource loaders
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def load_llm():
    """Load the Qwen-2.5-7B-Instruct base model with fine-tuned LoRA adapters."""
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    return model, tokenizer, device


@functools.lru_cache(maxsize=1)
def load_rag_db():
    """Load the ChromaDB vector store with BGE-M3 embeddings."""
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_stock_data(company_name: str) -> str:
    """
    Look up the ticker for *company_name* and return live market context,
    or an empty string if the company is not in the known list.
    """
    for company, ticker in COMPANY_TICKERS.items():
        if company.lower() in company_name.lower():
            return get_live_stock_data(ticker)
    return ""


def get_rag_context(query: str, k: int = 3) -> str:
    """Retrieve *k* relevant chunks from the ChromaDB knowledge base."""
    db = load_rag_db()
    docs = db.similarity_search(query, k=k)
    return "\n".join([doc.page_content for doc in docs])


def generate_response(
    system_instruction: str,
    user_query: str,
    context: str = "",
    live_context: str = "",
    max_new_tokens: int = 400,
    temperature: float = 0.3,
) -> str:
    """
    Run the LLM pipeline and return the generated response string.

    Parameters
    ----------
    system_instruction: str
        The system prompt that defines the assistant's persona.
    user_query: str
        The raw user message.
    context: str
        Optional RAG-retrieved document context to inject.
    live_context: str
        Optional live stock market data to inject.
    max_new_tokens: int
        Maximum number of new tokens to generate.
    temperature: float
        Sampling temperature.
    """
    model, tokenizer, device = load_llm()

    injected = f"Official Document Context:\n{context}\n{live_context}".strip()

    messages = [
        {"role": "system", "content": system_instruction},
    ]
    if injected:
        messages.append({"role": "user", "content": injected})
    messages.append({"role": "user", "content": user_query})

    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        repetition_penalty=1.1,
    )

    response = tokenizer.batch_decode(
        outputs[:, inputs.shape[1]:], skip_special_tokens=True
    )[0]

    return response
