# 📈 BharatFinanceEdu (RelicLLM)

BharatFinanceEdu is an AI-powered Indian financial educator. This repository contains the complete pipeline to fine-tune a 7B LLM (Qwen-2.5) on educational financial data, augment it with an offline RAG vector database of SEBI/RBI guidelines, and connect it to live NSE stock market data.

---

## 📁 Key Files & Workflow

The project is structured into four main phases:

### 1. Data Preparation & Cleaning
* **`1_extract_pdfs.py`**: Extracts text from raw regulatory PDFs (`rag_raw_pdfs/`), strips noise/TOC/indexes, and saves clean files to `rag_clean_text/`.
* **`2_generate_dataset.py`**: Reads cleaned text and uses `gpt-4o-mini` with Pydantic Structured Outputs to auto-generate Q&A pairs (Explanation, Analogy, Example, Misconception). Outputs `bharat_edu_base.jsonl`.
* **`3_filter_fiqa.py`**: Downloads the FiQA dataset from Hugging Face, removes corporate Wall Street jargon (EBITDA, swaps, etc.), and standardizes the schema.
* **`6_csv_to_jsonl.py`**: Parses the multi-language translated spreadsheet (`multilingual_finance.csv`) and formats Hindi and Marathi rows into valid JSONL.
* **`final_train.jsonl`**: The merged master dataset (English + FiQA + Hindi + Marathi) ready for fine-tuning.

---

### 2. Model Training
* **`relic1.ipynb`**: Google Colab training notebook using **Unsloth + QLoRA**. Fine-tunes `Qwen2.5-7B-Instruct-bnb-4bit` on `final_train.jsonl` within 150 steps on a free T4 GPU, tests inference across all 3 languages, and exports the LoRA adapters (`bharat_finance_model/`).

---

### 3. RAG Knowledge Base & Market API
* **`7_build_rag.py`**: Chunks documents from `rag_clean_text/` and embeds them using `BAAI/bge-m3` into a local **ChromaDB** store (`chroma_db/`). This acts as the offline, authoritative ground-truth knowledge base.
* **`live_stock_api.py`**: Agentic utility using `yfinance` to fetch live NSE stock prices, 52-week ranges, and risk profiles (Beta metric) for context injection.

---

### 4. Testing & Frontend
* **`test_pipeline.py`**: CLI test harness. Verifies the entire end-to-end loop (ChromaDB retrieval + live stock fetching + model generation) directly in the terminal before running the UI.
* **`app.py`**: Streamlit chatbot interface connecting the fine-tuned model, ChromaDB retriever, and live NSE stock API.
* **`llm_core.py`**: Shared Python module extracted from `app.py`. Centralizes model loading, RAG retrieval, live stock data, and response generation so both the Streamlit UI and FastAPI endpoints reuse the same logic.
* **`api.py`**: FastAPI application exposing two sets of endpoints — an **investment-eligibility quiz** and an **AI financial assistant** — for integration with any frontend (React, Vue, Flutter, etc.).

---

## API Endpoints (FastAPI)

Run the API server:
```bash
python -m uvicorn api:app --reload --port 8000
# or  python api.py
```

Interactive docs are available at `http://localhost:8000/docs`.

### Quiz Endpoints — `/quiz`

The quiz flow has three steps:

---

#### `GET /quiz` — List available stocks

```http
GET /quiz
```

**Response:**
```json
{
  "message": "Welcome to the BharatFinanceEdu Investment Quiz! Choose a stock...",
  "available_stocks": {
    "RELIANCE": "RELIANCE",
    "TATA MOTORS": "TATA MOTORS",
    "HDFC": "HDFC",
    "SBI": "SBI",
    "INFOSYS": "INFOSYS"
  }
}
```

---

#### `POST /quiz` — Generate 5 multiple-choice questions for a stock

```http
POST /quiz
Content-Type: application/json

{"stock_name": "RELIANCE"}
```

**Response:**
```json
{
  "quiz_id": "quiz_a1b2c3d4",
  "stock_name": "RELIANCE",
  "questions": [
    {
      "question": "What is Reliance's dividend payout trend in FY2024?",
      "options": {
        "A": "15%",
        "B": "25%",
        "C": "35%",
        "D": "45%"
      }
    }
    // ... 4 more questions
  ]
}
```

---

#### `POST /quiz/submit` — Submit answers and receive a score

```http
POST /quiz/submit
Content-Type: application/json

{
  "quiz_id": "quiz_a1b2c3d4",
  "answers": ["A", "C", "B", "D", "A"]
}
```

**Response:**
```json
{
  "score": 4,
  "total": 5,
  "eligible": true,
  "correct_answers": ["A", "C", "B", "D", "C"],
  "feedback": "Good job! You understand the basics. Trade responsibly."
}
```

> **Eligibility rule:** Score ≥ 60% → `eligible: true`

---

### Assistant Endpoints — `/assistant`

An AI assistant bubble for answering basic finance questions.

---

#### `POST /assistant` — Send a message and get a reply

```http
POST /assistant
Content-Type: application/json

{
  "message": "Should I invest in Reliance right now?",
  "session_id": "user_123",
  "chat_history": []
}
```

**Response:**
```json
{
  "reply": "Based on the latest data, Reliance (RELIANCE.NS) is currently trading at ₹2,450...",
  "sources": ["SEBI/RBI Knowledge Base (ChromaDB)"],
  "live_data": true
}
```

| Field | Type | Description |
|-------|------|-------------|
| `reply` | `string` | The assistant's response text |
| `sources` | `array[string]` | Knowledge base sources consulted |
| `live_data` | `boolean` | Whether live stock data was used |

---

#### `GET /assistant/history/{session_id}` — Retrieve chat history

```http
GET /assistant/history/user_123
```

**Response:**
```json
{
  "session_id": "user_123",
  "history": [
    {"role": "user", "content": "What is a mutual fund?"},
    {"role": "assistant", "content": "A mutual fund is..."}
  ]
}
```

---

## (Local Setup)

### 1. Install Dependencies
```bash
pip install peft transformers langchain langchain-community langchain-huggingface chromadb sentence-transformers yfinance streamlit gTTS playsound3 fastapi uvicorn

# Quick test
python test_pipeline.py

# Run the Streamlit UI
streamlit run app.py

# Run the FastAPI server (alternative to Streamlit)
python api.py
# or
python -m uvicorn api:app --reload --port 8000
```
