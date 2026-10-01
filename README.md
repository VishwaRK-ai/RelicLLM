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

---

## (Local Setup)

### 1. Install Dependencies
```bash
pip install peft transformers langchain langchain-community langchain-huggingface chromadb sentence-transformers yfinance streamlit
python test_pipeline.py
streamlit run app.py
