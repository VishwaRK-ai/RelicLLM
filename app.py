import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from live_stock_api import get_live_stock_data

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="BharatFinanceEdu", page_icon="📈", layout="centered")
st.title("📈 BharatFinanceEdu")
st.markdown("Your AI Financial Educator — Powered by SEBI/RBI context & Live Market Data")

# --- 2. CACHE HEAVY MODELS (So it doesn't crash on reload) ---
@st.cache_resource
def load_rag_db():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    return Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

@st.cache_resource
def load_llm():
    # Load the base Qwen model 
    # (Using 'mps' for Mac Apple Silicon acceleration, or 'cpu' if unavailable)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Load base model in lower precision to fit in Mac memory
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # Attach your fine-tuned LoRA adapters from Colab
    model = PeftModel.from_pretrained(base_model, "./bharat_finance_model")
    return model, tokenizer, device

# Load resources quietly in the background
with st.spinner("Loading AI Brain and Knowledge Base... (Takes a minute on first boot)"):
    vector_db = load_rag_db()
    model, tokenizer, device = load_llm()

# --- 3. SESSION STATE FOR CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. THE PROMPTING & INFERENCE PIPELINE ---
# Dictionary for the hackathon demo to map company names to NSE tickers
COMPANY_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TATA MOTORS": "TATAMOTORS.NS",
    "HDFC": "HDFCBANK.NS",
    "SBI": "SBIN.NS",
    "INFOSYS": "INFY.NS"
}

if user_prompt := st.chat_input("Ask a financial question or check a stock..."):
    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # 2. Fetch RAG Context (Top 3 chunks from SEBI/RBI PDFs)
            docs = vector_db.similarity_search(user_prompt, k=3)
            rag_context = "\n".join([doc.page_content for doc in docs])
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in docs]))

            # 3. Check for Live Stock request
            live_context = ""
            for company, ticker in COMPANY_TICKERS.items():
                if company.lower() in user_prompt.lower():
                    live_context = get_live_stock_data(ticker)
                    break
            
            # 4. Construct the Final Prompt 
            # This forces the model to use the retrieved context AND your fine-tuned format
            system_instruction = (
                "You are BharatFinanceEdu, an expert Indian financial educator. "
                "Use the provided context to answer. Keep it simple, use analogies, "
                "and dispel misconceptions. Always respond in the language asked."
            )
            
            injected_context = f"Official Document Context:\n{rag_context}\n{live_context}"

            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"{injected_context}\n\nUser Question: {user_prompt}"}
            ]
            
            # 5. Generate Response
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(device)
            
            outputs = model.generate(
                input_ids=inputs, 
                max_new_tokens=400, 
                temperature=0.3,
                repetition_penalty=1.1
            )
            
            response_text = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]

            # 6. Append Source Citations for the Judges
            if sources:
                response_text += f"\n\n**Sources Consulted:** {', '.join(sources)}"
            if live_context:
                response_text += "\n*(Includes live market data fetched via API)*"

            # Display the response
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})