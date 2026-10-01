import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from live_stock_api import get_live_stock_data

def run_cli_test():
    print("\n--- 1. LOADING AI BRAIN ---")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Load base model and tokenizer
    base_model_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        torch_dtype=torch.float16, 
        device_map=device,
        low_cpu_mem_usage=True
    )
    
    # Attach your trained LoRA adapters
    model = PeftModel.from_pretrained(base_model, "./bharat_finance_model")
    print("Model Loaded Successfully!")

    print("\n--- 2. LOADING RAG KNOWLEDGE ---")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    print("ChromaDB Loaded Successfully!")

    # The test question
    test_question = "What is the risk of investing in RELIANCE right now?"
    print(f"\n--- 3. TESTING QUERY: '{test_question}' ---")

    # A. Fetch RAG Context
    print("\nFetching RAG Context...")
    docs = vector_db.similarity_search(test_question, k=3)
    rag_context = "\n".join([doc.page_content for doc in docs])
    for i, doc in enumerate(docs):
        print(f"  Chunk {i+1} Source: {doc.metadata.get('source', 'Unknown')}")

    # B. Fetch Live API Data
    print("\nFetching Live API Data...")
    live_context = ""
    if "RELIANCE" in test_question:
        live_context = get_live_stock_data("RELIANCE.NS")
        print(f"  API Response: {live_context.strip()}")

    # C. LLM Synthesis
    print("\nSynthesizing Final Answer (Generating...)\n")
    system_instruction = (
        "You are BharatFinanceEdu, an expert Indian financial educator. "
        "Use the provided context to answer. Keep it simple, use analogies, "
        "and dispel misconceptions. Always respond in the language asked."
    )
    
    injected_context = f"Official Document Context:\n{rag_context}\n{live_context}"
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"{injected_context}\n\nUser Question: {test_question}"}
    ]
    
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
    
    print("="*60)
    print(response_text)
    print("="*60)

if __name__ == "__main__":
    run_cli_test()