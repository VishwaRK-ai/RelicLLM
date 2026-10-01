import json
from datasets import load_dataset

COMPLEX_JARGON = [
    "ebitda", "derivative", "tranche", "arbitrage", "hedge fund", 
    "liquidation", "options trading", "swap", "amortization", 
    "merger", "acquisition", "corporate bond", "macroeconomics"
]

def is_simple(text):
    if not text:
        return False
    text_lower = text.lower()
    return not any(jargon in text_lower for jargon in COMPLEX_JARGON)

def process_datasets():
    print("Downloading FinGPT FiQA dataset from HuggingFace...")
    
    try:
        # Using the stable, actively maintained FinGPT dataset
        dataset = load_dataset("FinGPT/fingpt-fiqa_qa")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    output_file = "fiqa_filtered.jsonl"
    valid_count = 0
    target_count = 1000  
    
    with open(output_file, "w", encoding="utf-8") as f:
        for split in dataset.keys():
            for item in dataset[split]:
                # FinGPT often uses 'input' and 'output', but we check standard names too just in case
                question = item.get('input') or item.get('question', '')
                answer = item.get('output') or item.get('answer', '')
                
                # Convert to string just in case the dataset has nested lists
                question = str(question).strip()
                answer = str(answer).strip()
                
                if not question or not answer:
                    continue
                    
                if is_simple(question) and is_simple(answer):
                    formatted_item = {
                        "question": question,
                        "simple_answer": answer,
                        "analogy": "",           
                        "example": "",           
                        "misconception": "",     
                        "difficulty": "intermediate", 
                        "language": "English",
                        "source": "FiQA"
                    }
                    
                    f.write(json.dumps(formatted_item) + "\n")
                    valid_count += 1
                    
                    if valid_count >= target_count:
                        break
            if valid_count >= target_count:
                break
                
    print(f"Success! Saved {valid_count} filtered FiQA examples to {output_file}")

if __name__ == "__main__":
    process_datasets()