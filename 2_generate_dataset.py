import os
from glob import glob
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic forces the API to strictly follow this schema
class FinanceQAPair(BaseModel):
    question: str = Field(description="A clear question about a financial concept.")
    simple_answer: str = Field(description="A simple, jargon-free answer.")
    analogy: str = Field(description="A relatable real-world analogy.")
    example: str = Field(description="A short, practical example.")
    misconception: str = Field(description="A common misconception about this topic.")
    difficulty: str = Field(default="beginner")
    language: str = Field(default="English")
    source: str = Field(description="The institution name (e.g., SEBI, RBI, NCFE).")

class DatasetBatch(BaseModel):
    examples: list[FinanceQAPair]

def generate_dataset():
    files = glob("rag_clean_text/*.txt")
    output_file = "bharat_edu_base.jsonl"
    
    with open(output_file, "a", encoding="utf-8") as f:
        for file_path in files:
            source_name = os.path.basename(file_path).split('_')[0].upper()
            
            with open(file_path, "r", encoding="utf-8") as doc:
                words = doc.read().split()
            
            # Chunk into ~1500 words
            chunks = [" ".join(words[i:i + 1500]) for i in range(0, len(words), 1500)]
            
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) < 50: 
                    continue
                    
                print(f"Processing chunk {i+1}/{len(chunks)} from {source_name}...")
                
                try:
                    response = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system", 
                                "content": f"Extract 3 to 5 fundamental beginner financial concepts from the text. Format them exactly as requested. The source is {source_name}. If the text contains no educational concepts, return an empty list."
                            },
                            {"role": "user", "content": f"Text:\n{chunk}"}
                        ],
                        response_format=DatasetBatch
                    )
                    
                    for item in response.choices[0].message.parsed.examples:
                        f.write(item.model_dump_json() + "\n")
                        
                except Exception as e:
                    print(f"Error on chunk {i+1}: {e}")

if __name__ == "__main__":
    generate_dataset()