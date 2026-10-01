import csv
import json

def convert_csv_to_jsonl(input_csv, output_jsonl):
    with open(input_csv, 'r', encoding='utf-8') as f_in, \
         open(output_jsonl, 'w', encoding='utf-8') as f_out:
        
        reader = csv.reader(f_in)
        headers = next(reader) # Skip the header row
        
        # The exact column indices based on your CSV structure:
        # English: 0 to 6
        # Hindi: 7 to 13
        # Marathi: 14 to 20
        
        count = 0
        for row in reader:
            # Skip empty lines
            if not row or len(row) < 21:
                continue
                
            # 1. English Record
            en_item = {
                "question": row[0],
                "simple_answer": row[1],
                "analogy": row[2],
                "example": row[3],
                "misconception": row[4],
                "source": row[5],
                "difficulty": row[6],
                "language": "English"
            }
            
            # 2. Hindi Record
            hi_item = {
                "question": row[7],
                "simple_answer": row[8],
                "analogy": row[9],
                "example": row[10],
                "misconception": row[11],
                "source": row[12],
                "difficulty": row[13],
                "language": "Hindi"
            }
            
            # 3. Marathi Record
            mr_item = {
                "question": row[14],
                "simple_answer": row[15],
                "analogy": row[16],
                "example": row[17],
                "misconception": row[18],
                "source": row[19],
                "difficulty": row[20],
                "language": "Marathi"
            }
            
            # Write to JSONL
            # ensure_ascii=False is CRITICAL so Devanagari text saves correctly 
            # instead of turning into \u0935\u093f... Unicode blocks
            f_out.write(json.dumps(en_item, ensure_ascii=False) + "\n")
            f_out.write(json.dumps(hi_item, ensure_ascii=False) + "\n")
            f_out.write(json.dumps(mr_item, ensure_ascii=False) + "\n")
            
            count += 3 # We generated 3 JSONL rows for every 1 CSV row
            
        print(f"Success! Generated {count} total rows in {output_jsonl}")

if __name__ == "__main__":
    convert_csv_to_jsonl("multilingual_finance.csv", "indic_translated.jsonl")