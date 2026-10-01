import json
import csv

input_file = "bharat_edu_base.jsonl"
output_file = "to_translate.csv"
target_rows = 300

fields = ["question", "simple_answer", "analogy", "example", "misconception", "source", "difficulty"]

with open(input_file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", newline="", encoding="utf-8") as f_out:
    
    writer = csv.DictWriter(f_out, fieldnames=fields)
    writer.writeheader()
    
    for i, line in enumerate(f_in):
        if i >= target_rows:
            break
        item = json.loads(line)
        writer.writerow({k: item.get(k, "") for k in fields})

print(f"Created {output_file} with {target_rows} rows.")