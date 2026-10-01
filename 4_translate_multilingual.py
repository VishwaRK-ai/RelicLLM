import json
from deep_translator import GoogleTranslator

def translate_dataset():
    input_file = "bharat_edu_base.jsonl"
    output_file = "indic_translated.jsonl"
    target_count = 300 
    
    # Define target languages and their ISO codes for Google Translate
    targets = {"Hindi": "hi", "Marathi": "mr", "Telugu": "te"}
    
    # Initialize translators once to save time
    translators = {
        lang: GoogleTranslator(source='en', target=code) 
        for lang, code in targets.items()
    }
    
    data_to_translate = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= target_count: break
            data_to_translate.append(json.loads(line))

    print(f"Translating {len(data_to_translate)} examples into {list(targets.keys())}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(data_to_translate):
            print(f"Translating row {i+1}/{target_count}...")
            
            for lang, translator in translators.items():
                try:
                    # Batch translate all 5 fields in one fast network request
                    texts_to_translate = [
                        item['question'], 
                        item['simple_answer'], 
                        item['analogy'], 
                        item['example'], 
                        item['misconception']
                    ]
                    
                    translated_texts = translator.translate_batch(texts_to_translate)
                    
                    # Rebuild the JSON object
                    translated_item = item.copy()
                    translated_item['question'] = translated_texts[0]
                    translated_item['simple_answer'] = translated_texts[1]
                    translated_item['analogy'] = translated_texts[2]
                    translated_item['example'] = translated_texts[3]
                    translated_item['misconception'] = translated_texts[4]
                    translated_item['language'] = lang
                    
                    # Save it (ensure_ascii=False keeps the native indic script)
                    f.write(json.dumps(translated_item, ensure_ascii=False) + "\n")
                    
                except Exception as e:
                    print(f"Error on {lang} row {i+1}: {e}")

    print("Translation complete! 900 new rows generated.")

if __name__ == "__main__":
    translate_dataset()