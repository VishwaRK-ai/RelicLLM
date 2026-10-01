import os
import re
import fitz  # PyMuPDF
from glob import glob

def clean_text(text):
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove purely numeric lines or special character spam (tables/graphs)
    lines = text.split('\n')
    valid_lines = [l for l in lines if len(l.strip()) > 20 and sum(c.isalpha() for c in l) > 10]
    return " ".join(valid_lines)

def extract_pdfs():
    pdf_files = glob("rag_raw_pdfs/*.pdf")
    print(f"Found {len(pdf_files)} PDFs.")

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path).replace('.pdf', '.txt')
        out_path = os.path.join("rag_clean_text", filename)
        
        print(f"Extracting {pdf_path}...")
        doc = fitz.open(pdf_path)
        full_text = []
        
        # Skip first 3 and last 3 pages (usually TOC and Index)
        start_page = min(3, len(doc))
        end_page = max(start_page, len(doc) - 3)
        
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            text = page.get_text("text")
            full_text.append(clean_text(text))
            
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
            
    print("Extraction complete. Text saved to rag_clean_text/")

if __name__ == "__main__":
    extract_pdfs()