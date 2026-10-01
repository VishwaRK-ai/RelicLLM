import os
from glob import glob
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db():
    print("Loading cleaned text files...")
    file_paths = glob("rag_clean_text/*.txt")
    
    if not file_paths:
        print("Error: No .txt files found in rag_clean_text/")
        return

    documents = []
    for path in file_paths:
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            # Attach the institution name as metadata for citation (e.g., SEBI, RBI)
            source_name = os.path.basename(path).split('_')[0].upper()
            for d in docs:
                d.metadata["source"] = source_name
            documents.extend(docs)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    
    # Small chunks (500 chars) prevent the model from getting confused by too much text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} searchable chunks.")

    print("Downloading BAAI/bge-m3 embedding model (this is ~2GB and takes a minute)...")
    # bge-m3 is state-of-the-art for multilingual retrieval (Hindi/Marathi/English)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    print("Building Chroma vector database...")
    # Persist the database locally so it survives reboots
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Vector database built successfully and saved to ./chroma_db!")

if __name__ == "__main__":
    build_vector_db()