import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import (TextLoader, PyPDFLoader)

load_dotenv()


# Text Loader
def load_text_file():
    # Create a temporary text file for demonstration
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
        temp_file.write(b"Hello, this is a sample text file.\nThis file is used to demonstrate TextLoader.")
        temp_file_path = temp_file.name

    try:
        # Load the text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        # Print the loaded documents
        for doc in documents:
            print("---")
            print(doc)
            print("---")
            print(doc.page_content)
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)



# PDF Loader
def pdf_loader(pdf_path):
    try:
        # Load the PDF file using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        print(f"Loaded {len(documents)} document(s) from PDF")
        for i, doc in enumerate(documents):
            print(f"\nPage {i+1}:")
            print(f"Metadata: {doc.metadata}")
            print(f"Content length: {len(doc.page_content)} characters")

        # Print the loaded documents
        for doc in documents:
            print("---")
            print(doc)
            print("---")
            print(doc.page_content)
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")

if __name__ == "__main__":
    pdf_loader("./docs/langchain_demo.pdf")
