

**Overview
This project processes PDFs and web content by extracting text and images, embedding the text using HuggingFaceEmbeddings, and building a searchable FAISS vector store. Users can query the processed data using the ChatGroq LLM (llama-3.1-70b-versatile).

Key Features
PDF Processing: Extracts text and images from uploaded PDFs.
Web Scraping: Extracts paragraph text from specified website URLs.
Searchable FAISS Index: Creates a FAISS vector store for efficient retrieval.
ChatGroq Integration: Handles queries using Groq’s language model.
Setup Instructions
Install dependencies: Run pip install -r requirements.txt (ensure all listed libraries are installed).
Input PDFs or URLs: Upload PDF files or provide website URLs when prompted.
Query Data: Ask questions to retrieve relevant information.
Usage
Run the script in your Python environment. Follow prompts to upload PDFs or input URLs, and query extracted content seamlessly.
