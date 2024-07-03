# RAG-Generative-AI-Chatbot-PDF

This repository contains a Streamlit-based application that integrates Google Generative AI for building a chatbot. The chatbot is enhanced with RAG (Retrieval-Augmented Generation) capabilities and allows users to upload PDF files for semantic similarity search and context retrieval.

## Features

- **Chatbot Interface**: A user-friendly chatbot interface using Streamlit.
- **PDF Upload**: Upload PDF files to extract and split text.
- **Vector Database**: Create a FAISS vector database from the extracted text.
- **Semantic Similarity Search**: Use Google Generative AI embeddings for semantic similarity search within the PDF content.
- **Chat History Management**: Clear chat history and delete the vector database with a click of a button.

## Functions

### `extract_from_pdf(pdf_path)`
Extracts text from a PDF file.

### `google_pdf_gemini_embedding(text, type)`
Gets embeddings from Google Generative AI.

### `create_vector_db(texts)`
Creates a FAISS vector database from the provided texts.

### `get_similar_context(v_db, v_user, n)`
Fetches similar context from the vector database.

### `get_response(query)`
Gets a response from the Google Generative AI model.

## How to Run the Project

### Prerequisites

- Python 3.7 or later
- Pip package manager

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/RAG-Generative-AI-Chatbot-PDF.git
    cd RAG-Generative-AI-Chatbot-PDF
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up your environment variables:
    - Create a `.env` file in the root directory.
    - Add your Google API key to the `.env` file:
        ```
        GOOGLE_API_KEY=your_google_api_key
        ```

### Running the Application

To start the Streamlit application, run the following command:
```bash
streamlit run app.py
