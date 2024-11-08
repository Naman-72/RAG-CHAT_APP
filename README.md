# LLM-Powered RAG ChatBot 🤖

Welcome to the LLM-Powered RAG (Retrieval-Augmented Generation) ChatBot! This application leverages the latest in AI-driven document processing and conversational models to allow users to upload PDFs, query content, and receive precise, context-aware responses in real-time.

## 📋 Project Overview

This ChatBot application, powered by advanced large language models (LLMs), uses Retrieval-Augmented Generation to process and generate answers based on PDF documents uploaded by users. Key technologies include OpenAI's ChatGPT, Google Gemini, FAISS, and Streamlit, combining both state-of-the-art retrieval techniques and powerful language models to deliver seamless conversational experiences.

## 🔍 Features

- **PDF Upload and Processing**: Easily upload PDFs, and the app will process the document, breaking it into manageable chunks for efficient querying.
- **Embeddings and Vectorization**: Uses HuggingFace and FAISS to vectorize document content, enabling similarity search and fast response times.
- **Dual-LLM Support**: Choose between OpenAI's GPT-based models or Google Gemini models for tailored responses.
- **GPU-Acceleration**: Optimized for GPU processing (if available), ensuring smooth and scalable performance.
- **Dynamic Question Answering**: Input queries about your PDF content and receive accurate, contextually-relevant answers.

## 🛠️ Technologies Used

- **Streamlit**: Frontend framework for creating interactive web applications.
- **PyPDF2**: Library for handling PDF file reading and parsing.
- **LangChain**: Manages LLM chains, embeddings, and vector storage.
- **FAISS**: Facebook's open-source library for similarity search and clustering of dense vectors.
- **Instructor Embeddings**: Embedding model used for document vectorization.
- **Google Gemini API**: Advanced language model API for question answering.

## 🚀 Quickstart Guide

Follow these steps to set up the application on your local machine.

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- CUDA-enabled GPU (optional, recommended for performance)

### Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/llm-chatbot.git
    cd llm-chatbot
    ```

2. **Install required packages**:
    Use the `requirements.txt` file to install dependencies.
    ```bash
    pip install -r requirements.txt
    ```

3. **Set up API keys**:
    Create a `.env` file in the project root and add your OpenAI and Google Gemini API keys:
    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    GEMINI_API_KEY=your_google_gemini_api_key
    ```

### Running the Application

Run the application with the following command:
```bash
streamlit run app.py
```

### View :
![image](https://github.com/user-attachments/assets/ce6ad885-4138-4b70-8884-4bba34036bd9)



