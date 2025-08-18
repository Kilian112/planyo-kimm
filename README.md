## KIMM


Overview

KIMM is a Flask-based web application that utilizes RAG architecture to provide accurate and informative responses to user queries. The application leverages the power of Pinecone, OpenAI, and Redis to efficiently process and store large volumes of knowledge.


Key Features

  -  KIMM combines the strengths of language models and knowledge graphs to provide accurate and informative responses to user queries.
  -  Pinecone Integration: KIMM integrates with Pinecone to efficiently store and retrieve large volumes of knowledge.
  -  OpenAI Integration: KIMM utilizes OpenAI's language model to generate human-like responses to user queries.
  -  Redis Integration: KIMM uses Redis to cache frequently accessed knowledge and improve response times.
  -  User-Friendly Interface: KIMM features a user-friendly interface that allows users to easily upload documents, select namespaces, and submit queries.

System Requirements

  -  Python 3.8+: KIMM requires Python 3.8 or later to run.
  - Flask 2.2.5: KIMM uses Flask 2.2.5 as its web framework.
  -  Pinecone 5.0.0+: KIMM requires Pinecone 5.0.0 or later to run.
  -  OpenAI 0.28.0: KIMM uses OpenAI 0.28.0 to generate human-like responses.
  -  Redis: KIMM uses Redis to cache frequently accessed knowledge.

Setup and Installation

  -  Clone the repository: git clone https://github.com/your-username/kimm.git
  -  Install dependencies: pip install -r requirements.txt
  -  Set environment variables: export API_KEY=your-api-key (replace with your actual API key)
  -  Run the application: gunicorn -c gunicorn_config.py app:app

Usage

  -  Upload a document: Click on the "Upload Document" button and select a PDF, DOCX, or TXT file.
  -  Select a namespace: Click on the "Knowledge Base" or "Previous Uploads" tab and select a namespace.
  -  Submit a query: Type a query in the input box and click the "Submit" button.


