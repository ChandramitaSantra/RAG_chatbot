# Document Processing and Chat Application

## Overview

This Flask application allows users to upload and process PDF and DOCX documents. It generates vector representations of the documents and enables a conversational chat interface to query the processed documents based on asset IDs.

## Features

- Upload PDF and DOCX files.
- Process documents to extract text and create vector embeddings.
- Start a chat session based on the selected asset ID.
- Query the document through a conversational interface and receive responses.

## Requirements

Make sure you have the following Python packages installed:

- `Flask==2.3.2`
- `python-dotenv==1.0.0`
- `PyPDF2==3.0.1`
- `python-docx==0.8.11`
- `langchain==0.0.16`
- `huggingface-hub==0.16.4`

You can install these dependencies using:

```bash
pip install -r requirements.txt


Setup
Clone the repository:
git clone <repository-url>
cd <repository-directory>


Create a virtual environment:
python -m venv venv
Activate the virtual environment:

On Windows:
venv\Scripts\activate
On macOS/Linux:
source venv/bin/activate


Install the dependencies:
pip install -r requirements.txt

Create a .env file in the root directory for hugging face api .
Structure : HUGGINGFACEHUB_API_TOKEN = "your_apitoken"

Run the Flask application:
python app.py

API Endpoints
POST /api/documents/process
Description: Processes uploaded documents and generates vector representations.

Request:
files (form-data): List of PDF or DOCX files to process.
Response:

asset_ids (JSON): A dictionary mapping asset IDs to filenames.
POST /api/chat/start
Description: Starts a chat session for the specified asset ID.

Request:
asset_id (JSON): The ID of the asset to start a chat session for.
Response:
message (JSON): Confirmation message and the selected asset ID.
POST /api/chat/message
Description: Sends a message to the chat session and receives a response.

Request:
message (JSON): The user's message.
Response:

response (JSON): The generated response based on the chat history and asset data.#   R A G _ c h a t b o t 
 
 
