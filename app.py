from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import docx
import os
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub
import uuid

app = Flask(__name__)
load_dotenv()

app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global session state
session_state = {
    "vectorstores": {},
    "asset_ids": {},
    "chat_histories": {},
    "processComplete": False,
    "selected_asset_id": None
}

@app.route('/')
def index():
    return render_template('index.html')

def save_file(uploaded_file):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(filepath)
    return filepath

@app.route('/api/documents/process', methods=['POST'])
def process_documents():
    uploaded_files = request.files.getlist('files')
    files_text = get_files_text(uploaded_files)
    
    # Reset session state
    session_state["vectorstores"] = {}
    session_state["asset_ids"] = {}
    session_state["chat_histories"] = {}
    session_state["processComplete"] = False

    for asset_id, (filename, text) in files_text.items():
        chroma_vectorstore = get_vectorstore(text, asset_id)
        session_state["vectorstores"][asset_id] = chroma_vectorstore
        session_state["asset_ids"][asset_id] = filename
        session_state["chat_histories"][asset_id] = []

    session_state["processComplete"] = True

    return jsonify({"asset_ids": session_state["asset_ids"]})

@app.route('/api/chat/start', methods=['POST'])
def start_chat():
    data = request.json
    asset_id = data.get("asset_id")
    if asset_id not in session_state["vectorstores"]:
        return jsonify({"error": "Invalid asset ID"}), 400

    session_state["selected_asset_id"] = asset_id
    session_state["chat_histories"][asset_id] = []

    return jsonify({"message": "Chat started", "asset_id": asset_id})

@app.route('/api/chat/message', methods=['POST'])
def chat_message():
    data = request.json
    user_message = data.get("message")
    asset_id = session_state["selected_asset_id"]

    if asset_id not in session_state["vectorstores"]:
        return jsonify({"error": "No asset selected"}), 400

    vectorstore = session_state["vectorstores"][asset_id]
    qa_chain = get_conversation_chain(vectorstore)

    # Get chat history specific to the selected asset ID
    chat_history = session_state["chat_histories"][asset_id]

    response = qa_chain({"question": user_message, "chat_history": chat_history})
    session_state["chat_histories"][asset_id].append((user_message, response["answer"]))

    return jsonify({"response": response["answer"]})

def get_files_text(uploaded_files):
    files_text = {}
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.filename)[1]
        text = ""
        if file_extension == ".pdf":
            text = get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text = get_docx_text(uploaded_file)
        
        # Generate a unique asset ID for the file
        asset_id = str(uuid.uuid4())
        files_text[asset_id] = (uploaded_file.filename, text)
    return files_text

def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    all_text = [para.text for para in doc.paragraphs if para.text.strip() != ""]
    text = ' '.join(all_text)
    return text

def get_vectorstore(text, asset_id):
    embeddings = HuggingFaceEmbeddings()
    text_chunks = get_text_chunks(text)
    metadatas = [{"asset_id": asset_id} for _ in range(len(text_chunks))]

    collection_name = f"collection_{asset_id}"
    chroma_vectorstore = Chroma.from_texts(
        text_chunks,
        embeddings,
        metadatas=metadatas,
        collection_name=collection_name
    )
    return chroma_vectorstore

def get_text_chunks(text):
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_conversation_chain(vectorstore):
    llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.7, "max_length": 64})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)
    return qa_chain

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
