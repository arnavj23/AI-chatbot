from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import json

from utils.document_processor import process_pdf
from utils.vector_store1 import VectorStore
from utils.groq_client import GroqClient
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 2048 * 2048  # 16MB max upload

# Initialize services
vector_store = VectorStore()
groq_client = GroqClient()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/admin')
def admin():
    # Get list of uploaded manuals
    manuals = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.endswith('.pdf'):
            manuals.append(filename)
    return render_template('admin.html', manuals=manuals)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the PDF and add to vector store
        try:
            product_name = request.form.get('product_name', os.path.splitext(filename)[0])
            chunks = process_pdf(filepath, product_name)
            vector_store.add_documents(chunks)
            return jsonify({"success": True, "message": f"File {filename} processed successfully"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query_text = data.get('query', '')
    
    if not query_text:
        return jsonify({"error": "Query is empty"}), 400
    
    try:
        # Get similar documents from vector store
        similar_docs = vector_store.search(query_text, k=3)
        
        # Format the context properly
        context = ""
        sources = []
        
        if similar_docs:
            for doc in similar_docs:
                # Check if doc is a dictionary with required keys
                if isinstance(doc, dict) and "content" in doc and "source" in doc:
                    context += f"\nFrom {doc['source']}:\n{doc['content']}\n\n"
                    sources.append(doc['source'])
        
        # Use Groq to generate a response
        response = groq_client.generate_response(query_text, context)
        
        # Ensure we're returning a proper string response
        if not isinstance(response, str):
            response = str(response)
        
        return jsonify({
            "response": response,
            "sources": sources
        })
    except Exception as e:
        import traceback
        print(f"Error in query endpoint: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "response": f"I'm sorry, I encountered an error while processing your query. Error: {str(e)}",
            "sources": []
        }), 500

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No audio file selected"}), 400
    
    # Save temporarily
    temp_path = os.path.join('/tmp', secure_filename(audio_file.filename))
    audio_file.save(temp_path)
    
    try:
        from speech_recognition import Recognizer, AudioFile
        
        recognizer = Recognizer()
        with AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.remove(temp_path)  # Clean up
        return jsonify({"text": text})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)