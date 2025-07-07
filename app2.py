from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
from werkzeug.utils import secure_filename
import json
import logging

from utils.document_processor import process_pdf
from utils.vector_store import VectorStore  # Fixed import
from utils.groq_client import GroqClient
from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("app.log"), 
                             logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload
app.secret_key = os.urandom(24)  # For session management

# Initialize services
vector_store = VectorStore()
groq_client = GroqClient()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# Admin dashboard
@app.route('/admin')
def admin():
    # Get list of uploaded manuals
    manuals = []
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.pdf'):
                manuals.append(filename)
    except Exception as e:
        logger.error(f"Error reading manuals directory: {str(e)}")
    
    # Get list of products from vector store
    products = set()
    for doc in vector_store.documents:
        if "product" in doc and doc["product"]:
            products.add(doc["product"])
    
    return render_template('admin.html', manuals=manuals, products=list(products))

# Handle file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the PDF and add to vector store
            product_name = request.form.get('product_name', os.path.splitext(filename)[0])
            logger.info(f"Processing PDF: {filename} for product: {product_name}")
            
            chunks = process_pdf(filepath, product_name)
            logger.info(f"Generated {len(chunks)} chunks from {filename}")
            
            vector_store.add_documents(chunks)
            return jsonify({
                "success": True, 
                "message": f"File {filename} processed with {len(chunks)} chunks"
            }), 200
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400

# Handle chat queries
@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data.get('query', '')
        product_filter = data.get('product', None)  # Optional product filter
        
        if not query_text:
            return jsonify({"error": "Query is empty"}), 400
        
        logger.info(f"Query received: '{query_text}' for product: {product_filter}")
        
        # Store last query in session for context
        if 'history' not in session:
            session['history'] = []
        
        # Add query to history (max 5 recent queries)
        session['history'].append(query_text)
        if len(session['history']) > 5:
            session['history'] = session['history'][-5:]
        
        # Enhance query with recent history for better context
        enhanced_query = query_text
        if len(session['history']) > 1:
            recent_context = " ".join(session['history'][:-1])
            enhanced_query = f"{recent_context}\n\nCurrent question: {query_text}"
        
        # Get similar documents from vector store (more documents for better context)
        num_docs = 5  # Get more documents to ensure enough context
        
        if product_filter:
            similar_docs = vector_store.search_by_product(enhanced_query, product_filter, k=num_docs)
        else:
            similar_docs = vector_store.search(enhanced_query, k=num_docs)
        
        logger.info(f"Found {len(similar_docs)} relevant chunks")
        
        if not similar_docs:
            return jsonify({
                "response": "I couldn't find any relevant information in the manuals. Please try rephrasing your question or check if manuals for this product have been uploaded.",
                "sources": []
            })
        
        # Use Groq to generate a response with the enhanced context
        response_data = groq_client.generate_response(
            query_text, 
            similar_docs,
            product_name=product_filter
        )
        
        return jsonify({
            "response": response_data["answer"],
            "sources": response_data["sources"]
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            "response": "I'm sorry, there was an error processing your request.",
            "error": str(e)
        }), 500

# Speech-to-text endpoint
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
        logger.info(f"Speech recognized: '{text}'")
        return jsonify({"text": text})
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        logger.error(f"Speech recognition error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# List available products
@app.route('/products', methods=['GET'])
def list_products():
    products = set()
    for doc in vector_store.documents:
        if "product" in doc and doc["product"]:
            products.add(doc["product"])
    
    return jsonify({"products": list(products)})

if __name__ == '__main__':
    app.run(debug=True)