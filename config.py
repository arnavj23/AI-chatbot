import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Upload folder configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "manuals")

# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

# Vector store config
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Initialize a dummy document if none exists to avoid cold start problems
DUMMY_DOCUMENT_PATH = os.path.join(VECTOR_STORE_PATH, "dummy_document.txt")
if not os.path.exists(DUMMY_DOCUMENT_PATH):
    with open(DUMMY_DOCUMENT_PATH, "w") as f:
        f.write("""
        # Philips Healthcare Support
        
        This is a support system for Philips Healthcare products. 
        You can ask questions about Philips medical devices and systems.
        
        ## Common Products:
        
        - DreamStation: Sleep therapy system
        - Respironics: Respiratory care products
        - IntelliVue: Patient monitoring systems
        - Epiq/Affiniti: Ultrasound systems
        - Azurion: Image-guided therapy platform
        
        For detailed information, please upload product manuals to enhance the knowledge base.
        """)