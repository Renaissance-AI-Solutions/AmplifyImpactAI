import os
from flask import current_app # To access UPLOAD_FOLDER config
import PyPDF2
import docx # Assuming python-docx is installed
import logging

logger = logging.getLogger(__name__)

class TextExtractionService:
    @staticmethod
    def extract_text_from_file(saved_doc_filename: str, file_type_arg: str) -> str:
        """
        Extracts text from a given file based on its type.
        'saved_doc_filename' is the filename as stored on disk (e.g., kb_user_timestamp_original.pdf).
        'file_type_arg' is the extension (e.g., 'pdf', 'docx', 'txt').
        """
        print(f"--- DEBUG TES: extract_text_from_file called ---") # TES for TextExtractionService
        print(f"--- DEBUG TES: Received saved_doc_filename: '{saved_doc_filename}' ---")
        print(f"--- DEBUG TES: Received file_type_arg: '{file_type_arg}' ---")
        
        # Check if saved_doc_filename is already a full path
        if os.path.isabs(saved_doc_filename):
            file_path = saved_doc_filename
            print(f"--- DEBUG TES: Using provided absolute file path: '{file_path}' ---")
        else:
            # If it's just a filename, construct the full path
            if not current_app.config.get('UPLOAD_FOLDER'):
                logger.error("UPLOAD_FOLDER is not configured in the Flask app.")
                return "" # Cannot proceed without upload folder
                
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_doc_filename)
            print(f"--- DEBUG TES: Constructed file_path from filename: '{file_path}' ---")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found for text extraction: {file_path}")
            return ""

        text_content = ""
        normalized_file_type = str(file_type_arg).strip().lower()
        print(f"--- DEBUG TES: Normalized file_type for comparison: '{normalized_file_type}' ---")

        try:
            if normalized_file_type == 'pdf':
                print("--- DEBUG TES: Matched 'pdf' type ---")
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if reader.is_encrypted:
                        try:
                            reader.decrypt('')
                            print("--- DEBUG TES: PDF decrypted with empty password ---")
                        except Exception as decrypt_err:
                            logger.warning(f"Could not decrypt PDF {file_path}: {decrypt_err}")
                    for page_num, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
                    print(f"--- DEBUG TES: Extracted {len(reader.pages)} pages from PDF ---")
            elif normalized_file_type == 'docx':
                print("--- DEBUG TES: Matched 'docx' type ---")
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
            elif normalized_file_type == 'txt':
                print("--- DEBUG TES: Matched 'txt' type ---")
                encodings_to_try = ['utf-8-sig', 'utf-8', 'latin-1', 'windows-1252']
                for enc in encodings_to_try:
                    try:
                        with open(file_path, 'r', encoding=enc) as f:
                            text_content = f.read()
                        print(f"--- DEBUG TES: TXT file read successfully with encoding '{enc}' ---")
                        break 
                    except UnicodeDecodeError:
                        print(f"--- DEBUG TES: TXT file failed to decode with '{enc}' ---")
                        if enc == encodings_to_try[-1]: # Last attempt failed
                            logger.warning(f"Could not decode TXT file {file_path} with common encodings.")
                    except Exception as e_txt_open: # Catch other file opening errors
                        logger.error(f"Error opening/reading TXT file {file_path} with encoding '{enc}': {e_txt_open}", exc_info=True)
                        break # Stop trying if a non-decode error occurs
            else: 
                print(f"--- DEBUG TES: UNMATCHED normalized_file_type: '{normalized_file_type}' ---")
                logger.error(f"Unsupported file type ('{normalized_file_type}') for file path: {file_path}")
                return "" # Return empty for unsupported types
        except Exception as e:
            logger.error(f"General error during text extraction for {file_path} (type: {normalized_file_type}): {e}", exc_info=True)
            return "" # Return empty on general extraction error

        print(f"--- DEBUG TES: _extract_text finished, text length: {len(text_content)} ---")
        return text_content
