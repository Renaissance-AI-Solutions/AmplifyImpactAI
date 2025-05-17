import os
import logging
from flask import current_app
from datetime import datetime, timezone
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

embedding_model = None
faiss_index = None
faiss_id_to_chunk_pk_map = {}
faiss_index_path_global = None
faiss_map_path_global = None

from app import db
from app.models import KnowledgeDocument, KnowledgeChunk
from app.utils.encryption import encrypt_token, decrypt_token
from langchain.text_splitter import TokenTextSplitter

logger = logging.getLogger(__name__)

def initialize_kb_components(app):
    global embedding_model, faiss_index, faiss_id_to_chunk_pk_map
    global faiss_index_path_global, faiss_map_path_global

    if embedding_model is not None:
        logger.info("Global SentenceTransformer model and FAISS components appear to be already initialized.")
        return

    logger.info("Attempting to initialize global Knowledge Base components...")
    # 1. Initialize SentenceTransformer Model
    model_loaded_successfully = False
    try:
        model_name_config = app.config.get('SENTENCE_TRANSFORMER_MODEL', 'all-MiniLM-L6-v2')
        logger.info(f"Loading SentenceTransformer model: '{model_name_config}'...")
        embedding_model = SentenceTransformer(model_name_config)
        logger.info(f"Global SentenceTransformer model '{model_name_config}' loaded successfully.")
        model_loaded_successfully = True
    except Exception as e:
        logger.error(f"CRITICAL FAILURE: Failed to initialize global SentenceTransformer model ('{model_name_config}'). This will prevent KB processing. Error: {e}", exc_info=True)
        embedding_model = None
        faiss_index = None
    # 2. Initialize FAISS Index (only if model loaded)
    if model_loaded_successfully and embedding_model:
        logger.info("Proceeding with FAISS index initialization...")
        try:
            embedding_dimension = embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension determined: {embedding_dimension}")
            instance_path = app.instance_path
            faiss_index_path_global = os.path.join(instance_path, app.config.get('FAISS_INDEX_FILENAME', "kb_faiss.index"))
            faiss_map_path_global = os.path.join(instance_path, app.config.get('FAISS_MAP_FILENAME', "kb_faiss_map.pkl"))
            if os.path.exists(faiss_index_path_global) and os.path.exists(faiss_map_path_global):
                logger.info(f"Attempting to load existing FAISS index from: {faiss_index_path_global}")
                faiss_index = faiss.read_index(faiss_index_path_global)
                with open(faiss_map_path_global, "rb") as f:
                    faiss_id_to_chunk_pk_map = pickle.load(f)
                logger.info(f"FAISS index loaded with {faiss_index.ntotal if faiss_index else 'N/A'} vectors. ID map loaded.")
            else:
                logger.info(f"No existing FAISS index found at {faiss_index_path_global}. Creating a new one.")
                faiss_index = faiss.IndexFlatL2(embedding_dimension)
                faiss_id_to_chunk_pk_map = {}
            if faiss_index is None:
                logger.warning("FAISS index is unexpectedly None after initialization attempt. Re-creating.")
                faiss_index = faiss.IndexFlatL2(embedding_dimension)
                faiss_id_to_chunk_pk_map = {}
            logger.info("FAISS index initialization complete.")
        except Exception as e_faiss:
            logger.error(f"CRITICAL FAILURE: Failed to initialize FAISS index. Error: {e_faiss}", exc_info=True)
            faiss_index = None
    elif not model_loaded_successfully:
        logger.error("Skipping FAISS initialization because embedding model failed to load.")
        faiss_index = None
    if embedding_model and faiss_index:
        logger.info("Knowledge Base components (model and FAISS index) initialized successfully.")
    else:
        logger.warning("Knowledge Base components initialization incomplete. Model or FAISS index might be None.")

class KnowledgeBaseManager:
    def __init__(self, portal_user_id: int):
        self.portal_user_id = portal_user_id
        self.embedding_model = embedding_model
        self.index = faiss_index
        # Properly initialize all mapping attributes
        self.id_map = faiss_id_to_chunk_pk_map
        self.index_to_chunk_id = self.id_map  # Alias for backward compatibility
        self.chunk_id_to_index = {}  # Initialize reverse mapping
        if self.embedding_model is None or self.index is None:
            logger.warning(f"KBM instance for user {portal_user_id} created, but global embedding_model or faiss_index is None. KB functionality will be impaired. Check startup logs.")

    def process_document(self, document: KnowledgeDocument):
        """Process a document and create chunks."""
        try:
            # Use new _extract_text for robust file type diagnosis
            text = self._extract_text(document.filename, document.file_type)
            
            # Token-based chunking for optimal LLM embedding
            chunks = self._chunk_text_by_tokens(text)
            
            # Create and save chunks
            for i, chunk in enumerate(chunks):
                self._create_and_save_chunk(document, chunk, i)
            
            # Update document status
            document.status = 'processed'
            document.processed_at = datetime.now(timezone.utc)
            db.session.commit()
            
            logger.info(f"Processed document {document.id} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {document.id}: {e}")
            document.status = 'error'
            document.error_message = str(e)
            db.session.commit()
            return False

    def _extract_text(self, saved_doc_filename: str, file_type_arg: str):
        print(f"--- DEBUG KBM: _extract_text ENTERED ---")
        print(f"--- DEBUG KBM: Received saved_doc_filename: '{saved_doc_filename}' (type: {type(saved_doc_filename)}) ---")
        print(f"--- DEBUG KBM: Received file_type_arg: '{file_type_arg}' (type: {type(file_type_arg)}) ---") # CRUCIAL
        from flask import current_app
        import os
        import PyPDF2
        import docx
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], saved_doc_filename)
        print(f"--- DEBUG KBM: Constructed file_path: '{file_path}' ---")
        text_content = ""
        # Normalize the received file_type_arg for robust comparison
        normalized_file_type = str(file_type_arg).strip().lower()
        print(f"--- DEBUG KBM: Normalized file_type for comparison: '{normalized_file_type}' ---")
        if normalized_file_type == 'pdf':
            print("--- DEBUG KBM: Matched 'pdf' type ---")
            try:
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if reader.is_encrypted:
                        try:
                            reader.decrypt('')
                        except Exception:
                            logger.warning(f"Could not decrypt PDF {file_path}")
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except Exception as e:
                logger.error(f"Error extracting PDF text from {file_path}: {e}", exc_info=True)
        elif normalized_file_type == 'docx':
            print("--- DEBUG KBM: Matched 'docx' type ---")
            try:
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text_content += para.text + "\n"
            except Exception as e:
                logger.error(f"Error extracting DOCX text from {file_path}: {e}", exc_info=True)
        elif normalized_file_type == 'txt':
            print("--- DEBUG KBM: Matched 'txt' type ---")
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    text_content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        text_content = f.read()
                except Exception as e_txt:
                    logger.error(f"Error extracting TXT text from {file_path} with multiple encodings: {e_txt}", exc_info=True)
            except Exception as e:
                logger.error(f"Error extracting TXT text from {file_path}: {e}", exc_info=True)
        else:
            print(f"--- DEBUG KBM: UNMATCHED normalized_file_type: '{normalized_file_type}' ---")
            logger.error(f"Unsupported file type ('{normalized_file_type}') for actual file path: {file_path}")
            return ""
        print(f"--- DEBUG KBM: _extract_text finished, text length: {len(text_content)} ---")
        return text_content

    def _split_text_into_chunks(self, text, max_chunk_size=500):
        """(Deprecated) Split text into chunks of approximately max_chunk_size tokens."""
        # This method is now replaced by _chunk_text_by_tokens for optimal LLM chunking.
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        for word in words:
            if current_length + len(word) > max_chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word)
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        return chunks

    def _chunk_text_by_tokens(self, text: str, 
                             chunk_size_tokens: int = 256,  # Optimal size can vary, 256-512 is common for ada-002
                             chunk_overlap_tokens: int = 32, # Common overlap
                             encoding_name: str = "cl100k_base" # For text-embedding-ada-002, gpt-3.5-turbo, gpt-4
                            ) -> list[str]:
        """
        Splits text into chunks based on token count using LangChain's TokenTextSplitter.
        """
        if not text or not text.strip():
            logger.warning("Attempted to chunk empty or whitespace-only text.")
            return []
        try:
            text_splitter = TokenTextSplitter(
                encoding_name=encoding_name,
                chunk_size=chunk_size_tokens,
                chunk_overlap=chunk_overlap_tokens
            )
            chunks = text_splitter.split_text(text)
            logger.info(f"TokenTextSplitter: Chunked text into {len(chunks)} chunks (target size: {chunk_size_tokens} tokens, overlap: {chunk_overlap_tokens} tokens).")
            meaningful_chunks = [chunk for chunk in chunks if chunk.strip()]
            if len(meaningful_chunks) < len(chunks):
                logger.info(f"Filtered out {len(chunks) - len(meaningful_chunks)} empty chunks after token splitting.")
            return meaningful_chunks
        except Exception as e:
            logger.error(f"Error during token-based text splitting: {e}", exc_info=True)
            return []

    def _create_and_save_chunk(self, document, chunk_text, chunk_index):
        """Create and save a chunk with its embedding."""
        chunk = KnowledgeChunk(
            document_id=document.id,
            chunk_text=chunk_text,
            faiss_index_id=None  # Will be set after indexing
        )
        db.session.add(chunk)
        db.session.flush()  # Get chunk ID
        
        # Add to index
        self._add_chunk_to_index(chunk)

    def _add_chunk_to_index(self, chunk):
        """Add a chunk to the FAISS index."""
        if self.index is None:
            self._initialize_index()
        # Get embedding
        embedding = self._get_embedding(chunk.chunk_text)
        self.index.add(np.array([embedding]))
        index_position = self.index.ntotal - 1
        self.id_map[index_position] = chunk.id  # Use self.id_map consistently
        self.chunk_id_to_index[chunk.id] = index_position
        chunk.faiss_index_id = index_position
        db.session.commit()

    def _initialize_index(self):
        """Initialize FAISS index if not exists."""
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dimension)

    def _get_embedding(self, text):
        """Get embedding for text using sentence transformer model."""
        return self.embedding_model.encode(text)

    def search_similar_chunks(self, query, top_k=5):
        """Search for chunks most similar to the query."""
        if self.index is None:
            return []
        query_embedding = self._get_embedding(query)
        D, I = self.index.search(np.array([query_embedding]), top_k)
        similar_chunks = []
        for i in I[0]:
            if i < 0:
                continue
            chunk_id = self.id_map.get(i)  # Use self.id_map instead of self.index_to_chunk_id
            if chunk_id:
                chunk = db.session.get(KnowledgeChunk, chunk_id)
                if chunk:
                    similar_chunks.append({
                        'chunk': chunk,
                        'score': D[0][list(I[0]).index(i)]
                    })
        return similar_chunks

    def generate_comment(self, post_content, post_author, post_context):
        """Generate a comment using knowledge base."""
        try:
            # Find relevant knowledge
            relevant_chunks = self.search_similar_chunks(post_content, top_k=3)
            
            # Build context from relevant chunks
            context = "\n".join([chunk.chunk_text for chunk in relevant_chunks])
            
            # Generate comment using AI (this would be your actual AI generation logic)
            comment = self._generate_comment_with_ai(
                post_content=post_content,
                post_author=post_author,
                post_context=post_context,
                relevant_context=context
            )
            
            return comment
            
        except Exception as e:
            logger.error(f"Error generating comment: {e}")
            return None

    def _generate_comment_with_ai(self, post_content, post_author, post_context, relevant_context):
        """Generate comment using AI (implementation would depend on your AI service)."""
        # This is a placeholder - you would implement your actual AI comment generation logic here
        return f"Great post by {post_author}! I found this particularly interesting: {relevant_context[:100]}..."

    def save_index(self, index_path, map_path):
        """Save FAISS index and mappings to disk."""
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            import pickle
            with open(map_path, 'wb') as f:
                pickle.dump({
                    'index_to_chunk_id': self.id_map,  # Use self.id_map consistently
                    'chunk_id_to_index': self.chunk_id_to_index
                }, f)

    def load_index(self, index_path, map_path):
        """Load FAISS index and mappings from disk."""
        try:
            self.index = faiss.read_index(index_path)
            
            import pickle
            with open(map_path, 'rb') as f:
                mappings = pickle.load(f)
                self.index_to_chunk_id = mappings['index_to_chunk_id']
                self.chunk_id_to_index = mappings['chunk_id_to_index']
            
            logger.info("Loaded FAISS index and mappings successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return False
