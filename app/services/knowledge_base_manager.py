import os
import logging
from flask import current_app, g
from sqlalchemy.orm import joinedload
from werkzeug.utils import secure_filename
from app import db
from app.models import KnowledgeDocument, KnowledgeChunk
from app.services.text_extraction import TextExtractionService
from app.utils.chunking import TokenTextSplitter
from .embedding_service import embedding_service_instance

logger = logging.getLogger(__name__)

faiss_index: Optional[faiss.Index] = None
faiss_id_to_chunk_pk_map: Dict[int, int] = {}
faiss_index_path_global: Optional[str] = None
faiss_map_path_global: Optional[str] = None
active_kb_model_name_for_faiss: str = ""

def initialize_kb_components(app):
    global faiss_index, faiss_id_to_chunk_pk_map, faiss_index_path_global, faiss_map_path_global
    global active_kb_model_name_for_faiss

    if embedding_service_instance is None or embedding_service_instance.client is None:
        logger.error("EmbeddingService not initialized or its client failed. Cannot initialize FAISS for KB.")
        return

    current_embedding_dimension = embedding_service_instance.get_embedding_dimension()
    current_model_name = embedding_service_instance.model_name
    
    if current_embedding_dimension == 0:
        logger.error(f"KB Embedding Provider '{current_model_name}' has dimension 0. FAISS cannot be initialized.")
        return
    
    active_kb_model_name_for_faiss = current_model_name

    logger.info(f"Initializing KB FAISS components for model: '{current_model_name}' with dimension {current_embedding_dimension}")

    instance_path = app.instance_path
    safe_model_name = secure_filename(current_model_name.replace('/', '_').replace('\\', '_'))
    faiss_index_filename = app.config.get('FAISS_INDEX_FILENAME_TPL', "kb_faiss_{model}.index").format(model=safe_model_name)
    faiss_map_filename = app.config.get('FAISS_MAP_FILENAME_TPL', "kb_faiss_map_{model}.pkl").format(model=safe_model_name)
    
    faiss_index_path_global = os.path.join(instance_path, faiss_index_filename)
    faiss_map_path_global = os.path.join(instance_path, faiss_map_filename)
    logger.info(f"FAISS index path: {faiss_index_path_global}")
    logger.info(f"FAISS map path: {faiss_map_path_global}")

    if os.path.exists(faiss_index_path_global) and os.path.exists(faiss_map_path_global):
        logger.info(f"Attempting to load FAISS index from {faiss_index_path_global}")
        try:
            faiss_index = faiss.read_index(faiss_index_path_global)
            if faiss_index.d != current_embedding_dimension:
                logger.warning(f"Loaded FAISS index dimension ({faiss_index.d}) from '{faiss_index_path_global}' does not match active provider dimension ({current_embedding_dimension}). Re-initializing FAISS index.")
                faiss_index = faiss.IndexFlatL2(current_embedding_dimension)
                faiss_id_to_chunk_pk_map = {}
            else:
                with open(faiss_map_path_global, "rb") as f:
                    faiss_id_to_chunk_pk_map = pickle.load(f)
                logger.info(f"FAISS index loaded with {faiss_index.ntotal} vectors. Map loaded with {len(faiss_id_to_chunk_pk_map)} entries.")
        except Exception as e:
            logger.error(f"Error loading FAISS index/map: {e}. Creating new ones.", exc_info=True)
            faiss_index = faiss.IndexFlatL2(current_embedding_dimension)
            faiss_id_to_chunk_pk_map = {}
    else:
        logger.info(f"No existing FAISS index found for model '{current_model_name}'. Creating a new one.")
        faiss_index = faiss.IndexFlatL2(current_embedding_dimension)
        faiss_id_to_chunk_pk_map = {}

class KnowledgeBaseManager:
    def __init__(self, portal_user_id: int):
        self.portal_user_id = portal_user_id
        
        if embedding_service_instance is None or embedding_service_instance.client is None:
            logger.error("KBM Instantiation: Global EmbeddingService not properly initialized!")
            raise RuntimeError("EmbeddingService not available for KnowledgeBaseManager.")
        
        self.embedding_service = embedding_service_instance
        self.index = faiss_index
        self.id_map = faiss_id_to_chunk_pk_map

        self.text_splitter = TokenTextSplitter(
            chunk_size=current_app.config.get('KB_CHUNK_SIZE_TOKENS', 256),
            chunk_overlap=current_app.config.get('KB_CHUNK_OVERLAP_TOKENS', 32)
        )
        self.text_extraction_service = TextExtractionService()

        if self.index is None:
            logger.error(f"KBM for user {portal_user_id} initialized, but FAISS index is None. Check startup logs.")
        else:
            logger.debug(f"KBM for user {portal_user_id} initialized with FAISS index (ntotal: {self.index.ntotal}) and id_map (len: {len(self.id_map)}).")


    def _ensure_services_loaded(self):
        if not self.embedding_service or not self.embedding_service.client or self.index is None:
            logger.critical("KBM CRITICAL: Embedding service/client or FAISS index is None.")
            raise RuntimeError("KB embedding service or FAISS index not initialized. Check app startup logs.")
        if self.embedding_service.model_name != active_kb_model_name_for_faiss:
            logger.error(f"KBM CRITICAL: Active embedding model '{self.embedding_service.model_name}' does not match FAISS index model '{active_kb_model_name_for_faiss}'. Re-initialization needed.")
            raise RuntimeError(f"Mismatch between active embedding model and FAISS index model. Please re-initialize KB components.")
        if self.index.d != self.embedding_service.get_embedding_dimension():
             logger.error(f"KBM CRITICAL: FAISS index dimension ({self.index.d}) doesn't match embedding service dimension ({self.embedding_service.get_embedding_dimension()}). Re-initialization needed.")
             raise RuntimeError("Mismatch between FAISS index dimension and embedding service dimension. Please re-initialize KB components.")


    def process_document(self, document_id: int, saved_filename: str, file_type: str) -> tuple[bool, str]:
        self._ensure_services_loaded()
        doc = db.session.get(KnowledgeDocument, document_id)
        if not doc:
            logger.error(f"Document with ID {document_id} not found.")
            return False, "Document not found."
        if doc.portal_user_id != self.portal_user_id:
            logger.error(f"User {self.portal_user_id} attempting to process document {document_id} belonging to user {doc.portal_user_id}")
            return False, "Access denied to document."

        doc.status = "processing"
        db.session.commit()

        try:
            upload_folder = current_app.config['UPLOAD_FOLDER']
            file_path = os.path.join(upload_folder, saved_filename)

            if not os.path.exists(file_path):
                logger.error(f"File {file_path} not found for document {document_id}")
                doc.status = "error_file_not_found"
                db.session.commit()
                return False, "File not found."

            text_content = self.text_extraction_service.extract_text(file_path, file_type)

            if not text_content:
                logger.warning(f"No text extracted from document {document_id} ({saved_filename}).")
                doc.status = "error_no_text"
                db.session.commit()
                return False, "No text could be extracted from the document."

            chunks_text = self.text_splitter.split_text(text_content)

            if not chunks_text:
                logger.warning(f"No chunks created for document {document_id}. Text length: {len(text_content)}")
                doc.status = "error_no_chunks"
                db.session.commit()
                return False, "Document yielded no text chunks after splitting."
            
            logger.info(f"Processing {len(chunks_text)} chunks for document {document_id} using {self.embedding_service.model_name}.")

            chunk_embeddings_list = self.embedding_service.get_embeddings(chunks_text, input_type="passage")

            if not chunk_embeddings_list or not any(e for e in chunk_embeddings_list):
                logger.error(f"Failed to generate any embeddings for chunks of document {document_id}.")
                doc.status = "error_embedding"
                db.session.commit()
                return False, "Failed to generate embeddings for document chunks."
            
            valid_embeddings = [e for e in chunk_embeddings_list if e]
            valid_chunks_text = [chunks_text[i] for i, e in enumerate(chunk_embeddings_list) if e]

            if len(valid_embeddings) != len(chunks_text):
                logger.warning(f"Partial embedding success for doc {document_id}: {len(valid_embeddings)}/{len(chunks_text)} chunks embedded.")
                if not valid_embeddings:
                    doc.status = "error_embedding_all_failed"
                    db.session.commit()
                    return False, "All document chunks failed to generate embeddings."

            chunk_embeddings_np = np.array(valid_embeddings, dtype='float32')

            self._remove_document_chunks_from_faiss(document_id)
            KnowledgeChunk.query.filter_by(document_id=document_id, portal_user_id=self.portal_user_id).delete()
            db.session.commit()

            new_faiss_ids = []
            for i, text_chunk in enumerate(valid_chunks_text):
                faiss_id = self.index.ntotal 
                new_faiss_ids.append(faiss_id)
                
                chunk_record = KnowledgeChunk(
                    document_id=document_id,
                    portal_user_id=self.portal_user_id,
                    chunk_text=text_chunk,
                    embedding_model_name=self.embedding_service.model_name,
                    faiss_index_id=faiss_id 
                )
                db.session.add(chunk_record)
            
            db.session.commit()

            temp_chunk_records = KnowledgeChunk.query.filter(
                KnowledgeChunk.document_id == document_id,
                KnowledgeChunk.portal_user_id == self.portal_user_id,
                KnowledgeChunk.faiss_index_id.in_(new_faiss_ids)
            ).order_by(KnowledgeChunk.faiss_index_id).all()

            if len(temp_chunk_records) != len(new_faiss_ids):
                 logger.error(f"Mismatch in new chunk records and faiss_ids for document {document_id}. Critical error.")
                 doc.status = "error_faiss_mapping"
                 db.session.commit()
                 return False, "Critical error mapping FAISS IDs to chunk primary keys."

            for i, chunk_record in enumerate(temp_chunk_records):
                self.id_map[chunk_record.faiss_index_id] = chunk_record.id

            self.index.add(chunk_embeddings_np) # Add to FAISS
            
            doc.status = "processed"
            doc.chunk_count = len(valid_chunks_text)
            doc.processed_at = datetime.utcnow()
            doc.embedding_model_name = self.embedding_service.model_name
            db.session.commit()
            
            save_kb_components() 
            logger.info(f"Document {document_id} processed successfully with {len(valid_chunks_text)} chunks. FAISS total: {self.index.ntotal}")
            return True, f"Document processed with {len(valid_chunks_text)} chunks using {self.embedding_service.model_name}."

        except Exception as e:
            db.session.rollback()
            logger.error(f"Error processing document {document_id}: {e}", exc_info=True)
            doc.status = "error_processing_failed"
            db.session.commit()
            return False, f"An error occurred during document processing: {str(e)}"


    def search_kb(self, query_text: str, top_k: int = 5, document_ids: Optional[List[int]] = None) -> List[Dict]:
        self._ensure_services_loaded()
        if not query_text:
            return []
        if self.index.ntotal == 0:
            logger.info("Search_kb called but FAISS index is empty.")
            return []

        query_embedding_list = self.embedding_service.get_embeddings(query_text, input_type="query")
        if not query_embedding_list or not query_embedding_list[0]:
            logger.error(f"Failed to generate embedding for query: {query_text[:100]} using {self.embedding_service.model_name}")
            return []
        
        query_embedding = np.array(query_embedding_list, dtype='float32')

        try:
            distances, indices = self.index.search(query_embedding, top_k * 3)
        except Exception as e:
            logger.error(f"FAISS search error: {e}", exc_info=True)
            return []

        results = []
        retrieved_chunk_pks = set()

        for i in range(indices.shape[1]):
            faiss_id = indices[0, i]
            if faiss_id == -1:
                continue
            
            chunk_pk = self.id_map.get(int(faiss_id))
            if chunk_pk is None:
                logger.warning(f"FAISS ID {faiss_id} not found in id_map. Inconsistency?")
                continue
            
            if chunk_pk in retrieved_chunk_pks:
                continue

            chunk_record = db.session.query(KnowledgeChunk).options(joinedload(KnowledgeChunk.document)).filter(
                KnowledgeChunk.id == chunk_pk,
                KnowledgeChunk.portal_user_id == self.portal_user_id
            ).first()

            if chunk_record:
                if document_ids and chunk_record.document_id not in document_ids:
                    continue
                
                if chunk_record.embedding_model_name != self.embedding_service.model_name:
                    logger.warning(f"Chunk {chunk_pk} was embedded with '{chunk_record.embedding_model_name}' but searching with '{self.embedding_service.model_name}'. Skipping.")
                    continue

                results.append({
                    'chunk_id': chunk_record.id,
                    'document_id': chunk_record.document_id,
                    'document_title': chunk_record.document.title if chunk_record.document else "Unknown Document",
                    'text': chunk_record.chunk_text,
                    'score': float(distances[0, i]),
                    'faiss_id': int(faiss_id)
                })
                retrieved_chunk_pks.add(chunk_pk)
                if len(results) >= top_k:
                    break
        
        results.sort(key=lambda x: x['score'])
        return results[:top_k]


    def _remove_document_chunks_from_faiss(self, document_id: int):
        self._ensure_services_loaded()
        chunks_to_remove = KnowledgeChunk.query.filter_by(
            document_id=document_id, 
            portal_user_id=self.portal_user_id
        ).all()
        
        if not chunks_to_remove:
            return

        faiss_ids_to_remove = [chunk.faiss_index_id for chunk in chunks_to_remove if chunk.faiss_index_id is not None]
        
        if not faiss_ids_to_remove:
            logger.info(f"No FAISS IDs associated with chunks for document {document_id} to remove.")
            return

        try:
            ids_to_remove_vector = faiss.IDSelectorArray(np.array(sorted(faiss_ids_to_remove), dtype='int64'))
            removed_count = self.index.remove_ids(ids_to_remove_vector)
            logger.info(f"Removed {removed_count} embeddings from FAISS for document {document_id}.")
            
            for faiss_id in faiss_ids_to_remove:
                if faiss_id in self.id_map:
                    del self.id_map[faiss_id]
        except Exception as e:
            logger.error(f"Error removing FAISS IDs for document {document_id}: {e}", exc_info=True)


    def migrate_embeddings(self, old_model_name: str, new_embedding_service: 'EmbeddingService', batch_size: int = 50):
        logger.info(f"Starting embedding migration for user {self.portal_user_id} from '{old_model_name}' to '{new_embedding_service.model_name}'.")
        
        documents_to_migrate = KnowledgeDocument.query.filter_by(
            portal_user_id=self.portal_user_id,
            embedding_model_name=old_model_name
        ).all()

        if not documents_to_migrate:
            logger.info(f"No documents found for user {self.portal_user_id} using model '{old_model_name}' to migrate.")
            return True, "No documents to migrate."

        total_docs = len(documents_to_migrate)
        logger.info(f"Found {total_docs} documents to migrate for user {self.portal_user_id}.")

        original_service = self.embedding_service
        self.embedding_service = new_embedding_service
        
        global faiss_index, faiss_id_to_chunk_pk_map, active_kb_model_name_for_faiss
        
        old_faiss_index = faiss_index
        old_faiss_map = faiss_id_to_chunk_pk_map
        old_active_model_name = active_kb_model_name_for_faiss

        try:
            with current_app.app_context():
                global embedding_service_instance
                original_global_embedding_service = embedding_service_instance
                embedding_service_instance = new_embedding_service
                
                initialize_kb_components(current_app)
                
                embedding_service_instance = original_global_embedding_service

            faiss_index = faiss_index
            faiss_id_to_chunk_pk_map = faiss_id_to_chunk_pk_map
            
            logger.info(f"New FAISS index initialized for model '{new_embedding_service.model_name}' with dimension {new_embedding_service.get_embedding_dimension()}.")    

            for i, doc in enumerate(documents_to_migrate):
                logger.info(f"Migrating document {i+1}/{total_docs}: ID {doc.id} ('{doc.title}')")
                success, message = self.process_document(doc.id, doc.saved_filename, doc.file_type)
                if success:
                    logger.info(f"Successfully migrated document ID {doc.id}.")
                    doc.embedding_model_name = new_embedding_service.model_name
                    db.session.commit()
                else:
                    logger.error(f"Failed to migrate document ID {doc.id}: {message}")
            
            logger.info(f"Migration completed for user {self.portal_user_id} to model '{new_embedding_service.model_name}'.")
            self.embedding_service = original_service
            return True, f"Migration to {new_embedding_service.model_name} completed."

        except Exception as e:
            logger.error(f"Critical error during migration process: {e}", exc_info=True)
            faiss_index = old_faiss_index
            faiss_id_to_chunk_pk_map = old_faiss_map
            active_kb_model_name_for_faiss = old_active_model_name
            self.embedding_service = original_service
            logger.info("Restored previous FAISS index and embedding service due to migration error.")
            return False, f"Migration failed: {str(e)}"


    def get_document_details(self, document_id: int) -> Optional[KnowledgeDocument]:
        self._ensure_services_loaded()
        return db.session.get(KnowledgeDocument, document_id)

    def list_documents(self) -> List[KnowledgeDocument]:
        self._ensure_services_loaded()
        return KnowledgeDocument.query.filter_by(portal_user_id=self.portal_user_id).order_by(KnowledgeDocument.uploaded_at.desc()).all()

    def delete_document(self, document_id: int) -> tuple[bool, str]:
        self._ensure_services_loaded()
        doc = db.session.get(KnowledgeDocument, document_id)
        if not doc or doc.portal_user_id != self.portal_user_id:
            return False, "Document not found or access denied."

        try:
            self._remove_document_chunks_from_faiss(document_id)
            KnowledgeChunk.query.filter_by(document_id=document_id, portal_user_id=self.portal_user_id).delete()
            db.session.delete(doc)
            db.session.commit()
            save_kb_components()
            logger.info(f"Document {document_id} and its chunks deleted successfully.")
            return True, "Document and its embeddings deleted."
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
            return False, f"Error deleting document: {str(e)}"

    def get_kb_status(self) -> dict:
        status = {
            "portal_user_id": self.portal_user_id,
            "embedding_service_initialized": False,
            "embedding_model_name": "N/A",
            "embedding_dimension": 0,
            "faiss_index_available": False,
            "faiss_index_model_name": active_kb_model_name_for_faiss,
            "faiss_index_dimension": 0,
            "faiss_vector_count": 0,
            "faiss_id_map_size": len(self.id_map if self.id_map else {}),
            "faiss_index_path": faiss_index_path_global,
            "faiss_map_path": faiss_map_path_global,
            "total_documents": 0,
            "total_chunks": 0
        }
        if self.embedding_service and self.embedding_service.client:
            status["embedding_service_initialized"] = True
            status["embedding_model_name"] = self.embedding_service.model_name
            status["embedding_dimension"] = self.embedding_service.get_embedding_dimension()
        
        if self.index:
            status["faiss_index_available"] = True
            status["faiss_index_dimension"] = self.index.d
            status["faiss_vector_count"] = self.index.ntotal

        status["total_documents"] = KnowledgeDocument.query.filter_by(portal_user_id=self.portal_user_id).count()
        status["total_chunks"] = KnowledgeChunk.query.filter_by(portal_user_id=self.portal_user_id).count()
        return status

def save_kb_components():
    global faiss_index, faiss_id_to_chunk_pk_map, faiss_index_path_global, faiss_map_path_global
    
    if faiss_index is not None and faiss_index_path_global:
        try:
            logger.info(f"Saving FAISS index with {faiss_index.ntotal} vectors to {faiss_index_path_global}")
            faiss.write_index(faiss_index, faiss_index_path_global)
        except Exception as e:
            logger.error(f"Error saving FAISS index to {faiss_index_path_global}: {e}", exc_info=True)
    else:
        logger.info("FAISS index is None or path not set, skipping save.")

    if faiss_id_to_chunk_pk_map is not None and faiss_map_path_global:
        try:
            logger.info(f"Saving FAISS ID map with {len(faiss_id_to_chunk_pk_map)} entries to {faiss_map_path_global}")
            with open(faiss_map_path_global, "wb") as f:
                pickle.dump(faiss_id_to_chunk_pk_map, f)
        except Exception as e:
            logger.error(f"Error saving FAISS ID map to {faiss_map_path_global}: {e}", exc_info=True)
    else:
        logger.info("FAISS ID map is None or path not set, skipping save.")
