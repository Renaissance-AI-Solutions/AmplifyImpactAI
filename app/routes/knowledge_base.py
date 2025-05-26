from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify
from werkzeug.utils import secure_filename
import os
from flask_login import login_required, current_user
from app.models import PortalUser, KnowledgeDocument, KnowledgeChunk
from app.forms import KnowledgeDocumentUploadForm
from app.services.knowledge_base_manager import KnowledgeBaseManager
from datetime import datetime, timezone
from app import db
import logging

logger = logging.getLogger(__name__)

kb_bp = Blueprint('kb_bp', __name__)

# Placeholder function - user needs to implement actual logic
def calculate_storage_used():
    # This function should calculate the total disk space used by uploaded documents
    # associated with the current_user. For now, a placeholder.
    total_size = 0
    try:
        if current_user.is_authenticated:
            docs = db.session.scalars(db.select(KnowledgeDocument).filter_by(portal_user_id=current_user.id)).all()
            upload_folder = current_app.config.get('UPLOAD_FOLDER')
            if upload_folder:
                for doc in docs:
                    if doc.filename:
                        try:
                            filepath = os.path.join(upload_folder, doc.filename)
                            if os.path.exists(filepath):
                                total_size += os.path.getsize(filepath)
                        except Exception as e_size:
                            logger.error(f"Error getting size for {doc.filename}: {e_size}")
    except Exception as e_calc:
        logger.error(f"Error in calculate_storage_used: {e_calc}")
    return total_size # Return size in bytes

@kb_bp.route('/', methods=['GET', 'POST'])
@login_required
def manage_knowledge_base():
    form = KnowledgeDocumentUploadForm()
    kb_manager = KnowledgeBaseManager(current_user.id)

    # STEP 1: Fetch documents and calculate template variables
    documents_query = db.select(KnowledgeDocument).filter_by(
        portal_user_id=current_user.id
    ).order_by(KnowledgeDocument.uploaded_at.desc())
    documents = db.session.scalars(documents_query).all()

    # Helper to calculate total chunks
    def calculate_total_chunks(docs):
        return sum(doc.chunks.count() if hasattr(doc, 'chunks') and hasattr(doc.chunks, 'count') else len(getattr(doc, 'chunks', [])) for doc in docs)

    # Helper to calculate last updated
    def calculate_last_updated(docs):
        dates = [doc.uploaded_at for doc in docs if getattr(doc, 'uploaded_at', None)]
        return max(dates) if dates else None

    total_chunks_val = calculate_total_chunks(documents)
    last_updated_val = calculate_last_updated(documents)
    storage_used_val = calculate_storage_used() if 'calculate_storage_used' in globals() else None

    # STEP 2: Handle POST request
    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                file = form.document.data
                filename = secure_filename(f"{current_user.id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{file.filename}")
                filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                document = KnowledgeDocument(
                    portal_user_id=current_user.id,
                    filename=filename,  # Store just the filename, not the full path
                    original_filename=file.filename,
                    file_type=file.filename.rsplit('.', 1)[1].lower(),
                    uploaded_at=datetime.now(timezone.utc)
                )
                db.session.add(document)
                db.session.commit()

                kbm = KnowledgeBaseManager(current_user.id)
                if kbm.process_document(document.id, filepath, file.filename.rsplit('.', 1)[1].lower()):
                    flash('Document uploaded and processed successfully!', 'success')
                else:
                    flash('Error processing document. Please check the logs.', 'danger')
            except Exception as e:
                db.session.rollback()
                logger.error("Upload error: %s", str(e), exc_info=True)
                flash('An error occurred during upload. Please try again.', 'danger')
            return redirect(url_for('kb_bp.manage_knowledge_base'))
        else:
            flash("Please correct the errors below.", "warning")
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{getattr(form, field).label.text}: {error}', 'danger')
            # Will fall through to render_template with form errors

    # STEP 3: Render template (handles both GET and failed POST)
    return render_template('knowledge_base/index.html',
                         form=form,
                         documents=documents,
                         total_chunks=total_chunks_val,
                         last_updated=last_updated_val,
                         storage_used=storage_used_val)


@kb_bp.route('/documents/<int:document_id>')
@login_required
def view_document(document_id):
    """View a specific document and its chunks.

    Args:
        document_id: The ID of the document to view

    Returns:
        Rendered template with document and chunks
    """
    document = KnowledgeDocument.query.get_or_404(document_id)

    # Verify ownership
    if document.portal_user_id != current_user.id:
        flash('Unauthorized access to document.', 'danger')
        return redirect(url_for('kb_bp.manage_knowledge_base'))

    # Get all chunks for this document
    chunks = KnowledgeChunk.query.filter_by(document_id=document_id)\
        .order_by(KnowledgeChunk.id.asc())\
        .all()

    return render_template(
        'knowledge_base/view.html',
        document=document,
        chunks=chunks,
        title=f"{document.original_filename} - Document"
    )

@kb_bp.route('/chunks/<int:chunk_id>')
@login_required
def view_chunk(chunk_id):
    """View a specific chunk as JSON, with ownership verification."""
    chunk = KnowledgeChunk.query.get_or_404(chunk_id)
    # Verify ownership via parent document
    if not chunk.document or chunk.document.portal_user_id != current_user.id:
        return {'error': 'Unauthorized access to chunk.'}, 403
    return {
        'id': chunk.id,
        'content': chunk.chunk_text,  # Updated from content to chunk_text
        'document_id': chunk.document_id,
        'title': chunk.document.original_filename if chunk.document else None,
        'faiss_index_id': chunk.faiss_index_id if hasattr(chunk, 'faiss_index_id') else None,
        'created_at': chunk.created_at.isoformat() if chunk.created_at else None
    }

@kb_bp.route('/search', methods=['GET', 'POST'])
@login_required
def search():
    logger.info("=== KNOWLEDGE BASE SEARCH ROUTE CALLED ===")
    logger.info(f"Request method: {request.method}")
    logger.info(f"Current user: {current_user.id if current_user else 'None'}")

    if request.method == 'POST':
        logger.info("Handling POST request for AJAX search")

        # Handle AJAX POST request from composer page
        query = request.form.get('query', '').strip()
        logger.info(f"Search query received: '{query}'")
        logger.info(f"Request form data: {dict(request.form)}")

        if not query:
            logger.warning("Empty query provided")
            return {'results': [], 'message': 'No query provided'}, 400

        try:
            logger.info(f"Initializing KnowledgeBaseManager for user {current_user.id}")
            kbm = KnowledgeBaseManager(current_user.id)
            logger.info("KnowledgeBaseManager initialized successfully")

            logger.info(f"Calling search_kb with query: '{query}', top_k=5")
            results = kbm.search_kb(query, top_k=5)
            logger.info(f"Search completed. Found {len(results)} results")

            # Log each result for debugging
            for i, result in enumerate(results):
                logger.info(f"Result {i}: {result}")

            # Format results for frontend consumption
            formatted_results = []
            for result in results:
                formatted_result = {
                    'title': result.get('document_title', 'Unknown Document'),
                    'content': result.get('text', '')[:200] + ('...' if len(result.get('text', '')) > 200 else ''),
                    'full_content': result.get('text', ''),
                    'score': 1.0 - (result.get('score', 1.0) / 2.0),  # Convert distance to similarity score
                    'document_id': result.get('document_id'),
                    'chunk_id': result.get('chunk_id')
                }
                formatted_results.append(formatted_result)
                logger.info(f"Formatted result: {formatted_result}")

            logger.info(f"Returning {len(formatted_results)} formatted results")
            return {'results': formatted_results}

        except Exception as e:
            logger.error(f"Error in knowledge base search: {e}", exc_info=True)
            return {'error': 'Search failed. Please try again.'}, 500

    else:
        # Handle GET request for search page
        query = request.args.get('q', '')
        if not query:
            return render_template('knowledge_base/search.html', results=[])

        try:
            kbm = KnowledgeBaseManager(current_user.id)
            results = kbm.search_kb(query, top_k=10)

            # Convert results to a format suitable for display
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'chunk_id': result.get('chunk_id'),
                    'text': result.get('text', ''),
                    'score': result.get('score', 0),
                    'document_title': result.get('document_title', 'Unknown Document'),
                    'document_id': result.get('document_id')
                })

            return render_template('knowledge_base/search.html', query=query, results=formatted_results)

        except Exception as e:
            logger.error(f"Error in knowledge base search: {e}", exc_info=True)
            flash('Search failed. Please try again.', 'danger')
            return render_template('knowledge_base/search.html', query=query, results=[])

@kb_bp.route('/debug-status')
@login_required
def debug_status():
    """Debug route to check knowledge base status."""
    try:
        kbm = KnowledgeBaseManager(current_user.id)
        status = kbm.get_kb_status()

        # Get document details
        documents = KnowledgeDocument.query.filter_by(portal_user_id=current_user.id).all()
        doc_details = []
        for doc in documents:
            chunks = KnowledgeChunk.query.filter_by(document_id=doc.id).all()
            doc_details.append({
                'id': doc.id,
                'filename': doc.original_filename,
                'status': doc.status,
                'chunk_count': doc.chunk_count,
                'actual_chunks': len(chunks),
                'embedding_model': doc.embedding_model_name,
                'processed_at': doc.processed_at,
                'chunks_with_faiss_ids': len([c for c in chunks if c.faiss_index_id is not None])
            })

        return {
            'kb_status': status,
            'documents': doc_details,
            'user_id': current_user.id
        }
    except Exception as e:
        logger.error(f"Error in debug status: {e}", exc_info=True)
        return {'error': str(e)}, 500

@kb_bp.route('/test-search')
@login_required
def test_search():
    """Test search functionality with a simple query."""
    try:
        kbm = KnowledgeBaseManager(current_user.id)

        # Check database chunks
        from app import db
        chunks = db.session.query(KnowledgeChunk).filter(
            KnowledgeChunk.document_id.in_(
                db.session.query(KnowledgeDocument.id).filter_by(portal_user_id=current_user.id)
            )
        ).all()

        # Check FAISS ID mapping
        faiss_mapped_chunks = [c for c in chunks if c.faiss_index_id is not None]

        # Test with a simple query
        test_query = "battlespace"
        results = kbm.search_kb(test_query, top_k=3)

        return {
            'query': test_query,
            'results_count': len(results),
            'results': results,
            'faiss_total': kbm.index.ntotal if hasattr(kbm, 'index') else 'No index',
            'embedding_service_loaded': kbm.embedding_service is not None,
            'db_chunks_total': len(chunks),
            'db_chunks_with_faiss_ids': len(faiss_mapped_chunks),
            'id_map_size': len(kbm.id_map) if hasattr(kbm, 'id_map') else 'No id_map',
            'sample_chunks': [
                {
                    'id': c.id,
                    'faiss_id': c.faiss_index_id,
                    'has_embedding_vector': c.embedding_vector is not None,
                    'embedding_vector_size': len(c.embedding_vector) if c.embedding_vector else 0,
                    'text_preview': c.chunk_text[:100] if c.chunk_text else 'No text'
                } for c in chunks[:3]
            ]
        }
    except Exception as e:
        logger.error(f"Error in test search: {e}", exc_info=True)
        return {'error': str(e)}, 500

@kb_bp.route('/check-chunks')
@login_required
def check_chunks():
    """Check the current state of chunks in the database."""
    try:
        from app import db

        # Get all chunks for this user
        chunks = db.session.query(KnowledgeChunk).filter(
            KnowledgeChunk.document_id.in_(
                db.session.query(KnowledgeDocument.id).filter_by(portal_user_id=current_user.id)
            )
        ).order_by(KnowledgeChunk.id.desc()).all()

        # Get chunks with and without embedding vectors
        chunks_with_vectors = [c for c in chunks if c.embedding_vector is not None]
        chunks_without_vectors = [c for c in chunks if c.embedding_vector is None]

        # Get recent chunks (highest IDs)
        recent_chunks = chunks[:10]
        old_chunks = chunks[-10:] if len(chunks) > 10 else []

        return {
            'total_chunks': len(chunks),
            'chunks_with_vectors': len(chunks_with_vectors),
            'chunks_without_vectors': len(chunks_without_vectors),
            'chunk_id_range': f"{chunks[-1].id} to {chunks[0].id}" if chunks else "No chunks",
            'recent_chunks': [
                {
                    'id': c.id,
                    'has_vector': c.embedding_vector is not None,
                    'vector_size': len(c.embedding_vector) if c.embedding_vector else 0,
                    'faiss_id': c.faiss_index_id,
                    'embedding_model': c.embedding_model_name,
                    'text_preview': c.chunk_text[:50] + '...' if c.chunk_text else 'No text'
                } for c in recent_chunks
            ],
            'old_chunks': [
                {
                    'id': c.id,
                    'has_vector': c.embedding_vector is not None,
                    'vector_size': len(c.embedding_vector) if c.embedding_vector else 0,
                    'faiss_id': c.faiss_index_id,
                    'embedding_model': c.embedding_model_name,
                    'text_preview': c.chunk_text[:50] + '...' if c.chunk_text else 'No text'
                } for c in old_chunks
            ]
        }
    except Exception as e:
        logger.error(f"Error checking chunks: {e}", exc_info=True)
        return {'error': str(e)}, 500

@kb_bp.route('/delete-all-chunks', methods=['POST'])
@login_required
def delete_all_chunks():
    """Delete all chunks for this user to start fresh."""
    try:
        from app import db

        # Get all chunks for this user
        chunks = db.session.query(KnowledgeChunk).filter(
            KnowledgeChunk.document_id.in_(
                db.session.query(KnowledgeDocument.id).filter_by(portal_user_id=current_user.id)
            )
        ).all()

        chunk_count = len(chunks)

        # Delete all chunks
        for chunk in chunks:
            db.session.delete(chunk)

        db.session.commit()

        # Clear FAISS index
        kbm = KnowledgeBaseManager(current_user.id)
        if kbm.index is not None:
            kbm.index.reset()
            kbm.id_map.clear()

        # Save empty FAISS index
        from app.services.knowledge_base_manager import save_kb_components
        save_kb_components()

        return {
            'message': f'Deleted {chunk_count} chunks successfully!',
            'deleted_chunks': chunk_count
        }

    except Exception as e:
        logger.error(f"Error deleting chunks: {e}", exc_info=True)
        return {'error': str(e)}, 500

@kb_bp.route('/fix-embeddings', methods=['POST'])
@login_required
def fix_embeddings():
    """Fix embedding model names and rebuild FAISS index properly."""
    try:
        from app import db

        # Step 1: Fix embedding model names in database
        chunks = db.session.query(KnowledgeChunk).filter(
            KnowledgeChunk.document_id.in_(
                db.session.query(KnowledgeDocument.id).filter_by(portal_user_id=current_user.id)
            )
        ).all()

        fixed_chunks = 0
        for chunk in chunks:
            if chunk.embedding_model_name is None or chunk.embedding_model_name == '':
                chunk.embedding_model_name = 'BAAI/bge-large-en-v1.5'
                fixed_chunks += 1

        db.session.commit()
        logger.info(f"Fixed embedding model names for {fixed_chunks} chunks")

        # Step 2: Rebuild FAISS index and ID mapping
        kbm = KnowledgeBaseManager(current_user.id)

        # Clear existing FAISS index by resetting it
        if kbm.index is not None:
            dimension = kbm.index.d
            kbm.index.reset()
            # Clear the ID map
            kbm.id_map.clear()
            logger.info(f"Cleared FAISS index and ID map")

        # Re-add all chunks to FAISS
        chunks_added = 0
        for chunk in chunks:
            if chunk.embedding_vector:
                try:
                    # Deserialize the embedding
                    import pickle
                    embedding = pickle.loads(chunk.embedding_vector)

                    # Add to FAISS index
                    faiss_id = kbm.index.ntotal
                    kbm.index.add(embedding.reshape(1, -1))
                    kbm.id_map[faiss_id] = chunk.id

                    # Update chunk's FAISS ID
                    chunk.faiss_index_id = faiss_id
                    chunks_added += 1

                except Exception as e:
                    logger.error(f"Error adding chunk {chunk.id} to FAISS: {e}")

        db.session.commit()

        # Save FAISS index using the global function
        from app.services.knowledge_base_manager import save_kb_components
        save_kb_components()

        return {
            'message': 'Embeddings fixed successfully!',
            'fixed_chunks': fixed_chunks,
            'chunks_added_to_faiss': chunks_added,
            'final_faiss_count': kbm.index.ntotal
        }

    except Exception as e:
        logger.error(f"Error fixing embeddings: {e}", exc_info=True)
        return {'error': str(e)}, 500

@kb_bp.route('/rebuild-embeddings', methods=['POST'])
@login_required
def rebuild_embeddings():
    """Rebuild embeddings for all documents."""
    try:
        kbm = KnowledgeBaseManager(current_user.id)

        # Get all processed documents for this user
        documents = KnowledgeDocument.query.filter_by(
            portal_user_id=current_user.id,
            status='processed'
        ).all()

        if not documents:
            return {'message': 'No processed documents found to rebuild'}, 200

        results = []
        for doc in documents:
            logger.info(f"Rebuilding embeddings for document {doc.id}: {doc.original_filename}")

            # Get the file path
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], doc.filename)

            if not os.path.exists(filepath):
                results.append({
                    'document_id': doc.id,
                    'filename': doc.original_filename,
                    'success': False,
                    'message': 'File not found'
                })
                continue

            # Reprocess the document
            success, message = kbm.process_document(doc.id, filepath, doc.file_type)
            results.append({
                'document_id': doc.id,
                'filename': doc.original_filename,
                'success': success,
                'message': message
            })

        return {
            'message': f'Processed {len(documents)} documents',
            'results': results
        }

    except Exception as e:
        logger.error(f"Error rebuilding embeddings: {e}", exc_info=True)
        return {'error': str(e)}, 500

@kb_bp.route('/delete', methods=['POST'])
@login_required
def delete_document():
    document_id = request.form.get('document_id')
    if not document_id:
        flash('No document specified.', 'danger')
        return redirect(url_for('kb_bp.manage_knowledge_base'))

    document_id = int(document_id)
    document = KnowledgeDocument.query.get_or_404(document_id)
    if document.portal_user_id != current_user.id:
        flash('Unauthorized access to document.', 'danger')
        return redirect(url_for('kb_bp.manage_knowledge_base'))

    try:
        # Delete file
        if os.path.exists(document.filename):
            os.remove(document.filename)

        # Delete chunks
        KnowledgeChunk.query.filter_by(document_id=document_id).delete()

        # Delete document
        db.session.delete(document)
        db.session.commit()

        flash('Document deleted successfully!', 'success')
        return redirect(url_for('kb_bp.manage_knowledge_base'))

    except Exception as e:
        current_app.logger.error(f"Error deleting document {document_id}: {e}")
        flash('Error deleting document.', 'danger')
        return redirect(url_for('kb_bp.manage_knowledge_base'))

@kb_bp.route('/chunks/<int:chunk_id>/delete', methods=['POST'])
@login_required
def delete_chunk(chunk_id):
    chunk = KnowledgeChunk.query.get_or_404(chunk_id)

    # Verify ownership via parent document
    if not chunk.document or chunk.document.portal_user_id != current_user.id:
        flash('Unauthorized access to chunk.', 'danger')
        return redirect(url_for('kb_bp.manage_knowledge_base'))

    document_id = chunk.document_id

    try:
        # Get KB manager
        kb_manager = KnowledgeBaseManager(current_user.id)

        # Delete chunk from FAISS index if it exists
        if hasattr(chunk, 'faiss_index_id') and chunk.faiss_index_id is not None:
            kb_manager.remove_chunk_from_index(chunk.faiss_index_id)

        # Delete from database
        db.session.delete(chunk)
        db.session.commit()

        flash('Chunk deleted successfully!', 'success')

    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting chunk {chunk_id}: {e}", exc_info=True)
        flash('Error deleting chunk.', 'danger')

    return redirect(url_for('kb_bp.view_document', document_id=document_id))
