from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
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
                    filename=filepath,
                    original_filename=file.filename,
                    file_type=file.filename.rsplit('.', 1)[1].lower(),
                    uploaded_at=datetime.now(timezone.utc)
                )
                db.session.add(document)
                db.session.commit()

                kbm = KnowledgeBaseManager(current_user.id)
                if kbm.process_document(document):
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
        'content': chunk.content,
        'document_id': chunk.document_id,
        'title': chunk.document.original_filename if chunk.document else None,
        'faiss_index_id': chunk.faiss_index_id if hasattr(chunk, 'faiss_index_id') else None
    }

@kb_bp.route('/search')
@login_required
def search():
    query = request.args.get('q', '')
    if not query:
        return render_template('knowledge_base/search.html', results=[])  # Empty results for empty query
    
    kbm = KnowledgeBaseManager()
    results = kbm.search_similar_chunks(query)
    
    # Convert results to a format suitable for display
    formatted_results = []
    for result in results:
        formatted_results.append({
            'chunk': result['chunk'],
            'score': result['score'],
            'document': KnowledgeDocument.query.get(result['chunk'].document_id)
        })
    
    return render_template('knowledge_base/search.html', query=query, results=formatted_results)

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
        return redirect(url_for('kb_bp.index'))
    
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
        return redirect(url_for('kb_bp.index'))
        
    except Exception as e:
        current_app.logger.error(f"Error deleting document {document_id}: {e}")
        flash('Error deleting document.', 'danger')
        return redirect(url_for('kb_bp.index'))

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
