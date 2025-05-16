from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from werkzeug.utils import secure_filename
import os
from flask_login import login_required, current_user
from app.models import PortalUser, KnowledgeDocument, KnowledgeChunk
from app.forms import KnowledgeDocumentUploadForm
from app.services.knowledge_base_manager import KnowledgeBaseManager
from datetime import datetime, timezone

kb_bp = Blueprint('kb_bp', __name__)

@kb_bp.route('/', methods=['GET', 'POST'])
@login_required
def manage_knowledge_base():
    form = KnowledgeDocumentUploadForm()
    documents = KnowledgeDocument.query.filter_by(portal_user_id=current_user.id).all()
    total_documents = len(documents)
    total_chunks = sum(getattr(doc, 'chunks', []).count() if hasattr(doc, 'chunks') else 0 for doc in documents)
    last_updated = max((getattr(doc, 'uploaded_at', None) for doc in documents), default=None)
    storage_used = calculate_storage_used() if 'calculate_storage_used' in globals() else None

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

                kbm = KnowledgeBaseManager()
                if kbm.process_document(document):
                    flash('Document uploaded and processed successfully!', 'success')
                else:
                    flash('Error processing document. Please check the logs.', 'danger')
            except Exception as e:
                flash(f'Error uploading document: {str(e)}', 'danger')
            return redirect(url_for('kb_bp.manage_knowledge_base'))
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{getattr(form, field).label.text}: {error}', 'danger')
            # fall through to render page with errors
    return render_template('knowledge_base/index.html', 
                           documents=documents, 
                           form=form,
                           total_documents=total_documents,
                           total_chunks=total_chunks,
                           last_updated=last_updated,
                           storage_used=storage_used)


@kb_bp.route('/<int:document_id>')
@login_required
def view_document(document_id):
    document = KnowledgeDocument.query.get_or_404(document_id)
    if document.portal_user_id != current_user.id:
        flash('Unauthorized access to document.', 'danger')
        return redirect(url_for('kb_bp.index'))
    
    chunks = KnowledgeChunk.query.filter_by(document_id=document_id).all()
    return render_template('knowledge_base/view.html', document=document, chunks=chunks)

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
