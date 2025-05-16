from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
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
    if form.validate_on_submit():
        # Save file
        file = form.document.data
        filename = os.path.join(current_app.config['UPLOAD_FOLDER'], f"{current_user.id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{file.filename}")
        file.save(filename)

        # Create document record
        document = KnowledgeDocument(
            portal_user_id=current_user.id,
            filename=filename,
            original_filename=file.filename,
            file_type=file.filename.rsplit('.', 1)[1].lower()
        )
        db.session.add(document)
        db.session.commit()

        # Process document
        kbm = KnowledgeBaseManager()
        if kbm.process_document(document):
            flash('Document uploaded and processed successfully!', 'success')
        else:
            flash('Error processing document. Please check the logs.', 'danger')

        return redirect(url_for('kb_bp.manage_knowledge_base'))

    # --- DEBUG: knowledge_base.py ---
    print(f"Type of 'form' object: {type(form)}")
    print(f"Fields available in 'form' object: {dir(form)}")
    if hasattr(form, 'document'):
        print(f"form.document exists. Type: {type(form.document)}")
        if hasattr(form.document, 'label'):
            print(f"form.document.label.text: {getattr(form.document.label, 'text', 'NO LABEL TEXT')}" )
        else:
            print(f"form.document has no label object or label text.")
    else:
        print(f"CRITICAL DEBUG: 'form' object does NOT have a 'document' attribute!")
    print(f"--- END DEBUG ---")
    documents = KnowledgeDocument.query.filter_by(portal_user_id=current_user.id).all()
    return render_template('knowledge_base/index.html', documents=documents, form=form)


@kb_bp.route('/knowledge-base/<int:document_id>')
@login_required
def view_document(document_id):
    document = KnowledgeDocument.query.get_or_404(document_id)
    if document.portal_user_id != current_user.id:
        flash('Unauthorized access to document.', 'danger')
        return redirect(url_for('kb_bp.index'))
    
    chunks = KnowledgeChunk.query.filter_by(document_id=document_id).all()
    return render_template('knowledge_base/view.html', document=document, chunks=chunks)

@kb_bp.route('/knowledge-base/search')
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

@kb_bp.route('/knowledge-base/delete', methods=['POST'])
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
