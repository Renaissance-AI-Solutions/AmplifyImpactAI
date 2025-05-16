from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, render_template
from flask_login import login_required, current_user
from app import db
from app.models import KnowledgeDocument, ManagedAccount
from app.services.post_generator_service import PostGeneratorService

bp = Blueprint('content_generation', __name__)
post_generator = PostGeneratorService()

@bp.route('/generate', methods=['GET'])
@login_required
def generate():
    """Render the post generation interface."""
    # Get user's documents
    documents = db.session.scalars(
        db.select(KnowledgeDocument)
        .filter_by(portal_user_id=current_user.id)
        .order_by(KnowledgeDocument.uploaded_at.desc())
    ).all()
    
    # Get user's managed accounts
    accounts = db.session.scalars(
        db.select(ManagedAccount)
        .filter_by(portal_user_id=current_user.id, is_active=True)
        .order_by(ManagedAccount.platform_name, ManagedAccount.account_display_name)
    ).all()
    
    return render_template(
        'content_generation/generate.html',
        documents=documents,
        accounts=accounts
    )

@bp.route('/api/generate-post', methods=['POST'])
@login_required
def generate_post():
    """Generate a post from knowledge base content."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        document_id = data.get('document_id')
        managed_account_id = data.get('managed_account_id')
        template_type = data.get('template_type', 'informational')
        scheduled_time = data.get('scheduled_time')
        
        # Validate inputs
        if not document_id or not managed_account_id:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Check document exists and belongs to user
        document = db.session.get(KnowledgeDocument, document_id)
        if not document or document.portal_user_id != current_user.id:
            return jsonify({'error': 'Document not found'}), 404
            
        # Check account exists and belongs to user
        account = db.session.get(ManagedAccount, managed_account_id)
        if not account or account.portal_user_id != current_user.id:
            return jsonify({'error': 'Account not found'}), 404
            
        # Parse scheduled time if provided
        if scheduled_time:
            try:
                scheduled_time = datetime.fromisoformat(scheduled_time.replace('Z', '+00:00'))
            except ValueError:
                return jsonify({'error': 'Invalid scheduled_time format'}), 400
                
        # Generate and schedule post
        post = post_generator.create_scheduled_post(
            portal_user_id=current_user.id,
            managed_account_id=managed_account_id,
            document_id=document_id,
            template_type=template_type,
            scheduled_time=scheduled_time
        )
        
        if not post:
            return jsonify({'error': 'Failed to generate post'}), 500
            
        return jsonify({
            'success': True,
            'post': {
                'id': post.id,
                'content': post.content,
                'scheduled_time': post.scheduled_time.isoformat(),
                'status': post.status
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/document/<int:document_id>/topics', methods=['GET'])
@login_required
def get_document_topics(document_id):
    """Get topics extracted from a document."""
    try:
        # Check document exists and belongs to user
        document = db.session.get(KnowledgeDocument, document_id)
        if not document or document.portal_user_id != current_user.id:
            return jsonify({'error': 'Document not found'}), 404
            
        # Extract topics
        topics = post_generator.extract_topics(document_id)
        
        return jsonify({
            'success': True,
            'topics': [{
                'terms': topic['terms'],
                'score': float(topic['score']),  # Convert numpy float to Python float
                'sample_text': topic['chunks'][0].chunk_text if topic['chunks'] else None
            } for topic in topics]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
