import logging
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request, render_template, flash, redirect, url_for
from flask_login import login_required, current_user
from flask_wtf.csrf import CSRFProtect
from app import db, limiter
from app.models import KnowledgeDocument, ManagedAccount, ScheduledPost
from app.forms import ContentGenerationForm, get_document_choices
from app.services.post_generator_service import PostGeneratorService

logger = logging.getLogger(__name__)

bp = Blueprint('content_generation', __name__)
post_generator = PostGeneratorService()

@bp.route('/generate', methods=['GET', 'POST'])
@login_required
@limiter.limit("10 per minute")
def generate():
    """Render the content generation interface with form."""
    form = ContentGenerationForm()
    
    # Populate document choices
    form.document_id.choices = get_document_choices(current_user.id, add_blank=True)
    
    # Check if there are no documents
    if len(form.document_id.choices) <= 1:  # Only the blank option
        flash('You need to upload and process documents before generating content.', 'warning')
    
    # Handle form submission
    generated_content = None
    if form.validate_on_submit():
        try:
            # Get the selected document
            document_id = form.document_id.data
            
            # Check if document_id is None (empty string was submitted)
            if document_id is None:
                flash('Please select a document', 'warning')
            else:
                document = db.session.get(KnowledgeDocument, document_id)
                
                if not document or document.portal_user_id != current_user.id:
                    flash('Document not found or access denied', 'danger')
                else:
                    # Extract parameters from form
                    params = {
                        'document_id': document_id,
                        'platform': form.platform.data,
                        'tone': form.tone.data,
                        'style': form.style.data,
                        'topic': form.topic.data,
                        'max_length': form.max_length.data,
                        'include_hashtags': form.include_hashtags.data,
                        'include_emoji': form.include_emoji.data,
                        'model': form.model.data
                    }
                    
                    # Generate content
                    generated_content, prompt_data = post_generator.generate_content(**params, return_prompt=True)
                    
                    # Handle save as draft if requested
                    if form.save_button.data and generated_content:
                        # Find the first account for the platform or use None
                        account = db.session.scalar(
                            db.select(ManagedAccount)
                            .filter_by(portal_user_id=current_user.id, platform_name=form.platform.data, is_active=True)
                        )
                        
                        # Create a draft post
                        draft = ScheduledPost(
                            portal_user_id=current_user.id,
                            managed_account_id=account.id if account else None,
                            content=generated_content,
                            status="draft"
                        )
                        db.session.add(draft)
                        db.session.commit()
                        
                        flash('Content saved as draft', 'success')
                        return redirect(url_for('content_generation.generate'))
            
        except Exception as e:
            logger.error(f"Error generating content: {e}", exc_info=True)
            flash(f'Error generating content: {str(e)}', 'danger')
    
    # Get recent drafts for display
    recent_drafts = db.session.scalars(
        db.select(ScheduledPost)
        .filter_by(portal_user_id=current_user.id, status="draft")
        .order_by(ScheduledPost.created_at.desc())
        .limit(5)
    ).all()
    
    return render_template(
        'content_generation/generate.html',
        form=form,
        generated_content=generated_content,
        recent_drafts=recent_drafts,
        prompt_data=prompt_data if 'prompt_data' in locals() else None
    )

@bp.route('/api/generate-content', methods=['POST'])
@login_required
@limiter.limit("10 per minute")
def generate_content_api():
    """API endpoint to generate content from knowledge base."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        document_id = data.get('document_id')
        platform = data.get('platform', 'twitter')
        tone = data.get('tone', 'informative')
        style = data.get('style', 'concise')
        topic = data.get('topic')
        max_length = data.get('max_length', 280)
        include_hashtags = data.get('include_hashtags', True)
        include_emoji = data.get('include_emoji', True)
        model = data.get('model', 'gpt-3.5-turbo')
        
        # Validate inputs
        if not document_id:
            return jsonify({'error': 'Missing document_id'}), 400
            
        # Check document exists and belongs to user
        document = db.session.get(KnowledgeDocument, document_id)
        if not document or document.portal_user_id != current_user.id:
            return jsonify({'error': 'Document not found or access denied'}), 404
            
        # Validate max_length
        try:
            max_length = int(max_length)
            if max_length < 10 or max_length > 3000:
                return jsonify({'error': 'max_length must be between 10 and 3000'}), 400
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid max_length value'}), 400
            
        # Generate content
        result = post_generator.generate_content(
            document_id=document_id,
            platform=platform,
            tone=tone,
            style=style,
            topic=topic,
            max_length=max_length,
            include_hashtags=include_hashtags,
            include_emoji=include_emoji,
            model=model,
            return_prompt=True
        )
        
        # Unpack the result
        if isinstance(result, tuple):
            content, prompt_data = result
        else:
            content = result
            prompt_data = None
        
        if not content:
            return jsonify({'error': 'Failed to generate content'}), 500
            
        return jsonify({
            'success': True,
            'content': content,
            'prompt_data': prompt_data
        })
        
    except Exception as e:
        logger.error(f"API error generating content: {e}", exc_info=True)
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
