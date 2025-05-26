from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import login_required, current_user
from app.models import PortalUser, KnowledgeDocument, KnowledgeChunk, ScheduledPost, ManagedAccount
from app.forms import PostComposerForm, SchedulePostForm, get_managed_account_choices
from app.services.knowledge_base_manager import KnowledgeBaseManager
from app.services.social_media_platforms import XPlatform
from datetime import datetime, timezone
from app import db

content_studio_bp = Blueprint('content_studio_bp', __name__)

@content_studio_bp.route('/')
@login_required
def index():
    # Get recent posts
    recent_posts = (
        ScheduledPost.query.filter_by(portal_user_id=current_user.id)
        .order_by(ScheduledPost.created_at.desc())
        .limit(10)
        .all()
    )

    # Get draft posts
    draft_posts = ScheduledPost.query.filter_by(
        portal_user_id=current_user.id,
        status='draft'
    ).all()

    return render_template('content_studio/index.html',
                         recent_posts=recent_posts,
                         draft_posts=draft_posts)

@content_studio_bp.route('/composer', methods=['GET', 'POST'])
@login_required
def composer():
    form = PostComposerForm()
    schedule_form = SchedulePostForm()

    # Populate account choices
    form.target_account_id.choices = get_managed_account_choices(current_user.id)
    schedule_form.target_account_id_schedule.choices = get_managed_account_choices(current_user.id)

    if form.validate_on_submit():
        if form.post_now.data:  # Post immediately
            return _handle_immediate_post(form)
        elif form.schedule_post_action.data:  # Schedule post
            return _handle_schedule_post(form, schedule_form)
        elif form.save_draft_action.data:  # Save as draft
            # Save draft logic
            if form.content.data:
                from app import db
                from app.models import ScheduledPost, ManagedAccount
                draft = ScheduledPost(
                    portal_user_id=current_user.id,
                    managed_account_id=form.target_account_id.data,
                    content=form.content.data,
                    scheduled_time=datetime.now(timezone.utc), # Placeholder
                    status="draft"
                )
                db.session.add(draft)
                db.session.commit()
                flash(f"Content saved as draft (ID: {draft.id}).", 'success')
                return redirect(url_for('content_studio_bp.composer'))
            else:
                flash("Cannot save an empty draft.", "warning")
        elif form.generate_with_ai.data:  # Generate with AI
            return _handle_ai_generation(form)

    return render_template('content_studio/composer.html', form=form, schedule_form=schedule_form)

@content_studio_bp.route('/drafts')
@login_required
def drafts():
    drafts = ScheduledPost.query.filter_by(
        portal_user_id=current_user.id,
        status='draft'
    ).all()

    return render_template('content_studio/drafts.html', drafts=drafts)

@content_studio_bp.route('/draft/<int:draft_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_draft(draft_id):
    draft = ScheduledPost.query.get_or_404(draft_id)
    if draft.portal_user_id != current_user.id:
        flash('Unauthorized access to draft.', 'danger')
        return redirect(url_for('content_studio_bp.drafts'))

    form = PostComposerForm(obj=draft)
    schedule_form = SchedulePostForm()

    # Populate account choices
    form.target_account_id.choices = get_managed_account_choices(current_user.id)
    schedule_form.target_account_id_schedule.choices = get_managed_account_choices(current_user.id)

    if form.validate_on_submit():
        if form.post_now.data:  # Post immediately
            return _handle_immediate_post(form, draft)
        elif form.schedule_post_action.data:  # Schedule post
            return _handle_schedule_post(form, schedule_form, draft)
        elif form.generate_with_ai.data:  # Generate with AI
            return _handle_ai_generation(form)

    return render_template('content_studio/composer.html', form=form, schedule_form=schedule_form)

@content_studio_bp.route('/draft/<int:draft_id>/delete')
@login_required
def delete_draft(draft_id):
    draft = ScheduledPost.query.get_or_404(draft_id)
    if draft.portal_user_id != current_user.id:
        flash('Unauthorized access to draft.', 'danger')
        return redirect(url_for('content_studio_bp.drafts'))

    try:
        db.session.delete(draft)
        db.session.commit()
        flash('Draft deleted successfully!', 'success')
    except Exception as e:
        current_app.logger.error(f"Error deleting draft {draft_id}: {e}")
        flash('Error deleting draft.', 'danger')

    return redirect(url_for('content_studio_bp.drafts'))

def _handle_immediate_post(form, draft=None):
    """Handle immediate post submission."""
    try:
        # Get selected account
        account = ManagedAccount.query.get(form.target_account_id.data)
        if not account or not account.is_active:
            flash('Selected account is not active', 'error')
            return redirect(url_for('content_studio_bp.composer'))

        # Initialize platform
        platform = XPlatform(account)
        if not platform.client:
            flash('Cannot initialize platform client', 'error')
            return redirect(url_for('content_studio_bp.composer'))

        # Post content
        response = platform.post_update(form.content.data)
        if response:
            # Create post record if not editing a draft
            if not draft:
                post = ScheduledPost(
                    managed_account_id=account.id,
                    content=form.content.data,
                    status='posted',
                    posted_at=datetime.now(timezone.utc),
                    platform_post_id=response.get('id')
                )
                db.session.add(post)
            else:
                draft.status = 'posted'
                draft.posted_at = datetime.now(timezone.utc)
                draft.platform_post_id = response.get('id')

            db.session.commit()

            flash('Post published successfully!', 'success')
        else:
            flash('Failed to publish post', 'error')

        return redirect(url_for('content_studio_bp.composer'))

    except Exception as e:
        current_app.logger.error(f"Error posting immediately: {e}")
        flash('Error publishing post', 'error')
        return redirect(url_for('content_studio_bp.composer'))

def _handle_schedule_post(form, schedule_form, draft=None):
    """Handle scheduled post submission."""
    if not schedule_form.validate_on_submit():
        return redirect(url_for('content_studio_bp.composer'))

    try:
        # Create or update scheduled post
        if draft:
            draft.managed_account_id = schedule_form.target_account_id_schedule.data
            draft.content = schedule_form.content_schedule.data
            draft.scheduled_time = schedule_form.scheduled_time.data
            draft.status = 'pending'
        else:
            post = ScheduledPost(
                managed_account_id=schedule_form.target_account_id_schedule.data,
                content=schedule_form.content_schedule.data,
                scheduled_time=schedule_form.scheduled_time.data,
                status='pending'
            )
            db.session.add(post)

        db.session.commit()

        # Schedule the post
        try:
            from app.services.scheduler_service import scheduler_service
            if draft:
                scheduler_service.schedule_post(draft.id)
            else:
                scheduler_service.schedule_post(post.id)
        except ImportError:
            current_app.logger.warning("Scheduler service not available")

        flash('Post scheduled successfully!', 'success')
        return redirect(url_for('content_studio_bp.composer'))

    except Exception as e:
        current_app.logger.error(f"Error scheduling post: {e}")
        flash('Error scheduling post', 'error')
        return redirect(url_for('content_studio_bp.composer'))

@content_studio_bp.route('/generate-ai-content', methods=['POST'])
@login_required
def generate_ai_content():
    """AJAX endpoint for AI content generation."""
    try:
        topic = request.form.get('topic', '').strip()
        tone = request.form.get('tone', '').strip()
        style = request.form.get('style', '').strip()

        if not all([topic, tone, style]):
            return jsonify({'error': 'Missing required parameters'}), 400

        # Initialize knowledge base manager
        kbm = KnowledgeBaseManager(current_user.id)

        # Generate content
        content = kbm.generate_content(
            topic=topic,
            tone=tone,
            style=style
        )

        return jsonify({'content': content})

    except Exception as e:
        current_app.logger.error(f"Error generating AI content: {e}")
        return jsonify({'error': 'Failed to generate content'}), 500

@content_studio_bp.route('/post-now', methods=['POST'])
@login_required
def post_now():
    """AJAX endpoint for immediate posting."""
    try:
        content = request.form.get('content', '').strip()
        target_account_id = request.form.get('target_account_id')

        if not content:
            return jsonify({'error': 'Content is required'}), 400

        if not target_account_id:
            return jsonify({'error': 'Target account is required'}), 400

        # Get the managed account
        account = ManagedAccount.query.filter_by(
            id=target_account_id,
            portal_user_id=current_user.id
        ).first()

        if not account:
            return jsonify({'error': 'Account not found'}), 404

        # Initialize platform
        platform = XPlatform(account)

        # Post content
        response = platform.post_update(content)
        if response:
            # Create post record
            post = ScheduledPost(
                portal_user_id=current_user.id,
                managed_account_id=account.id,
                content=content,
                status='posted',
                posted_at=datetime.now(timezone.utc),
                platform_post_id=response.get('id')
            )
            db.session.add(post)
            db.session.commit()

            return jsonify({'success': True, 'message': 'Post published successfully'})
        else:
            return jsonify({'error': 'Failed to publish post'}), 500

    except Exception as e:
        current_app.logger.error(f"Error posting content: {e}")
        return jsonify({'error': 'Failed to publish post'}), 500

def _handle_ai_generation(form):
    """Handle AI content generation."""
    try:
        # Initialize knowledge base manager
        kbm = KnowledgeBaseManager(current_user.id)

        # Generate content
        content = kbm.generate_content(
            topic=form.topic.data,
            tone=form.tone.data,
            style=form.style.data
        )

        # Update form with generated content
        form.generated_content_display.data = content

        flash('AI content generated successfully!', 'success')
        # Create a new schedule form for rendering
        schedule_form = SchedulePostForm()
        schedule_form.target_account_id_schedule.choices = get_managed_account_choices(current_user.id)
        return render_template('content_studio/composer.html', form=form, schedule_form=schedule_form)

    except Exception as e:
        current_app.logger.error(f"Error generating AI content: {e}")
        flash('Error generating AI content', 'error')
        # Create a new schedule form for rendering
        schedule_form = SchedulePostForm()
        schedule_form.target_account_id_schedule.choices = get_managed_account_choices(current_user.id)
        return render_template('content_studio/composer.html', form=form, schedule_form=schedule_form)

