from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app
from flask_login import login_required, current_user
from app import db
from app.models import PortalUser, GeneratedComment, CommentAutomationSetting, ManagedAccount
from app.forms import CommentSettingsForm, EditCommentForm
from app.services.knowledge_base_manager import KnowledgeBaseManager
from datetime import datetime, timezone

engagement_bp = Blueprint('engagement_bp', __name__)

@engagement_bp.route('/')
@login_required
def index():
    # Get pending comments
    pending_comments = GeneratedComment.query.filter_by(
        portal_user_id=current_user.id,
        status='pending_review'
    ).all()
    
    # Get recent activity
    recent_activity = (
        GeneratedComment.query.filter_by(
            portal_user_id=current_user.id
        )
        .order_by(GeneratedComment.created_at.desc())
        .limit(10)
        .all()
    )
    
    return render_template('engagement/index.html',
                         pending_comments=pending_comments,
                         recent_activity=recent_activity)

@engagement_bp.route('/comment-settings', methods=['GET', 'POST'])
@login_required
def comment_settings():
    settings_obj = db.session.scalar(
        db.select(CommentAutomationSetting).filter_by(portal_user_id=current_user.id)
    )
    form = CommentSettingsForm(obj=settings_obj)
    form.default_posting_account_id.choices = get_managed_account_choices(current_user.id, platform='X', add_blank=True)
    
    if form.validate_on_submit():
        if not settings_obj:
            settings_obj = CommentAutomationSetting(portal_user_id=current_user.id)
            db.session.add(settings_obj)
        form.populate_obj(settings_obj)
        if settings_obj.default_posting_managed_account_id:
            acc = db.session.get(ManagedAccount, settings_obj.default_posting_managed_account_id)
            if not acc or acc.portal_user_id != current_user.id or acc.platform_name != 'X':
                flash("Invalid default posting account selected.", "danger")
                settings_obj.default_posting_managed_account_id = None
            elif not acc.is_active:
                flash(f"Warning: Default posting account '{acc.account_display_name}' is not currently active.", "warning")
        db.session.commit()
        flash('Comment automation settings saved successfully!', 'success')
        log_action_db(current_user.id, "COMMENT_SETTINGS_UPDATED", "SUCCESS", f"Keywords: {settings_obj.keywords[:50]}..., Monitored: {settings_obj.monitored_x_accounts[:50]}...")
        return redirect(url_for('.comment_settings'))
    elif request.method == 'POST':
        flash('Please correct the errors in the form.', 'danger')
    
    return render_template('engagement/comment_settings.html', title='Comment Automation Settings', form=form)
    form = CommentSettingsForm()
    
    # Get current settings
    settings = CommentAutomationSetting.query.filter_by(portal_user_id=current_user.id).first()
    if settings:
        form.keywords.data = settings.keywords
        form.monitored_x_accounts.data = settings.monitored_x_accounts
        form.default_posting_account_id.choices = get_managed_account_choices(current_user.id)
        form.default_posting_account_id.data = settings.default_posting_managed_account_id
        form.is_active.data = settings.is_active
    else:
        form.default_posting_account_id.choices = get_managed_account_choices(current_user.id)
    
    if form.validate_on_submit():
        if settings:
            # Update existing settings
            settings.keywords = form.keywords.data
            settings.monitored_x_accounts = form.monitored_x_accounts.data
            settings.default_posting_managed_account_id = form.default_posting_account_id.data
            settings.is_active = form.is_active.data
        else:
            # Create new settings
            settings = CommentAutomationSetting(
                portal_user_id=current_user.id,
                keywords=form.keywords.data,
                monitored_x_accounts=form.monitored_x_accounts.data,
                default_posting_managed_account_id=form.default_posting_account_id.data,
                is_active=form.is_active.data
            )
            db.session.add(settings)
        
        db.session.commit()
        
        # Schedule comment discovery if enabled
        if form.is_active.data:
            scheduler_service.schedule_comment_discovery(settings.id)
        
        flash('Comment automation settings saved successfully!', 'success')
        return redirect(url_for('engagement_bp.comment_settings'))
    
    return render_template('engagement/settings.html', form=form)

@engagement_bp.route('/comments/<int:comment_id>/approve')
@login_required
def approve_comment(comment_id):
    comment = GeneratedComment.query.get_or_404(comment_id)
    if comment.portal_user_id != current_user.id:
        flash('Unauthorized access to comment.', 'danger')
        return redirect(url_for('engagement_bp.index'))
    
    try:
        # Get posting account
        account = ManagedAccount.query.get(comment.managed_account_id_to_post_from)
        if not account or not account.is_active:
            flash('Posting account is not active. Please select a different account.', 'warning')
            return redirect(url_for('engagement_bp.index'))
        
        # Initialize platform
        platform = XPlatform(account)
        if not platform.client:
            flash('Cannot initialize platform client. Please check your account connection.', 'warning')
            return redirect(url_for('engagement_bp.index'))
        
        # Post comment
        response = platform.post_comment(comment.target_post_id_on_platform, comment.suggested_comment_text)
        if response:
            comment.status = 'posted'
            comment.posted_at = datetime.now(timezone.utc)
            comment.platform_comment_id = response.get('id')
            db.session.commit()
            
            flash('Comment posted successfully!', 'success')
        else:
            flash('Failed to post comment. Please try again.', 'danger')
            
        return redirect(url_for('engagement_bp.index'))
        
    except Exception as e:
        current_app.logger.error(f"Error posting comment {comment_id}: {e}")
        flash('Error posting comment. Please try again.', 'danger')
        return redirect(url_for('engagement_bp.index'))

@engagement_bp.route('/comments/<int:comment_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_comment(comment_id):
    comment = GeneratedComment.query.get_or_404(comment_id)
    if comment.portal_user_id != current_user.id:
        flash('Unauthorized access to comment.', 'danger')
        return redirect(url_for('engagement_bp.index'))
    
    form = EditCommentForm()
    if form.validate_on_submit():
        if form.submit_edit.data:  # User edited the comment
            comment.suggested_comment_text = form.comment_text.data
            comment.status = 'pending_review'
            db.session.commit()
            flash('Comment updated successfully!', 'success')
        elif form.approve_direct.data:  # User approved the original comment
            return redirect(url_for('engagement_bp.approve_comment', comment_id=comment_id))
        
        return redirect(url_for('engagement_bp.index'))
    
    form.comment_text.data = comment.suggested_comment_text
    return render_template('engagement/edit_comment.html', form=form, comment=comment)

@engagement_bp.route('/comments/<int:comment_id>/reject')
@login_required
def reject_comment(comment_id):
    comment = GeneratedComment.query.get_or_404(comment_id)
    if comment.portal_user_id != current_user.id:
        flash('Unauthorized access to comment.', 'danger')
        return redirect(url_for('engagement_bp.index'))
    
    comment.status = 'rejected'
    db.session.commit()
    flash('Comment rejected successfully!', 'success')
    return redirect(url_for('engagement_bp.index'))

@engagement_bp.route('/log', methods=['GET'])
@login_required
def comments_log():
    page = request.args.get('page', 1, type=int)
    per_page = 15
    comments_query = db.select(GeneratedComment).join(GeneratedComment.posting_account)\
        .filter(ManagedAccount.portal_user_id == current_user.id)\
        .filter(GeneratedComment.status.in_(['posted', 'rejected', 'failed']))\
        .order_by(GeneratedComment.reviewed_at.desc().nullslast(), GeneratedComment.posted_at.desc().nullslast(), GeneratedComment.created_at.desc())
    comments_pagination = db.paginate(comments_query, page=page, per_page=per_page, error_out=False)
    logged_comments = comments_pagination.items
    
    return render_template('engagement/comments_log.html', 
                           title='Sent & Archived Comments Log', 
                           comments=logged_comments,
                           pagination=comments_pagination)

@engagement_bp.route('/review-comments', methods=['GET'])
@login_required
def review_comments_queue():
    page = request.args.get('page', 1, type=int)
    per_page = 10 

    comments_query = db.select(GeneratedComment).join(GeneratedComment.posting_account)\
        .filter(ManagedAccount.portal_user_id == current_user.id)\
        .filter(GeneratedComment.status == "pending_review")\
        .order_by(GeneratedComment.created_at.desc())
    
    comments_pagination = db.paginate(comments_query, page=page, per_page=per_page, error_out=False)
    comments_to_review = comments_pagination.items
    
    edit_form = EditCommentForm()

    return render_template('engagement/review_comments.html', 
                           title='Review Suggested Comments', 
                           comments=comments_to_review,
                           pagination=comments_pagination,
                           edit_form=edit_form)
    page = request.args.get('page', 1, type=int)
    per_page = 10 

    comments_query = db.select(GeneratedComment).join(GeneratedComment.posting_account)\
        .filter(ManagedAccount.portal_user_id == current_user.id)\
        .filter(GeneratedComment.status == "pending_review")\
        .order_by(GeneratedComment.created_at.desc())
    
    comments_pagination = db.paginate(comments_query, page=page, per_page=per_page, error_out=False)
    comments_to_review = comments_pagination.items
    
    edit_form = EditCommentForm()

    return render_template('engagement/review_comments.html', 
                           title='Review Suggested Comments', 
                           comments=comments_to_review,
                           pagination=comments_pagination,
                           edit_form=edit_form)

def get_managed_account_choices(portal_user_id, platform='X', add_blank=False):
    accounts = (
        ManagedAccount.query.filter_by(
            portal_user_id=portal_user_id,
            platform_name=platform,
            is_active=True
        )
        .order_by(ManagedAccount.account_display_name)
        .all()
    )
    
    choices = [(acc.id, acc.account_display_name or acc.account_id_on_platform) for acc in accounts]
    if add_blank:
        choices.insert(0, ('', '--- Select Account ---'))
    return choices
