from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import PortalUser, ManagedAccount, KnowledgeDocument, ScheduledPost, GeneratedComment, ActionLog, ApiKey, CommentAutomationSetting, RecurringPostSchedule
from app.forms import PostComposerForm, SchedulePostForm, CommentSettingsForm, ApiKeyForm, RecurringPostScheduleForm
from app.services.social_media_platforms import XPlatform
from app.services.knowledge_base_manager import KnowledgeBaseManager
from app.services.analytics_service import AnalyticsService
from datetime import datetime, timezone

main_bp = Blueprint('main_bp', __name__)

@main_bp.route('/')
@login_required
def index_redirect():
    return redirect(url_for('main_bp.dashboard'))

@main_bp.route('/dashboard')
@login_required
def dashboard():
    # Get user statistics
    account_count = ManagedAccount.query.filter_by(portal_user_id=current_user.id, is_active=True).count()
    scheduled_posts_count = ScheduledPost.query.filter_by(portal_user_id=current_user.id).count()
    pending_comments_count = db.session.scalar(
    db.select(db.func.count(GeneratedComment.id))
    .join(GeneratedComment.posting_account)
    .filter(ManagedAccount.portal_user_id == current_user.id, GeneratedComment.status == 'pending_review')
)
    
    # Get recent activity
    recent_activity = (
        ActionLog.query.filter_by(portal_user_id=current_user.id)
        .order_by(ActionLog.timestamp.desc())
        .limit(20)
        .all()
    )
    
    return render_template('main/dashboard.html',
                         account_count=account_count,
                         scheduled_posts_count=scheduled_posts_count,
                         pending_comments_count=pending_comments_count,
                         recent_activity=recent_activity)

@main_bp.route('/post-composer', methods=['GET', 'POST'])
@login_required
def post_composer():
    form = PostComposerForm()
    schedule_form = SchedulePostForm()
    
    # Populate account choices
    form.target_account_id.choices = get_managed_account_choices(current_user.id)
    schedule_form.target_account_id_schedule.choices = get_managed_account_choices(current_user.id)
    
    if form.validate_on_submit():
        if form.post_now.data:  # Post immediately
            return self._handle_immediate_post(form)
        elif form.schedule_post_action.data:  # Schedule post
            return self._handle_schedule_post(form, schedule_form)
        elif form.generate_with_ai.data:  # Generate with AI
            return self._handle_ai_generation(form)
    
    return render_template('post_composer.html', form=form, schedule_form=schedule_form)

@main_bp.route('/comment-manager')
@login_required
def comment_manager():
    pending_comments = db.session.scalars(
    db.select(GeneratedComment)
    .join(GeneratedComment.posting_account)
    .filter(ManagedAccount.portal_user_id == current_user.id, GeneratedComment.status == 'pending_review')
).all()
    
    approved_comments = db.session.scalars(
    db.select(GeneratedComment)
    .join(GeneratedComment.posting_account)
    .filter(ManagedAccount.portal_user_id == current_user.id, GeneratedComment.status == 'approved')
    .order_by(GeneratedComment.posted_at.desc())
    .limit(10)
).all()
    
    return render_template('comment_manager.html',
                         pending_comments=pending_comments,
                         approved_comments=approved_comments)

@main_bp.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    comment_settings_form = CommentSettingsForm()
    api_key_form = ApiKeyForm()
    
    # Pre-populate comment settings form with current settings if they exist
    comment_settings = CommentAutomationSetting.query.filter_by(portal_user_id=current_user.id).first()
    if comment_settings:
        comment_settings_form.keywords.data = comment_settings.keywords
        comment_settings_form.monitored_x_accounts.data = comment_settings.monitored_x_accounts
        comment_settings_form.default_posting_account_id.choices = get_managed_account_choices(current_user.id, add_blank=True)
        comment_settings_form.default_posting_account_id.data = comment_settings.default_posting_managed_account_id
        comment_settings_form.is_active.data = comment_settings.is_active
    else:
        comment_settings_form.default_posting_account_id.choices = get_managed_account_choices(current_user.id, add_blank=True)
    
    # Pre-populate API key form with current API keys if they exist
    api_keys = ApiKey.query.filter_by(portal_user_id=current_user.id).first()
    if api_keys and api_keys.openai_api_key:
        # Mask the API key for display (show only last 4 characters)
        masked_key = '••••••••' + api_keys.openai_api_key[-4:] if len(api_keys.openai_api_key) > 4 else '••••'
        api_key_form.openai_api_key.render_kw = {'placeholder': masked_key}
    
    # Handle form submissions
    if request.method == 'POST':
        # Handle Comment Settings form
        if 'submit' in request.form and comment_settings_form.validate():
            if not comment_settings:
                comment_settings = CommentAutomationSetting(portal_user_id=current_user.id)
                db.session.add(comment_settings)
            
            comment_settings.keywords = comment_settings_form.keywords.data
            comment_settings.monitored_x_accounts = comment_settings_form.monitored_x_accounts.data
            comment_settings.default_posting_managed_account_id = comment_settings_form.default_posting_account_id.data or None
            comment_settings.is_active = comment_settings_form.is_active.data
            
            db.session.commit()
            flash('Comment settings saved successfully!', 'success')
            return redirect(url_for('main_bp.settings'))
            
        # Handle API Key form
        elif 'submit' in request.form and api_key_form.validate():
            if not api_keys:
                api_keys = ApiKey(portal_user_id=current_user.id)
                db.session.add(api_keys)
            
            # Only update if a new key is provided
            if api_key_form.openai_api_key.data:
                api_keys.openai_api_key = api_key_form.openai_api_key.data
            
            db.session.commit()
            flash('API key saved successfully!', 'success')
            return redirect(url_for('main_bp.settings'))
        comment_settings_form.is_active.data = settings.is_active
    
    return render_template('settings.html', comment_settings_form=comment_settings_form)


@main_bp.route('/analytics')
@login_required
def analytics():
    # Initialize analytics service
    analytics_service = AnalyticsService(current_user.id)
    
    # Get analytics data
    stats = analytics_service.get_overview_stats(days=30)
    post_volume_data = analytics_service.get_post_volume_data(days=30)
    comment_distribution_data = analytics_service.get_comment_distribution_data()
    account_performance = analytics_service.get_account_performance()
    top_content = analytics_service.get_top_performing_content(limit=5)
    
    return render_template('analytics.html',
                           stats=stats,
                           post_volume_data=post_volume_data,
                           comment_distribution_data=comment_distribution_data,
                           account_performance=account_performance,
                           top_content=top_content)

@main_bp.route('/api/generate-comment', methods=['POST'])
@login_required
def api_generate_comment():
    data = request.json
    if not data or not all(key in data for key in ['post_content', 'post_author', 'post_context']):
        return {'error': 'Missing required parameters'}, 400
    
    try:
        # Initialize knowledge base manager
        kbm = KnowledgeBaseManager()
        
        # Generate comment
        comment = kbm.generate_comment(
            post_content=data['post_content'],
            post_author=data['post_author'],
            post_context=data['post_context']
        )
        
        return {'comment': comment}
    except Exception as e:
        current_app.logger.error(f"Error generating comment: {e}")
        return {'error': str(e)}, 500

@main_bp.route('/api/schedule-comment', methods=['POST'])
@login_required
def api_schedule_comment():
    data = request.json
    if not data or not all(key in data for key in ['comment_id', 'target_post_id', 'target_platform']):
        return {'error': 'Missing required parameters'}, 400
    
    try:
        # Get comment
        comment = GeneratedComment.query.get(data['comment_id'])
        if not comment or comment.portal_user_id != current_user.id:
            return {'error': 'Comment not found or unauthorized'}, 404
        
        # Get posting account
        account = ManagedAccount.query.get(comment.managed_account_id_to_post_from)
        if not account or not account.is_active:
            return {'error': 'Posting account not found or inactive'}, 404
        
        # Initialize platform
        platform = XPlatform(account)
        if not platform.client:
            return {'error': 'Cannot initialize platform client'}, 500
        
        # Post comment
        response = platform.post_comment(data['target_post_id'], comment.suggested_comment_text)
        if response:
            comment.status = 'posted'
            comment.posted_at = datetime.now(timezone.utc)
            comment.platform_comment_id = response.get('id')
            db.session.commit()
            
            # Log action
            action_log = ActionLog(
                portal_user_id=current_user.id,
                managed_account_id=account.id,
                action_type='comment_posted',
                status='SUCCESS',
                details=f"Posted comment on post {data['target_post_id']}"
            )
            db.session.add(action_log)
            db.session.commit()
            
            return {'status': 'success', 'comment_id': comment.id}
        else:
            return {'error': 'Failed to post comment'}, 500
            
    except Exception as e:
        current_app.logger.error(f"Error scheduling comment: {e}")
        return {'error': str(e)}, 500

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

def _handle_immediate_post(form):
    """Handle immediate post submission."""
    try:
        # Get selected account
        account = ManagedAccount.query.get(form.target_account_id.data)
        if not account or not account.is_active:
            flash('Selected account is not active', 'error')
            return redirect(url_for('main_bp.post_composer'))
        
        # Initialize platform
        platform = XPlatform(account)
        if not platform.client:
            flash('Cannot initialize platform client', 'error')
            return redirect(url_for('main_bp.post_composer'))
        
        # Post content
        response = platform.post_update(form.content.data)
        if response:
            # Create post record
            post = ScheduledPost(
                managed_account_id=account.id,
                content=form.content.data,
                status='posted',
                posted_at=datetime.now(timezone.utc),
                platform_post_id=response.get('id')
            )
            db.session.add(post)
            db.session.commit()
            
            flash('Post published successfully!', 'success')
        else:
            flash('Failed to publish post', 'error')
            
        return redirect(url_for('main_bp.post_composer'))
        
    except Exception as e:
        current_app.logger.error(f"Error posting immediately: {e}")
        flash('Error publishing post', 'error')
        return redirect(url_for('main_bp.post_composer'))

def _handle_schedule_post(form, schedule_form):
    """Handle scheduled post submission."""
    if not schedule_form.validate_on_submit():
        return redirect(url_for('main_bp.post_composer'))
        
    try:
        # Create scheduled post
        post = ScheduledPost(
            managed_account_id=schedule_form.target_account_id_schedule.data,
            content=schedule_form.content_schedule.data,
            scheduled_time=schedule_form.scheduled_time.data,
            status='pending'
        )
        db.session.add(post)
        db.session.commit()
        
        # Schedule the post
        from flask import current_app
        current_app.scheduler_service.schedule_post(post.id)
        
        flash('Post scheduled successfully!', 'success')
        return redirect(url_for('main_bp.post_composer'))
        
    except Exception as e:
        current_app.logger.error(f"Error scheduling post: {e}")
        flash('Error scheduling post', 'error')
        return redirect(url_for('main_bp.post_composer'))

def _handle_ai_generation(form):
    """Handle AI content generation."""
    try:
        # Initialize knowledge base manager
        kbm = KnowledgeBaseManager()
        
        # Generate content
        content = kbm.generate_content(
            topic=form.topic.data,
            tone=form.tone.data,
            style=form.style.data
        )
        
        # Update form with generated content
        form.generated_content_display.data = content
        
        flash('AI content generated successfully!', 'success')
        return render_template('post_composer.html', form=form, schedule_form=schedule_form)
        
    except Exception as e:
        current_app.logger.error(f"Error generating AI content: {e}")
        flash('Error generating AI content', 'error')
        return render_template('post_composer.html', form=form, schedule_form=schedule_form)

def _get_engagement_stats():
    """Get engagement statistics."""
    # Implementation would depend on your analytics data
    return {
        'total_posts': 0,
        'total_comments': 0,
        'average_engagement': 0.0,
        'top_performing_content': []
    }

def _get_performance_metrics():
    """Get performance metrics."""
    # Implementation would depend on your analytics data
    return {
        'comment_success_rate': 0.0,
        'post_success_rate': 0.0,
        'average_response_time': 0.0,
        'top_performing_accounts': []
    }

@main_bp.route('/recurring-posts')
@login_required
def recurring_posts():
    """View and manage recurring post schedules."""
    schedules = RecurringPostSchedule.query.filter_by(portal_user_id=current_user.id).all()
    
    # Get the count of posts created from each schedule
    for schedule in schedules:
        schedule.post_count = ScheduledPost.query.filter_by(
            recurring_schedule_id=schedule.id,
            is_from_recurring_schedule=True
        ).count()
    
    return render_template('recurring_posts.html', schedules=schedules)

@main_bp.route('/recurring-posts/new', methods=['GET', 'POST'])
@login_required
def new_recurring_post():
    """Create a new recurring post schedule."""
    form = RecurringPostScheduleForm()
    form.target_account_id.choices = get_managed_account_choices(current_user.id)
    
    if form.validate_on_submit():
        try:
            # Create new recurring schedule
            schedule = RecurringPostSchedule(
                portal_user_id=current_user.id,
                managed_account_id=form.target_account_id.data,
                name=form.name.data,
                content_template=form.content_template.data,
                frequency=form.frequency.data,
                time_of_day=form.time_of_day.data,
                is_active=form.is_active.data
            )
            
            # Set day_of_week or day_of_month based on frequency
            if form.frequency.data == 'weekly':
                schedule.day_of_week = form.day_of_week.data
            elif form.frequency.data == 'monthly':
                schedule.day_of_month = form.day_of_month.data
            
            db.session.add(schedule)
            db.session.commit()
            
            # Schedule the recurring post
            current_app.scheduler_service.schedule_recurring_post(schedule.id)
            
            flash('Recurring post schedule created successfully!', 'success')
            return redirect(url_for('main_bp.recurring_posts'))
            
        except Exception as e:
            current_app.logger.error(f"Error creating recurring post schedule: {e}")
            flash('Error creating recurring post schedule', 'error')
    
    return render_template('recurring_post_form.html', form=form, is_edit=False)

@main_bp.route('/recurring-posts/edit/<int:schedule_id>', methods=['GET', 'POST'])
@login_required
def edit_recurring_post(schedule_id):
    """Edit an existing recurring post schedule."""
    schedule = RecurringPostSchedule.query.filter_by(id=schedule_id, portal_user_id=current_user.id).first_or_404()
    
    form = RecurringPostScheduleForm(obj=schedule)
    form.target_account_id.choices = get_managed_account_choices(current_user.id)
    
    if form.validate_on_submit():
        try:
            # Update schedule with form data
            schedule.name = form.name.data
            schedule.managed_account_id = form.target_account_id.data
            schedule.content_template = form.content_template.data
            schedule.frequency = form.frequency.data
            schedule.time_of_day = form.time_of_day.data
            schedule.is_active = form.is_active.data
            
            # Reset day_of_week and day_of_month
            schedule.day_of_week = None
            schedule.day_of_month = None
            
            # Set day_of_week or day_of_month based on frequency
            if form.frequency.data == 'weekly':
                schedule.day_of_week = form.day_of_week.data
            elif form.frequency.data == 'monthly':
                schedule.day_of_month = form.day_of_month.data
            
            schedule.updated_at = datetime.now(timezone.utc)
            db.session.commit()
            
            # Update the schedule in the scheduler
            current_app.scheduler_service.update_recurring_schedule(schedule.id)
            
            flash('Recurring post schedule updated successfully!', 'success')
            return redirect(url_for('main_bp.recurring_posts'))
            
        except Exception as e:
            current_app.logger.error(f"Error updating recurring post schedule: {e}")
            flash('Error updating recurring post schedule', 'error')
    
    return render_template('recurring_post_form.html', form=form, is_edit=True, schedule=schedule)

@main_bp.route('/recurring-posts/toggle/<int:schedule_id>', methods=['POST'])
@login_required
def toggle_recurring_post(schedule_id):
    """Toggle the active status of a recurring post schedule."""
    schedule = RecurringPostSchedule.query.filter_by(id=schedule_id, portal_user_id=current_user.id).first_or_404()
    
    try:
        schedule.is_active = not schedule.is_active
        db.session.commit()
        
        # Update the schedule in the scheduler
        current_app.scheduler_service.update_recurring_schedule(schedule.id)
        
        status = 'activated' if schedule.is_active else 'deactivated'
        flash(f'Recurring post schedule {status} successfully!', 'success')
        
        return jsonify({'success': True, 'is_active': schedule.is_active})
        
    except Exception as e:
        current_app.logger.error(f"Error toggling recurring post schedule: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@main_bp.route('/recurring-posts/delete/<int:schedule_id>', methods=['POST'])
@login_required
def delete_recurring_post(schedule_id):
    """Delete a recurring post schedule."""
    schedule = RecurringPostSchedule.query.filter_by(id=schedule_id, portal_user_id=current_user.id).first_or_404()
    
    try:
        # Remove the schedule from the scheduler
        job_id = f'recurring_{schedule.id}'
        try:
            current_app.scheduler_service.scheduler.remove_job(job_id)
        except Exception:
            # Job might not exist, which is fine
            pass
        
        # Delete the schedule
        db.session.delete(schedule)
        db.session.commit()
        
        flash('Recurring post schedule deleted successfully!', 'success')
        return jsonify({'success': True})
        
    except Exception as e:
        current_app.logger.error(f"Error deleting recurring post schedule: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
