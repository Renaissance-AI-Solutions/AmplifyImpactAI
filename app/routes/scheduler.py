from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify, current_app
from flask_login import login_required, current_user
from app import db
from app.models import ScheduledPost, ManagedAccount, RecurringPostSchedule
from app.forms import SchedulePostForm, RecurringPostScheduleForm, BulkScheduleForm
from app.services.scheduler_service import scheduler_service
from datetime import datetime, timedelta, timezone
import json

scheduler_bp = Blueprint('scheduler_bp', __name__)

@scheduler_bp.route('/content-calendar')
@login_required
def content_calendar():
    """Content calendar view showing all scheduled posts."""
    # Get all active accounts for the filter
    accounts = ManagedAccount.query.filter_by(
        portal_user_id=current_user.id,
        is_active=True
    ).all()
    
    return render_template('content_calendar.html', accounts=accounts)

@scheduler_bp.route('/api/scheduled-posts')
@login_required
def get_scheduled_posts():
    """API endpoint to get scheduled posts for the calendar."""
    # Get filter parameters
    platforms = request.args.get('platforms', '').split(',') if request.args.get('platforms') else None
    statuses = request.args.get('statuses', '').split(',') if request.args.get('statuses') else None
    account_id = request.args.get('account_id')
    
    # Build query
    query = db.select(ScheduledPost).join(
        ManagedAccount,
        ScheduledPost.managed_account_id == ManagedAccount.id
    ).filter(ManagedAccount.portal_user_id == current_user.id)
    
    # Apply filters
    if platforms and '' not in platforms:
        query = query.filter(ManagedAccount.platform_name.in_(platforms))
    
    if statuses and '' not in statuses:
        query = query.filter(ScheduledPost.status.in_(statuses))
    
    if account_id and account_id != 'all':
        query = query.filter(ScheduledPost.managed_account_id == int(account_id))
    
    # Execute query
    posts = db.session.execute(query).scalars().all()
    
    # Format posts for FullCalendar
    events = []
    for post in posts:
        account = ManagedAccount.query.get(post.managed_account_id)
        
        # Create event object
        event = {
            'id': str(post.id),
            'title': (post.content[:30] + '...') if len(post.content) > 30 else post.content,
            'start': post.scheduled_time.isoformat(),
            'end': (post.scheduled_time + timedelta(minutes=30)).isoformat(),
            'extendedProps': {
                'postId': post.id,
                'accountName': account.account_display_name or f"Account {account.id}",
                'platform': account.platform_name,
                'content': post.content,
                'status': post.status,
                'postedTime': post.posted_at.isoformat() if post.posted_at else None
            }
        }
        
        # Set color based on status
        if post.status == 'pending':
            event['backgroundColor'] = '#ffc107'
        elif post.status == 'posted':
            event['backgroundColor'] = '#28a745'
        elif post.status == 'failed':
            event['backgroundColor'] = '#dc3545'
        else:
            event['backgroundColor'] = '#6c757d'
        
        events.append(event)
    
    return jsonify(events)

@scheduler_bp.route('/bulk-schedule', methods=['GET', 'POST'])
@login_required
def bulk_schedule():
    """Bulk post scheduling interface."""
    form = BulkScheduleForm()
    
    # Populate account choices
    form.target_account_id.choices = [
        (account.id, f"{account.account_display_name} ({account.platform_name})")
        for account in ManagedAccount.query.filter_by(
            portal_user_id=current_user.id,
            is_active=True
        ).all()
    ]
    
    return render_template('bulk_schedule.html', form=form)

@scheduler_bp.route('/bulk-schedule/submit', methods=['POST'])
@login_required
def bulk_schedule_submit():
    """Process bulk scheduling form submission."""
    form = BulkScheduleForm()
    
    # Populate account choices for validation
    form.target_account_id.choices = [
        (account.id, f"{account.account_display_name} ({account.platform_name})")
        for account in ManagedAccount.query.filter_by(
            portal_user_id=current_user.id,
            is_active=True
        ).all()
    ]
    
    if form.validate_on_submit():
        try:
            # Get form data
            target_account_id = form.target_account_id.data
            post_contents = request.form.getlist('post_content[]')
            
            # Get scheduling method
            scheduling_method = request.form.get('schedulingMethod')
            
            scheduled_posts = []
            
            if scheduling_method == 'sequential':
                # Sequential scheduling
                start_datetime_str = request.form.get('startDateTime')
                post_interval = int(request.form.get('postInterval', 24))
                interval_unit = request.form.get('intervalUnit', 'hours')
                skip_weekends = 'skipWeekends' in request.form
                
                # Parse start datetime
                start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M')
                start_datetime = start_datetime.replace(tzinfo=timezone.utc)
                
                # Schedule each post
                current_datetime = start_datetime
                for content in post_contents:
                    if not content.strip():
                        continue
                    
                    # Skip weekends if requested
                    if skip_weekends and current_datetime.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                        # Move to Monday
                        days_to_add = 7 - current_datetime.weekday() + 0  # Monday = 0
                        current_datetime += timedelta(days=days_to_add)
                    
                    # Create scheduled post
                    post = ScheduledPost(
                        portal_user_id=current_user.id,
                        managed_account_id=target_account_id,
                        content=content,
                        scheduled_time=current_datetime,
                        status='pending'
                    )
                    db.session.add(post)
                    scheduled_posts.append(post)
                    
                    # Calculate next datetime
                    if interval_unit == 'hours':
                        current_datetime += timedelta(hours=post_interval)
                    elif interval_unit == 'days':
                        current_datetime += timedelta(days=post_interval)
                    elif interval_unit == 'weeks':
                        current_datetime += timedelta(weeks=post_interval)
                
            else:
                # Custom scheduling
                custom_times = request.form.getlist('custom_time[]')
                
                # Create posts with custom times
                for i, content in enumerate(post_contents):
                    if not content.strip() or i >= len(custom_times):
                        continue
                    
                    # Parse custom datetime
                    custom_datetime = datetime.strptime(custom_times[i], '%Y-%m-%d %H:%M')
                    custom_datetime = custom_datetime.replace(tzinfo=timezone.utc)
                    
                    # Create scheduled post
                    post = ScheduledPost(
                        portal_user_id=current_user.id,
                        managed_account_id=target_account_id,
                        content=content,
                        scheduled_time=custom_datetime,
                        status='pending'
                    )
                    db.session.add(post)
                    scheduled_posts.append(post)
            
            # Commit all posts
            db.session.commit()
            
            # Schedule posts with the scheduler service
            for post in scheduled_posts:
                scheduler_service.schedule_post(post.id)
            
            flash(f'Successfully scheduled {len(scheduled_posts)} posts!', 'success')
            return redirect(url_for('scheduler_bp.content_calendar'))
            
        except Exception as e:
            current_app.logger.error(f"Error in bulk scheduling: {e}")
            db.session.rollback()
            flash(f'Error scheduling posts: {str(e)}', 'danger')
    
    return render_template('bulk_schedule.html', form=form)

@scheduler_bp.route('/edit-post/<int:post_id>', methods=['GET', 'POST'])
@login_required
def edit_post(post_id):
    """Edit a scheduled post."""
    post = db.session.get(ScheduledPost, post_id)
    
    if not post or post.portal_user_id != current_user.id:
        flash('Post not found or you do not have permission to edit it.', 'danger')
        return redirect(url_for('scheduler_bp.content_calendar'))
    
    form = SchedulePostForm()
    
    # Populate account choices
    form.target_account_id_schedule.choices = [
        (account.id, f"{account.account_display_name} ({account.platform_name})")
        for account in ManagedAccount.query.filter_by(
            portal_user_id=current_user.id,
            is_active=True
        ).all()
    ]
    
    if form.validate_on_submit():
        # Update post data
        post.managed_account_id = form.target_account_id_schedule.data
        post.content = form.content_schedule.data
        post.scheduled_time = form.scheduled_time.data.replace(tzinfo=timezone.utc)
        
        # If the post status is failed, reset it to pending
        if post.status == 'failed':
            post.status = 'pending'
            post.error_message = None
        
        db.session.commit()
        
        # Reschedule the post
        scheduler_service.schedule_post(post.id)
        
        flash('Post updated successfully!', 'success')
        return redirect(url_for('scheduler_bp.content_calendar'))
    
    # Populate form with existing data
    form.target_account_id_schedule.data = post.managed_account_id
    form.content_schedule.data = post.content
    form.scheduled_time.data = post.scheduled_time
    
    return render_template('edit_post.html', form=form, post=post)

@scheduler_bp.route('/delete-post/<int:post_id>', methods=['POST'])
@login_required
def delete_post(post_id):
    """Delete a scheduled post."""
    post = db.session.get(ScheduledPost, post_id)
    
    if not post or post.portal_user_id != current_user.id:
        return jsonify({'success': False, 'error': 'Post not found or you do not have permission to delete it.'})
    
    try:
        # Remove from scheduler if pending
        if post.status == 'pending':
            try:
                scheduler_service.scheduler.remove_job(f'post_{post.id}')
            except Exception:
                # Job might not exist, which is fine
                pass
        
        # Delete from database
        db.session.delete(post)
        db.session.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        current_app.logger.error(f"Error deleting post {post_id}: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})
