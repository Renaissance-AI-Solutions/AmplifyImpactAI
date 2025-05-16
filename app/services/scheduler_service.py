from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
from sqlalchemy import inspect as sqlalchemy_inspect

logger = logging.getLogger(__name__)

class SchedulerService:
    def __init__(self):
        self.scheduler = None
        self.app = None

    def init_app(self, app):
        from flask import current_app
        from app.models import ScheduledPost, GeneratedComment, CommentAutomationSetting
        self.ScheduledPost = ScheduledPost
        self.GeneratedComment = GeneratedComment
        self.CommentAutomationSetting = CommentAutomationSetting
        from app import db
        self.db = db
        from app.services.social_media_platforms import XPlatform

        self.app = app
        jobstores = {
            'default': SQLAlchemyJobStore(url=app.config['SQLALCHEMY_DATABASE_URI'])
        }
        executors = {
            'default': ThreadPoolExecutor(max_workers=5)
        }
        job_defaults = {
            'coalesce': False,
            'max_instances': 3
        }
        self.scheduler = BackgroundScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=timezone.utc
        )
        if self.scheduler and not self.scheduler.running:
            if app.config.get('SCHEDULER_API_ENABLED', False):
                try:
                    self.scheduler.start()
                    logger.info("APScheduler started via init_app.")
                    with app.app_context():
                        # Check if the necessary tables exist before trying to query them
                        inspector = sqlalchemy_inspect(db.engine)
                        if inspector.has_table("scheduled_post") and inspector.has_table("apscheduler_jobs"):
                            self._add_existing_scheduled_posts()
                            self._add_existing_comment_automation_jobs()
                        else:
                            logger.warning(
                                "Skipping task rescheduling during app init: "
                                "'scheduled_post' or 'apscheduler_jobs' table not found. "
                                "This is normal during initial DB setup or migrations."
                            )
                except Exception as e:
                    logger.error(f"Failed to start APScheduler or reschedule tasks in init_app: {e}", exc_info=True)
            else:
                logger.info("APScheduler API is disabled in config. Scheduler not started.")
        elif self.scheduler and self.scheduler.running:
             logger.info("APScheduler already running.")
        elif not self.scheduler:
            logger.error("SchedulerService: Scheduler could not be initialized. Scheduling disabled.")

    def schedule_post(self, post_id: int):
        """Schedule a post to be published at its scheduled time."""
        with self.app.app_context():
            post = db.session.get(ScheduledPost, post_id)
            if not post or post.status != 'pending':
                logger.warning(f"Cannot schedule post {post_id}: Post not found or not pending.")
                return

            job_id = f'post_{post_id}'
            self.scheduler.add_job(
                self._publish_post,
                'date',
                run_date=post.scheduled_time,
                args=[post_id],
                id=job_id,
                replace_existing=True
            )
            logger.info(f"Scheduled post {post_id} for {post.scheduled_time}")

    def _publish_post(self, post_id: int):
        """Publish a scheduled post."""
        with self.app.app_context():
            post = db.session.get(ScheduledPost, post_id)
            if not post or post.status != 'pending':
                logger.warning(f"Cannot publish post {post_id}: Post not found or not pending.")
                return

            try:
                account = post.managed_account
                if not account or not account.is_active:
                    raise ValueError("Account is inactive or not found.")

                platform = XPlatform(account)
                if not platform.client:
                    raise ValueError("Cannot initialize platform client.")

                # Publish the post
                response = platform.post_update(post.content, post.media_urls.split(',') if post.media_urls else None)
                if response:
                    post.status = 'posted'
                    post.posted_at = datetime.now(timezone.utc)
                    post.platform_post_id = response.get('id')
                    self.db.session.commit()
                    logger.info(f"Successfully published post {post_id}")
                else:
                    post.status = 'failed'
                    post.error_message = "Failed to get response from platform API"
                    self.db.session.commit()
                    logger.error(f"Failed to publish post {post_id}: No response from platform")

            except Exception as e:
                post.status = 'failed'
                post.error_message = str(e)
                self.db.session.commit()
                logger.error(f"Error publishing post {post_id}: {e}")

    def _add_existing_scheduled_posts(self):
        """Add existing scheduled posts to the scheduler."""
        posts = self.db.session.scalars(
            self.db.select(self.ScheduledPost).filter_by(status='pending')
        ).all()
        for post in posts:
            self.schedule_post(post.id)

    def schedule_comment_discovery(self, setting_id: int):
        """Schedule comment discovery job for a comment automation setting."""
        job_id = f'comment_discovery_{setting_id}'
        self.scheduler.add_job(
            self._run_comment_discovery,
            'interval',
            minutes=30,  # Run every 30 minutes
            args=[setting_id],
            id=job_id,
            replace_existing=True
        )
        logger.info(f"Scheduled comment discovery for setting {setting_id}")

    def _run_comment_discovery(self, setting_id: int):
        """Run comment discovery job for a comment automation setting."""
        with self.app.app_context():
            setting = db.session.get(CommentAutomationSetting, setting_id)
            if not setting or not setting.is_active:
                logger.warning(f"Cannot run comment discovery for setting {setting_id}: Setting not found or not active.")
                return

            try:
                # Get the default posting account
                posting_account = setting.default_posting_account
                if not posting_account or not posting_account.is_active:
                    raise ValueError("Default posting account is inactive or not found.")

                # Initialize platform
                platform = XPlatform(posting_account)
                if not platform.client:
                    raise ValueError("Cannot initialize platform client.")

                # Search for posts matching keywords
                posts = platform.search_posts(
                    query=setting.keywords,
                    count=20  # Limit to 20 posts per run
                )

                # Generate comments for each post
                for post in posts:
                    try:
                        # Check if we already have a comment for this post
                        existing_comment = self.db.session.scalar(
                            self.db.select(self.GeneratedComment)
                            .filter_by(target_post_id_on_platform=post['id'])
                        )
                        if existing_comment:
                            continue

                        # Generate comment using AI (implementation would depend on your AI service)
                        comment_text = self._generate_comment_for_post(post)
                        
                        # Create GeneratedComment record
                        comment = GeneratedComment(
                            managed_account_id_to_post_from=posting_account.id,
                            target_platform='X',
                            target_post_id_on_platform=post['id'],
                            target_post_content=post['text'],
                            target_post_author=post['author'],
                            target_post_url=post['url'],
                            suggested_comment_text=comment_text,
                            status='pending_review'
                        )
                        db.session.add(comment)
                        self.db.session.commit()
                        logger.info(f"Generated comment for post {post['id']}")

                    except Exception as e:
                        logger.error(f"Error processing post {post['id']}: {e}")

                # Update last_run_at
                setting.last_run_at = datetime.now(timezone.utc)
                self.db.session.commit()

            except Exception as e:
                logger.error(f"Error running comment discovery for setting {setting_id}: {e}")

    def _generate_comment_for_post(self, post):
        """Generate a comment for a post using AI (implementation would depend on your AI service)."""
        # This is a placeholder - you would implement your actual AI comment generation logic here
        return "Great post! I really appreciate your insights on this topic."

    def _add_existing_comment_automation_jobs(self):
        """Add existing active comment automation settings to the scheduler."""
        settings = self.db.session.scalars(
            self.db.select(self.CommentAutomationSetting).filter_by(is_active=True)
        ).all()
        for setting in settings:
            self.schedule_comment_discovery(setting.id)
