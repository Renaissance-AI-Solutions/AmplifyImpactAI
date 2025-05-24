import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, desc, and_
from app import db
from app.models import ScheduledPost, ManagedAccount, GeneratedComment, ActionLog

logger = logging.getLogger(__name__)

class AnalyticsService:
    """Service for collecting and processing analytics data."""
    
    def __init__(self, portal_user_id: int):
        """Initialize the analytics service for a specific user."""
        self.portal_user_id = portal_user_id
        
    def get_overview_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get overview statistics for the dashboard."""
        try:
            # Define time periods
            now = datetime.now(timezone.utc)
            current_period_start = now - timedelta(days=days)
            previous_period_start = current_period_start - timedelta(days=days)
            
            # Get current period stats
            current_posts = self._get_post_count(current_period_start, now)
            current_comments = self._get_comment_count(current_period_start, now)
            current_engagement_rate = self._calculate_engagement_rate(current_period_start, now)
            current_avg_post_length = self._calculate_avg_post_length(current_period_start, now)
            current_avg_response_time = self._calculate_avg_response_time(current_period_start, now)
            
            # Get previous period stats
            previous_posts = self._get_post_count(previous_period_start, current_period_start)
            previous_comments = self._get_comment_count(previous_period_start, current_period_start)
            previous_engagement_rate = self._calculate_engagement_rate(previous_period_start, current_period_start)
            previous_avg_post_length = self._calculate_avg_post_length(previous_period_start, current_period_start)
            previous_avg_response_time = self._calculate_avg_response_time(previous_period_start, current_period_start)
            
            # Calculate changes
            posts_change = self._calculate_percentage_change(previous_posts, current_posts)
            comments_change = self._calculate_percentage_change(previous_comments, current_comments)
            engagement_change = self._calculate_percentage_change(previous_engagement_rate, current_engagement_rate)
            length_change = self._calculate_percentage_change(previous_avg_post_length, current_avg_post_length)
            response_time_change = self._calculate_percentage_change(previous_avg_response_time, current_avg_response_time, inverse=True)
            
            return {
                'total_posts': current_posts,
                'total_comments': current_comments,
                'engagement_rate': round(current_engagement_rate, 1),
                'avg_post_length': current_avg_post_length,
                'avg_response_time': current_avg_response_time,
                
                'prev_total_posts': previous_posts,
                'prev_total_comments': previous_comments,
                'prev_engagement_rate': round(previous_engagement_rate, 1),
                'prev_avg_post_length': previous_avg_post_length,
                'prev_avg_response_time': previous_avg_response_time,
                
                'posts_change': posts_change,
                'comments_change': comments_change,
                'engagement_change': engagement_change,
                'length_change': length_change,
                'response_time_change': response_time_change
            }
            
        except Exception as e:
            logger.error(f"Error getting overview stats: {e}", exc_info=True)
            return self._get_default_stats()
            
    def get_post_volume_data(self, days: int = 30) -> Dict[str, List]:
        """Get post volume data for charting."""
        try:
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=days)
            
            # Get posts by day
            posts_by_day = db.session.query(
                func.date(ScheduledPost.posted_at).label('post_date'),
                func.count(ScheduledPost.id).label('post_count')
            ).join(
                ManagedAccount, ScheduledPost.managed_account_id == ManagedAccount.id
            ).filter(
                ManagedAccount.portal_user_id == self.portal_user_id,
                ScheduledPost.status == 'posted',
                ScheduledPost.posted_at >= start_date
            ).group_by(
                func.date(ScheduledPost.posted_at)
            ).order_by(
                func.date(ScheduledPost.posted_at)
            ).all()
            
            # Create date range for all days
            date_range = []
            current_date = start_date
            while current_date <= now:
                date_range.append(current_date.date())
                current_date += timedelta(days=1)
                
            # Map post counts to dates
            post_counts = [0] * len(date_range)
            date_dict = {d.date(): i for i, d in enumerate(date_range)}
            
            for post_date, post_count in posts_by_day:
                if post_date in date_dict:
                    post_counts[date_dict[post_date]] = post_count
                    
            # Format dates for display
            formatted_dates = [d.strftime('%b %d') for d in date_range]
            
            return {
                'labels': formatted_dates,
                'values': post_counts
            }
            
        except Exception as e:
            logger.error(f"Error getting post volume data: {e}", exc_info=True)
            return {'labels': [], 'values': []}
            
    def get_comment_distribution_data(self) -> Dict[str, List]:
        """Get comment distribution data for charting."""
        try:
            # Get comment counts by status
            comment_distribution = db.session.query(
                GeneratedComment.status,
                func.count(GeneratedComment.id).label('count')
            ).join(
                ManagedAccount, GeneratedComment.managed_account_id_to_post_from == ManagedAccount.id
            ).filter(
                ManagedAccount.portal_user_id == self.portal_user_id
            ).group_by(
                GeneratedComment.status
            ).all()
            
            # Map status to readable labels
            status_labels = {
                'pending_review': 'Pending Review',
                'approved': 'Approved',
                'posted': 'Posted',
                'failed': 'Failed',
                'rejected': 'Rejected'
            }
            
            labels = []
            values = []
            
            for status, count in comment_distribution:
                labels.append(status_labels.get(status, status.capitalize()))
                values.append(count)
                
            return {
                'labels': labels,
                'values': values
            }
            
        except Exception as e:
            logger.error(f"Error getting comment distribution data: {e}", exc_info=True)
            return {'labels': [], 'values': []}
            
    def get_account_performance(self) -> List[Dict[str, Any]]:
        """Get performance metrics by account."""
        try:
            # Get all active accounts
            accounts = ManagedAccount.query.filter_by(
                portal_user_id=self.portal_user_id,
                is_active=True
            ).all()
            
            now = datetime.now(timezone.utc)
            start_date = now - timedelta(days=30)
            
            account_metrics = []
            
            for account in accounts:
                # Get post count
                post_count = db.session.scalar(
                    db.select(func.count(ScheduledPost.id))
                    .filter(
                        ScheduledPost.managed_account_id == account.id,
                        ScheduledPost.status == 'posted',
                        ScheduledPost.posted_at >= start_date
                    )
                ) or 0
                
                # Get comment count (comments made by this account)
                comment_count = db.session.scalar(
                    db.select(func.count(GeneratedComment.id))
                    .filter(
                        GeneratedComment.managed_account_id_to_post_from == account.id,
                        GeneratedComment.status == 'posted',
                        GeneratedComment.posted_at >= start_date
                    )
                ) or 0
                
                # Calculate engagement rate (simplified)
                engagement_rate = round((comment_count / post_count * 100) if post_count > 0 else 0, 1)
                
                # Calculate average response time
                avg_response_time = self._calculate_account_avg_response_time(account.id, start_date, now)
                
                account_metrics.append({
                    'account_id': account.id,
                    'account_display_name': account.account_display_name or f"Account {account.id}",
                    'account_id_on_platform': account.account_id_on_platform,
                    'platform_name': account.platform_name,
                    'post_count': post_count,
                    'comment_count': comment_count,
                    'engagement_rate': engagement_rate,
                    'avg_response_time': avg_response_time
                })
                
            # Sort by engagement rate (descending)
            account_metrics.sort(key=lambda x: x['engagement_rate'], reverse=True)
            
            return account_metrics
            
        except Exception as e:
            logger.error(f"Error getting account performance: {e}", exc_info=True)
            return []
            
    def get_top_performing_content(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing content based on engagement."""
        try:
            # Get the most engaged posts based on available metrics
            top_posts = db.session.query(
                ScheduledPost,
                ManagedAccount.platform_name,
                ManagedAccount.account_display_name
            ).join(
                ManagedAccount, ScheduledPost.managed_account_id == ManagedAccount.id
            ).filter(
                ManagedAccount.portal_user_id == self.portal_user_id,
                ScheduledPost.status == 'posted'
            ).order_by(
                ScheduledPost.posted_at.desc()
            ).limit(limit).all()
            
            # For actual metrics, we would pull engagement data from the social platforms
            # For now, we'll generate some representative data
            result = []
            for post, platform_name, account_name in top_posts:
                # Generate some placeholder metrics based on content length and post ID
                # In a real implementation, these would come from the platform's API
                content_length = len(post.content)
                post_id_int = int(post.id)
                
                likes = (post_id_int % 50) + (content_length // 20)
                comments = (post_id_int % 10) + (content_length // 100)
                shares = (post_id_int % 5) + (content_length // 200)
                
                # Calculate engagement rate (likes + comments + shares) / estimated impressions
                estimated_impressions = 100 + (likes * 5)
                engagement_rate = round(((likes + comments + shares) / estimated_impressions) * 100, 1)
                
                # Create a truncated version of the content for display
                truncated_content = post.content[:100] + '...' if len(post.content) > 100 else post.content
                
                result.append({
                    'post_id': post.id,
                    'platform': platform_name,
                    'account': account_name or f"Account {post.managed_account_id}",
                    'content': truncated_content,
                    'posted_at': post.posted_at,
                    'likes': likes,
                    'comments': comments,
                    'shares': shares,
                    'engagement_rate': engagement_rate,
                    'platform_post_id': post.platform_post_id,
                })
            
            # Sort by engagement rate descending
            result.sort(key=lambda x: x['engagement_rate'], reverse=True)
            return result
            
        except Exception as e:
            logger.error(f"Error getting top performing content: {e}", exc_info=True)
            return []
            
    def _get_post_count(self, start_date: datetime, end_date: datetime) -> int:
        """Get the number of posts in a given period."""
        try:
            return db.session.scalar(
                db.select(func.count(ScheduledPost.id))
                .join(ManagedAccount, ScheduledPost.managed_account_id == ManagedAccount.id)
                .filter(
                    ManagedAccount.portal_user_id == self.portal_user_id,
                    ScheduledPost.status == 'posted',
                    ScheduledPost.posted_at >= start_date,
                    ScheduledPost.posted_at < end_date
                )
            ) or 0
            
        except Exception as e:
            logger.error(f"Error getting post count: {e}", exc_info=True)
            return 0
            
    def _get_comment_count(self, start_date: datetime, end_date: datetime) -> int:
        """Get the number of comments in a given period."""
        try:
            return db.session.scalar(
                db.select(func.count(GeneratedComment.id))
                .join(ManagedAccount, GeneratedComment.managed_account_id_to_post_from == ManagedAccount.id)
                .filter(
                    ManagedAccount.portal_user_id == self.portal_user_id,
                    GeneratedComment.status == 'posted',
                    GeneratedComment.posted_at >= start_date,
                    GeneratedComment.posted_at < end_date
                )
            ) or 0
            
        except Exception as e:
            logger.error(f"Error getting comment count: {e}", exc_info=True)
            return 0
            
    def _calculate_engagement_rate(self, start_date: datetime, end_date: datetime) -> float:
        """Calculate the engagement rate for a given period."""
        try:
            post_count = self._get_post_count(start_date, end_date)
            comment_count = self._get_comment_count(start_date, end_date)
            
            # Simple engagement rate calculation
            if post_count > 0:
                return round((comment_count / post_count) * 100, 1)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating engagement rate: {e}", exc_info=True)
            return 0.0
            
    def _calculate_avg_post_length(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate the average post length in words for a given period."""
        try:
            result = db.session.query(
                func.avg(func.length(ScheduledPost.content) - 
                        func.length(func.replace(ScheduledPost.content, ' ', '')) + 1)
            ).join(
                ManagedAccount, ScheduledPost.managed_account_id == ManagedAccount.id
            ).filter(
                ManagedAccount.portal_user_id == self.portal_user_id,
                ScheduledPost.status == 'posted',
                ScheduledPost.posted_at >= start_date,
                ScheduledPost.posted_at < end_date
            ).scalar()
            
            return int(result) if result else 0
            
        except Exception as e:
            logger.error(f"Error calculating average post length: {e}", exc_info=True)
            return 0
            
    def _calculate_avg_response_time(self, start_date: datetime, end_date: datetime) -> int:
        """Calculate the average response time in minutes for a given period."""
        try:
            # This would typically require data from the social media platforms
            # For now, we'll return a placeholder value
            return 15
            
        except Exception as e:
            logger.error(f"Error calculating average response time: {e}", exc_info=True)
            return 0
            
    def _calculate_account_avg_response_time(self, account_id: int, start_date: datetime, end_date: datetime) -> int:
        """Calculate the average response time for a specific account."""
        try:
            # This would typically require data from the social media platforms
            # For now, we'll return a placeholder value based on the account ID
            return 10 + (account_id % 20)  # Random-ish value between 10-30
            
        except Exception as e:
            logger.error(f"Error calculating account average response time: {e}", exc_info=True)
            return 0
            
    def _calculate_percentage_change(self, previous: float, current: float, inverse: bool = False) -> int:
        """Calculate percentage change between two values."""
        if previous == 0:
            return 100 if current > 0 else 0
            
        change = ((current - previous) / previous) * 100
        
        # For metrics where a decrease is positive (like response time)
        if inverse and change != 0:
            change = -change
            
        return round(change)
        
    def _get_default_stats(self) -> Dict[str, Any]:
        """Return default stats when data cannot be retrieved."""
        return {
            'total_posts': 0,
            'total_comments': 0,
            'engagement_rate': 0.0,
            'avg_post_length': 0,
            'avg_response_time': 0,
            
            'prev_total_posts': 0,
            'prev_total_comments': 0,
            'prev_engagement_rate': 0.0,
            'prev_avg_post_length': 0,
            'prev_avg_response_time': 0,
            
            'posts_change': 0,
            'comments_change': 0,
            'engagement_change': 0,
            'length_change': 0,
            'response_time_change': 0
        }
