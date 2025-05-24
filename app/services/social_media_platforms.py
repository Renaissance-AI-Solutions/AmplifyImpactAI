import abc
import tweepy 
import requests
import logging
from app.models import ManagedAccount
from app import db
from datetime import datetime, timezone, timedelta
from flask import current_app
from flask.globals import session
import json

logger = logging.getLogger(__name__)

class SocialMediaPlatform(abc.ABC):
    def __init__(self, managed_account: ManagedAccount = None):
        self.managed_account = managed_account
        self.client = None 
        if managed_account and managed_account.access_token:
            try:
                self._initialize_client()
            except Exception as e:
                logger.error(f"Failed to initialize client for {self.get_platform_name()} account {managed_account.id if managed_account else 'N/A'}: {e}")
                self.client = None
        elif managed_account and not managed_account.access_token:
            logger.warning(f"Cannot initialize client for {self.get_platform_name()} account {managed_account.id}: Access token missing.")
            self.client = None

    @abc.abstractmethod
    def _initialize_client(self):
        """Initializes the platform-specific API client using stored credentials."""
        pass

    @abc.abstractmethod
    def get_platform_name(self) -> str:
        """Returns the name of the platform (e.g., "X", "Facebook")."""
        pass
    
    @abc.abstractmethod
    def get_oauth_authorization_url(self) -> tuple[str, str | None]:
        """
        For OAuth 2.0 PKCE: Returns the authorization URL and the code_verifier.
        For OAuth 1.0a: Returns authorization URL and stores request token in session. Returns (auth_url, None).
        """
        pass

    @abc.abstractmethod
    def fetch_oauth_tokens(self, authorization_response_url: str = None, code: str = None, verifier: str = None, oauth_verifier: str = None) -> dict:
        """
        Fetches access tokens after user authorization.
        Returns a dictionary with 'access_token', 'refresh_token' (optional), 'expires_in' (optional),
        'access_token_secret' (for OAuth 1.0a), 'user_id_on_platform', 'screen_name'/'username'.
        """
        pass

    @abc.abstractmethod
    def refresh_access_token(self) -> bool:
        """Refreshes the access token if it's expired. Returns True on success."""
        pass

    @abc.abstractmethod
    def post_update(self, content: str, media_ids: list = None) -> dict:
        """Posts an update. Returns API response or identifier."""
        pass

    @abc.abstractmethod
    def post_comment(self, target_post_id: str, comment_text: str) -> dict:
        """Posts a comment/reply. Returns API response or identifier."""
        pass

    @abc.abstractmethod
    def search_posts(self, query: str, count: int = 10) -> list:
        """Searches for posts based on a query. Returns a list of posts."""
        pass
    
    @abc.abstractmethod
    def get_user_info(self) -> dict | None:
        """Gets basic info for the authenticated user (ID, username). Returns None on failure."""
        pass
    
    @abc.abstractmethod
    def validate_credentials(self) -> bool:
        """Validates if the current credentials for the client are working."""
        pass

class XPlatform(SocialMediaPlatform):
    PLATFORM_NAME = "X"

    def get_platform_name(self) -> str:
        return self.PLATFORM_NAME

    def _initialize_client(self):
        if not self.managed_account or not self.managed_account.access_token:
            logger.warning("XPlatform: Cannot initialize client, no access token for account %s", self.managed_account.id if self.managed_account else "N/A")
            self.client = None
            return

        consumer_key = current_app.config.get('X_CONSUMER_KEY')
        consumer_secret = current_app.config.get('X_CONSUMER_SECRET')
        access_token = self.managed_account.access_token
        access_token_secret = self.managed_account.access_token_secret

        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            logger.warning(f"XPlatform: Incomplete OAuth 1.0a credentials for account {self.managed_account.id if self.managed_account else 'N/A'}. Cannot initialize client.")
            self.client = None
            return
        
        try:
            self.client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret,
                wait_on_rate_limit=True
            )
            logger.info(f"XPlatform: Tweepy client initialized for account {self.managed_account.account_display_name or self.managed_account.id} using OAuth 1.0a tokens.")
        except tweepy.TweepyException as e:
            logger.error(f"XPlatform: TweepyException during client initialization for account {self.managed_account.id if self.managed_account else 'N/A'}: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"XPlatform: Unexpected error initializing Tweepy client for account {self.managed_account.id if self.managed_account else 'N/A'}: {e}")
            self.client = None

    def get_oauth_authorization_url(self) -> tuple[str, str | None]:
        consumer_key = current_app.config.get('X_CONSUMER_KEY')
        consumer_secret = current_app.config.get('X_CONSUMER_SECRET')
        callback_url = current_app.config.get('X_CALLBACK_URL')

        if not consumer_key or not consumer_secret:
            logger.error("X_CONSUMER_KEY or X_CONSUMER_SECRET not configured.")
            raise ValueError("X OAuth 1.0a application credentials not fully configured.")

        if not callback_url or callback_url.strip().lower() == 'oob' or not callback_url.startswith('http'):
            logger.error(f"X_CALLBACK_URL is invalid for redirect flow (current: '{callback_url}'). It must be a full HTTP/HTTPS URL and not 'oob'.")
            raise ValueError("X_CALLBACK_URL is not configured for redirect-based OAuth.")

        try:
            auth = tweepy.OAuth1UserHandler(
                consumer_key,
                consumer_secret,
                callback=callback_url
            )
            redirect_url = auth.get_authorization_url(signin_with_twitter=True)
            session['x_oauth_request_token_key'] = auth.request_token['oauth_token']
            session['x_oauth_request_token_secret'] = auth.request_token['oauth_token_secret']
            logger.info(f"X OAuth 1.0a: Generated auth URL with callback '{callback_url}'. Request token stored.")
            return redirect_url, None
        except tweepy.TweepyException as e:
            logger.error(f"TweepyException in get_oauth_authorization_url: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in get_oauth_authorization_url: {e}")
            raise

    def fetch_oauth_tokens(self, oauth_verifier: str = None, **kwargs) -> dict:
        request_token_key = session.pop('x_oauth_request_token_key', None)
        request_token_secret = session.pop('x_oauth_request_token_secret', None)

        if not request_token_key or not request_token_secret or not oauth_verifier:
            logger.error("X OAuth 1.0a: Request token key/secret or oauth_verifier missing from session/args.")
            raise ValueError("X OAuth 1.0a: Incomplete information for fetching access tokens.")

        consumer_key = current_app.config.get('X_CONSUMER_KEY')
        consumer_secret = current_app.config.get('X_CONSUMER_SECRET')

        if not all([consumer_key, consumer_secret]):
            logger.error("X OAuth 1.0a: Missing consumer credentials.")
            raise ValueError("X OAuth 1.0a: Missing consumer credentials.")

        # Create OAuth1UserHandler without request_token in constructor
        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret)
        
        # Set the request token as an attribute
        auth.request_token = {
            'oauth_token': request_token_key,
            'oauth_token_secret': request_token_secret
        }

        try:
            # Get access tokens using the verifier
            access_token, access_token_secret = auth.get_access_token(oauth_verifier)
            
            # Get user info
            client = tweepy.Client(
                consumer_key=consumer_key,
                consumer_secret=consumer_secret,
                access_token=access_token,
                access_token_secret=access_token_secret
            )
            
            user_info = client.get_me(user_auth=True)
            
            logger.info(f"X OAuth 1.0a: Successfully fetched access tokens for user {user_info.data.username} ({user_info.data.id}).")
            
            return {
                'access_token': access_token,
                'access_token_secret': access_token_secret,
                'user_id_on_platform': str(user_info.data.id),
                'screen_name': user_info.data.username,
                'display_name': user_info.data.name,
                'refresh_token': None,
                'expires_in': None
            }
        except tweepy.TweepyException as e:
            logger.error(f"TweepyException fetching X OAuth 1.0a access_token: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Twitter API response details: Status {e.response.status_code}, Text: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching X OAuth 1.0a access_token: {e}")
            raise

    def refresh_access_token(self) -> bool:
        # X OAuth 1.0a tokens don't expire, so no refresh needed
        return True

    def post_update(self, content: str, media_ids: list = None) -> dict:
        if not self.client:
            logger.error("XPlatform: Cannot post update - client not initialized")
            return {'error': 'Client not initialized'}

        try:
            if media_ids:
                response = self.client.create_tweet(
                    text=content,
                    media_ids=media_ids
                )
            else:
                response = self.client.create_tweet(
                    text=content
                )
            return response
        except Exception as e:
            logger.error(f"XPlatform: Error posting update: {e}")
            return {'error': str(e)}

    def post_comment(self, target_post_id: str, comment_text: str) -> dict:
        if not self.client:
            logger.error("XPlatform: Cannot post comment - client not initialized")
            return {'error': 'Client not initialized'}

        try:
            response = self.client.create_tweet(
                text=comment_text,
                in_reply_to_tweet_id=target_post_id
            )
            return response
        except Exception as e:
            logger.error(f"XPlatform: Error posting comment: {e}")
            return {'error': str(e)}

    def search_posts(self, query: str, count: int = 10) -> list:
        if not self.client:
            logger.error("XPlatform: Cannot search posts - client not initialized")
            return []

        try:
            tweets = self.client.search_recent_tweets(
                query=query,
                max_results=count
            )
            return tweets.data if tweets.data else []
        except Exception as e:
            logger.error(f"XPlatform: Error searching posts: {e}")
            return []

    def get_user_info(self) -> dict | None:
        if not self.client:
            logger.error("XPlatform: Cannot get user info - client not initialized")
            return None

        try:
            response = self.client.get_me(user_auth=True)
            if response.data:
                return {
                    'id': response.data.id,
                    'username': response.data.username,
                    'name': response.data.name,
                    'followers_count': response.data.public_metrics['followers_count'],
                    'following_count': response.data.public_metrics['following_count']
                }
            return None
        except Exception as e:
            logger.error(f"XPlatform: Error getting user info: {e}")
            return None

    def validate_credentials(self) -> bool:
        if not self.client:
            logger.error("XPlatform: Cannot validate credentials - client not initialized")
            return False

        try:
            response = self.client.get_me(user_auth=True)
            return response.data is not None
        except Exception as e:
            logger.error(f"XPlatform: Error validating credentials: {e}")
            return False

class InstagramPlatform(SocialMediaPlatform):
    PLATFORM_NAME = "Instagram"

    def get_platform_name(self) -> str:
        return self.PLATFORM_NAME

    def _initialize_client(self):
        if not self.managed_account or not self.managed_account.access_token:
            logger.warning("InstagramPlatform: Cannot initialize client, no access token for account %s", self.managed_account.id if self.managed_account else "N/A")
            self.client = None
            return

        # In a real implementation, you would initialize an Instagram API client here
        # For now, we'll create a minimal client structure to maintain compatibility
        self.client = {
            'access_token': self.managed_account.access_token,
            'user_id': self.managed_account.account_id_on_platform
        }
        
        # Log successful initialization
        logger.info(f"Instagram client initialized for account {self.managed_account.account_display_name}")

    def get_oauth_authorization_url(self) -> tuple[str, str | None]:
        # Instagram uses OAuth 2.0
        client_id = current_app.config.get('INSTAGRAM_CLIENT_ID')
        redirect_uri = current_app.config.get('INSTAGRAM_REDIRECT_URI')
        
        if not client_id or not redirect_uri:
            logger.error("InstagramPlatform: Missing Instagram API credentials in configuration")
            raise ValueError("Missing Instagram API configuration")
            
        # Scope for basic permissions (could be expanded as needed)
        scope = "user_profile,user_media"
        
        # Build the authorization URL
        auth_url = (f"https://api.instagram.com/oauth/authorize"
                   f"?client_id={client_id}"
                   f"&redirect_uri={redirect_uri}"
                   f"&scope={scope}"
                   f"&response_type=code")
        
        return auth_url, None  # No code verifier needed for basic OAuth 2.0

    def fetch_oauth_tokens(self, authorization_response_url: str = None, code: str = None, verifier: str = None, oauth_verifier: str = None) -> dict:
        client_id = current_app.config.get('INSTAGRAM_CLIENT_ID')
        client_secret = current_app.config.get('INSTAGRAM_CLIENT_SECRET')
        redirect_uri = current_app.config.get('INSTAGRAM_REDIRECT_URI')
        
        if not all([client_id, client_secret, redirect_uri, code]):
            logger.error("InstagramPlatform: Missing required parameters for token exchange")
            raise ValueError("Missing required parameters for Instagram token exchange")
            
        try:
            # Exchange authorization code for access token
            token_url = "https://api.instagram.com/oauth/access_token"
            data = {
                'client_id': client_id,
                'client_secret': client_secret,
                'grant_type': 'authorization_code',
                'redirect_uri': redirect_uri,
                'code': code
            }
            
            response = requests.post(token_url, data=data)
            response.raise_for_status()
            token_data = response.json()
            
            # Get user info using the short-lived access token
            user_id = token_data.get('user_id')
            access_token = token_data.get('access_token')
            
            # Optional: Exchange for a long-lived token
            # (In a real implementation, you would do this)
            
            # Get additional user info
            user_info = self._get_user_info_with_token(user_id, access_token)
            
            return {
                'access_token': access_token,
                'user_id_on_platform': user_id,
                'username': user_info.get('username', ''),
                'display_name': user_info.get('full_name', '')
            }
            
        except requests.RequestException as e:
            logger.error(f"InstagramPlatform: Error fetching OAuth tokens: {e}")
            raise
        except Exception as e:
            logger.error(f"InstagramPlatform: Unexpected error fetching OAuth tokens: {e}")
            raise
    
    def _get_user_info_with_token(self, user_id, access_token):
        # This would use the Instagram Graph API to get user info
        # For now, returning mock data
        return {
            'id': user_id,
            'username': f"instagram_user_{user_id}",
            'full_name': f"Instagram User {user_id}"
        }

    def refresh_access_token(self) -> bool:
        # In a real implementation, you would refresh the Instagram access token
        # For now, we'll assume it's valid
        return True

    def post_update(self, content: str, media_ids: list = None) -> dict:
        if not self.client:
            logger.error("InstagramPlatform: Cannot post update - client not initialized")
            return {'error': 'Client not initialized'}
            
        try:
            # In a real implementation, you would post to Instagram
            # For now, return mock response
            return {
                'id': f"ig_post_{datetime.now(timezone.utc).timestamp()}",
                'status': 'success',
                'message': 'Instagram post created successfully (mock)'
            }
        except Exception as e:
            logger.error(f"InstagramPlatform: Error posting update: {e}")
            return {'error': str(e)}

    def post_comment(self, target_post_id: str, comment_text: str) -> dict:
        if not self.client:
            logger.error("InstagramPlatform: Cannot post comment - client not initialized")
            return {'error': 'Client not initialized'}
            
        try:
            # In a real implementation, you would post a comment to Instagram
            # For now, return mock response
            return {
                'id': f"ig_comment_{datetime.now(timezone.utc).timestamp()}",
                'status': 'success',
                'message': 'Instagram comment created successfully (mock)'
            }
        except Exception as e:
            logger.error(f"InstagramPlatform: Error posting comment: {e}")
            return {'error': str(e)}

    def search_posts(self, query: str, count: int = 10) -> list:
        if not self.client:
            logger.error("InstagramPlatform: Cannot search posts - client not initialized")
            return []
            
        try:
            # In a real implementation, you would search Instagram posts
            # For now, return empty list as Instagram API has limited search capabilities
            return []
        except Exception as e:
            logger.error(f"InstagramPlatform: Error searching posts: {e}")
            return []

    def get_user_info(self) -> dict | None:
        if not self.client:
            logger.error("InstagramPlatform: Cannot get user info - client not initialized")
            return None
            
        try:
            # In a real implementation, you would fetch user info from Instagram
            # For now, return mock data
            return {
                'id': self.managed_account.account_id_on_platform,
                'username': self.managed_account.account_display_name or f"instagram_user_{self.managed_account.account_id_on_platform}",
                'name': self.managed_account.account_display_name or f"Instagram User {self.managed_account.account_id_on_platform}",
                'followers_count': 1000,  # Mock data
                'following_count': 500    # Mock data
            }
        except Exception as e:
            logger.error(f"InstagramPlatform: Error getting user info: {e}")
            return None

    def validate_credentials(self) -> bool:
        if not self.client:
            logger.error("InstagramPlatform: Cannot validate credentials - client not initialized")
            return False
            
        try:
            # In a real implementation, you would validate the Instagram credentials
            # For now, assume they're valid if client is initialized
            return True
        except Exception as e:
            logger.error(f"InstagramPlatform: Error validating credentials: {e}")
            return False

# Platform class mapping
PLATFORM_CLASS_MAP = {
    'X': XPlatform,
    'Instagram': InstagramPlatform
}

def get_platform_connector_by_name(platform_name: str) -> SocialMediaPlatform | None:
    platform_class = PLATFORM_CLASS_MAP.get(platform_name.capitalize())
    if platform_class:
        try:
            return platform_class()
        except Exception as e:
            logger.error(f"Error instantiating platform connector for {platform_name}: {e}")
    return None
