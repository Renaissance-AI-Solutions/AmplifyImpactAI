from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, session
from flask_login import login_required, current_user
from app.models import PortalUser, ManagedAccount, ActionLog
from app.services.social_media_platforms import get_platform_connector_by_name
from datetime import datetime, timezone
from app import db

accounts_bp = Blueprint('accounts_bp', __name__)

@accounts_bp.route('/add/<platform_name>')
@login_required
def add_account_start(platform_name):
    """Start the account connection process for the specified platform."""
    platform_name_clean = platform_name.capitalize()
    
    # Get platform-specific connector
    platform_connector = get_platform_connector_by_name(platform_name_clean)
    if not platform_connector:
        flash(f"Platform '{platform_name_clean}' is not supported.", "danger")
        return redirect(url_for('.manage_accounts'))
    
    try:
        auth_url, code_verifier_or_none = platform_connector.get_oauth_authorization_url()
        session['oauth_platform_name'] = platform_name_clean
        
        # Handle platform-specific session data
        if platform_name_clean.lower() == 'x':
            if code_verifier_or_none:
                session['x_oauth_code_verifier'] = code_verifier_or_none
        
        return redirect(auth_url)
    except ValueError as ve:
        flash(f"Configuration error for {platform_name_clean}: {str(ve)}", "danger")
        return redirect(url_for('.manage_accounts'))
    except Exception as e:
        flash(f"Error initiating authentication with {platform_name_clean}: {str(e)}", "danger")
        return redirect(url_for('.manage_accounts'))

@accounts_bp.route('/')
@login_required
def manage_accounts():
    accounts = ManagedAccount.query.filter_by(portal_user_id=current_user.id).all()
    return render_template('accounts/manage_accounts.html', accounts=accounts)

@accounts_bp.route('/<platform_name>/connect')
@login_required
def connect_platform(platform_name):
    """Start OAuth flow for any supported platform."""
    # Special case for backward compatibility
    if platform_name.lower() == 'x':
        return redirect(url_for('.add_account_start', platform_name='x'))
        
    platform_name_clean = platform_name.capitalize()
    platform_connector = get_platform_connector_by_name(platform_name_clean)
    
    if not platform_connector:
        flash(f"Platform '{platform_name_clean}' is not supported.", "danger")
        return redirect(url_for('.manage_accounts'))
        
    try:
        auth_url, code_verifier = platform_connector.get_oauth_authorization_url()
        # Store platform name in session to use in callback
        session['oauth_platform_name'] = platform_name_clean
        
        # Store code verifier if provided (for PKCE flows)
        if code_verifier:
            session[f'{platform_name.lower()}_code_verifier'] = code_verifier
            
        return redirect(auth_url)
    except Exception as e:
        current_app.logger.error(f"Error getting {platform_name_clean} OAuth URL: {e}")
        flash(f'Error connecting to {platform_name_clean}. Please try again.', 'danger')
        return redirect(url_for('.manage_accounts'))

@accounts_bp.route('/x/callback')
def x_callback():
    """Handle X OAuth 1.0a callback (legacy, for backward compatibility)."""
    oauth_verifier = request.args.get('oauth_verifier')
    if not oauth_verifier:
        flash('OAuth verification failed. Please try again.', 'danger')
        return redirect(url_for('accounts_bp.index'))
    
    platform = XPlatform()
    try:
        tokens = platform.fetch_oauth_tokens(oauth_verifier=oauth_verifier)
        
        # Create or update ManagedAccount
        account = ManagedAccount(
            portal_user_id=current_user.id,
            platform_name='X',
            account_id_on_platform=tokens['user_id_on_platform'],
            account_display_name=tokens['screen_name'],
            encrypted_access_token=tokens['access_token'],
            encrypted_access_token_secret=tokens['access_token_secret']
        )
        db.session.add(account)
        db.session.commit()
        
        # Log action
        action_log = ActionLog(
            portal_user_id=current_user.id,
            managed_account_id=account.id,
            action_type='account_connected',
            status='SUCCESS',
            details=f"Connected X account {tokens['screen_name']}"
        )
        db.session.add(action_log)
        db.session.commit()
        
        flash('Successfully connected X account!', 'success')
        return redirect(url_for('accounts_bp.index'))
        
    except Exception as e:
        current_app.logger.error(f"Error processing X OAuth callback: {e}")
        flash('Error connecting X account. Please try again.', 'danger')
        return redirect(url_for('accounts_bp.index'))

@accounts_bp.route('/callback/<platform_name_route>')
@login_required
def oauth_callback(platform_name_route):
    current_app.logger.info(f"OAuth callback for platform: {platform_name_route}, Args: {request.args}")

    if request.args.get('denied'):
        flash(f"{platform_name_route.capitalize()} authentication denied by user.", "warning")
        session.pop('x_oauth_request_token_key', None)
        session.pop('x_oauth_request_token_secret', None)
        session.pop('oauth_platform_name', None)
        return redirect(url_for('.manage_accounts'))
    
    # Ensure we have required authorization parameters
    if not (oauth_verifier or code):
        flash('OAuth verification failed. Missing required parameters.', 'danger')
        session.pop('oauth_platform_name', None)
        return redirect(url_for('.manage_accounts'))

    try:
        platform_name_clean = platform_name_route.capitalize()
        platform_connector = get_platform_connector_by_name(platform_name_clean)
        if not platform_connector:
            flash(f"Platform connector for {platform_name_clean} not found.", "danger")
            return redirect(url_for('.manage_accounts'))

        # Get code verifier from session if available (for PKCE flow)
        code_verifier = session.get(f'{platform_name_route.lower()}_code_verifier')
        
        # Different parameter sets for different auth types
        if oauth_verifier:  # OAuth 1.0a (X/Twitter)
            token_data = platform_connector.fetch_oauth_tokens(oauth_verifier=oauth_verifier)
        elif code:  # OAuth 2.0 (Instagram, etc.)
            token_data = platform_connector.fetch_oauth_tokens(code=code, verifier=code_verifier)
        else:
            raise ValueError("No valid OAuth parameters found")
        
        # Check if account already exists for this user and platform
        existing_account = db.session.scalar(
            db.select(ManagedAccount)
            .filter_by(portal_user_id=current_user.id, 
                     platform_name=platform_name_clean,
                     account_id_on_platform=token_data['user_id_on_platform'])
        )
        
        if existing_account:
            # Update existing account
            existing_account.account_display_name = token_data.get('username') or token_data.get('screen_name') or token_data.get('display_name', '')
            existing_account.is_active = True
            existing_account.last_validated_at = datetime.now(timezone.utc)
            
            # Update tokens based on what's available
            existing_account.set_tokens(
                access_token=token_data.get('access_token'),
                access_token_secret=token_data.get('access_token_secret'),
                refresh_token=token_data.get('refresh_token'),
                expires_at=token_data.get('expires_at')
            )
            
            account = existing_account
            action_type = 'account_reconnected'
            flash_message = f"{platform_name_clean} account reconnected successfully!"
        else:
            # Create new account
            account = ManagedAccount(
                portal_user_id=current_user.id,
                platform_name=platform_name_clean,
                account_id_on_platform=token_data['user_id_on_platform'],
                account_display_name=token_data.get('username') or token_data.get('screen_name') or token_data.get('display_name', ''),
                is_active=True,
                last_validated_at=datetime.now(timezone.utc)
            )
            
            # Set tokens based on what's available
            account.set_tokens(
                access_token=token_data.get('access_token'),
                access_token_secret=token_data.get('access_token_secret'),
                refresh_token=token_data.get('refresh_token'),
                expires_at=token_data.get('expires_at')
            )
            
            db.session.add(account)
            action_type = 'account_connected'
            flash_message = f"{platform_name_clean} account connected successfully!"
        
        # Commit account changes
        db.session.commit()
        
        # Log the action
        display_name = account.account_display_name or account.account_id_on_platform
        action_log = ActionLog(
            portal_user_id=current_user.id,
            managed_account_id=account.id,
            action_type=action_type,
            status='SUCCESS',
            details=f"{action_type.replace('_', ' ').title()}: {platform_name_clean} account {display_name}"
        )
        db.session.add(action_log)
        db.session.commit()
        
        flash(flash_message, "success")
    except Exception as e:
        current_app.logger.error(f"Error processing OAuth callback for {platform_name_route}: {e}", exc_info=True)
        flash(f"An error occurred connecting your {platform_name_route.capitalize()} account: {str(e)}", "danger")
    finally:
        # Clean up session data
        session.pop(f'{platform_name_route.lower()}_code_verifier', None)
        session.pop('oauth_platform_name', None)

    return redirect(url_for('.manage_accounts'))

@accounts_bp.route('/<int:account_id>/validate')
@login_required
def validate_account(account_id):
    """Validate account credentials."""
    account = ManagedAccount.query.get_or_404(account_id)
    if account.portal_user_id != current_user.id:
        flash('Unauthorized access to account.', 'danger')
        return redirect(url_for('.manage_accounts'))
    
    # Get platform-specific connector based on the account's platform
    platform_connector = get_platform_connector_by_name(account.platform_name)
    if not platform_connector:
        flash(f"Platform connector for {account.platform_name} not found.", "danger")
        return redirect(url_for('.manage_accounts'))
        
    # Initialize with account credentials
    platform_connector.managed_account = account
    
    try:
        if platform_connector.validate_credentials():
            account.last_validated_at = datetime.now(timezone.utc)
            db.session.commit()
            flash(f'{account.platform_name} account credentials are valid!', 'success')
        else:
            flash(f'{account.platform_name} account credentials are invalid. Please reconnect the account.', 'warning')
            account.is_active = False
            db.session.commit()
        
        return redirect(url_for('.manage_accounts'))
        
    except Exception as e:
        current_app.logger.error(f"Error validating account {account_id}: {e}")
        flash(f'Error validating {account.platform_name} account credentials.', 'danger')
        return redirect(url_for('.manage_accounts'))

@accounts_bp.route('/<int:account_id>/disconnect')
@login_required
def disconnect_account(account_id):
    """Disconnect account."""
    account = ManagedAccount.query.get_or_404(account_id)
    if account.portal_user_id != current_user.id:
        flash('Unauthorized access to account.', 'danger')
        return redirect(url_for('.manage_accounts'))
    
    try:
        # Log action
        action_log = ActionLog(
            portal_user_id=current_user.id,
            managed_account_id=account.id,
            action_type='account_disconnected',
            status='SUCCESS',
            details=f"Disconnected {account.platform_name} account {account.account_display_name or account.account_id_on_platform}"
        )
        db.session.add(action_log)
        
        # Delete account
        db.session.delete(account)
        db.session.commit()
        
        flash(f'{account.platform_name} account disconnected successfully!', 'success')
        return redirect(url_for('.manage_accounts'))
        
    except Exception as e:
        current_app.logger.error(f"Error disconnecting account {account_id}: {e}")
        flash(f'Error disconnecting {account.platform_name} account.', 'danger')
        return redirect(url_for('.manage_accounts'))
