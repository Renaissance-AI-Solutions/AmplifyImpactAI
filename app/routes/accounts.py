from flask import Blueprint, render_template, redirect, url_for, flash, request, current_app, session
from flask_login import login_required, current_user
from app.models import PortalUser, ManagedAccount, ActionLog
from app.services.social_media_platforms import XPlatform, get_platform_connector_by_name
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

@accounts_bp.route('/x/connect')
def connect_x():
    """Start X (Twitter) OAuth 1.0a flow."""
    platform = XPlatform()
    try:
        auth_url, _ = platform.get_oauth_authorization_url()
        return redirect(auth_url)
    except Exception as e:
        current_app.logger.error(f"Error getting X OAuth URL: {e}")
        flash('Error connecting to X. Please try again.', 'danger')
        return redirect(url_for('accounts_bp.index'))

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

    oauth_verifier = request.args.get('oauth_verifier')
    if not oauth_verifier:
        flash(f"X authentication failed: No verifier token received in callback for {platform_name_route}.", "danger")
        current_app.logger.error(f"OAuth callback for {platform_name_route} missing oauth_verifier. Args: {request.args}")
        session.pop('x_oauth_request_token_key', None)
        session.pop('x_oauth_request_token_secret', None)
        session.pop('oauth_platform_name', None)
        return redirect(url_for('.manage_accounts'))

    try:
        platform_connector = get_platform_connector_by_name(platform_name_route.capitalize())
        if not platform_connector:
            flash(f"Platform connector for {platform_name_route} not found.", "danger")
            return redirect(url_for('.manage_accounts'))

        token_data = platform_connector.fetch_oauth_tokens(oauth_verifier=oauth_verifier)
        account = ManagedAccount(
            portal_user_id=current_user.id,
            platform_name=platform_name_route.capitalize(),
            account_id_on_platform=token_data['user_id_on_platform'],
            account_display_name=token_data['screen_name'],
            encrypted_access_token=token_data['access_token'],
            encrypted_access_token_secret=token_data['access_token_secret']
        )
        db.session.add(account)
        db.session.commit()
        action_log = ActionLog(
            portal_user_id=current_user.id,
            managed_account_id=account.id,
            action_type='account_connected',
            status='SUCCESS',
            details=f"Connected {platform_name_route.capitalize()} account {token_data['screen_name']}"
        )
        db.session.add(action_log)
        db.session.commit()
        flash(f"{platform_name_route.capitalize()} account connected successfully!", "success")
    except Exception as e:
        current_app.logger.error(f"Error processing OAuth callback for {platform_name_route}: {e}", exc_info=True)
        flash(f"An error occurred connecting your {platform_name_route.capitalize()} account: {str(e)}", "danger")
    finally:
        session.pop('x_oauth_request_token_key', None)
        session.pop('x_oauth_request_token_secret', None)
        session.pop('oauth_platform_name', None)

    return redirect(url_for('.manage_accounts'))

@accounts_bp.route('/<int:account_id>/validate')
@login_required
def validate_account(account_id):
    """Validate account credentials."""
    account = ManagedAccount.query.get_or_404(account_id)
    if account.portal_user_id != current_user.id:
        flash('Unauthorized access to account.', 'danger')
        return redirect(url_for('accounts_bp.index'))
    
    platform = XPlatform(account)
    try:
        if platform.validate_credentials():
            flash('Account credentials are valid!', 'success')
        else:
            flash('Account credentials are invalid. Please reconnect the account.', 'warning')
            account.is_active = False
            db.session.commit()
        
        return redirect(url_for('accounts_bp.index'))
        
    except Exception as e:
        current_app.logger.error(f"Error validating account {account_id}: {e}")
        flash('Error validating account credentials.', 'danger')
        return redirect(url_for('accounts_bp.index'))

@accounts_bp.route('/<int:account_id>/disconnect')
@login_required
def disconnect_account(account_id):
    """Disconnect account."""
    account = ManagedAccount.query.get_or_404(account_id)
    if account.portal_user_id != current_user.id:
        flash('Unauthorized access to account.', 'danger')
        return redirect(url_for('accounts_bp.index'))
    
    try:
        # Log action
        action_log = ActionLog(
            portal_user_id=current_user.id,
            managed_account_id=account.id,
            action_type='account_disconnected',
            status='SUCCESS',
            details=f"Disconnected account {account.account_display_name or account.account_id_on_platform}"
        )
        db.session.add(action_log)
        
        # Delete account
        db.session.delete(account)
        db.session.commit()
        
        flash('Account disconnected successfully!', 'success')
        return redirect(url_for('accounts_bp.index'))
        
    except Exception as e:
        current_app.logger.error(f"Error disconnecting account {account_id}: {e}")
        flash('Error disconnecting account.', 'danger')
        return redirect(url_for('accounts_bp.index'))
