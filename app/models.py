from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager
from app.utils.encryption import encrypt_token, decrypt_token

class PortalUser(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True, nullable=False)
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    password_hash = db.Column(db.String(256))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<PortalUser {self.username}>'

@login_manager.user_loader
def load_user(id):
    return db.session.get(PortalUser, int(id))

class ManagedAccount(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portal_user_id = db.Column(db.Integer, db.ForeignKey('portal_user.id'), nullable=False)
    platform_name = db.Column(db.String(50), nullable=False)
    account_id_on_platform = db.Column(db.String(255), nullable=False, index=True)
    account_display_name = db.Column(db.String(255))

    encrypted_access_token = db.Column(db.String(1024))
    encrypted_access_token_secret = db.Column(db.String(1024))
    encrypted_refresh_token = db.Column(db.String(1024))
    token_expires_at = db.Column(db.DateTime)

    is_active = db.Column(db.Boolean, default=True)
    last_validated_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    portal_user = db.relationship('PortalUser', backref=db.backref('managed_accounts', lazy=True))
    scheduled_posts = db.relationship('ScheduledPost', backref='managed_account', lazy='dynamic', cascade="all, delete-orphan")
    action_logs = db.relationship('ActionLog', backref='managed_account', lazy='dynamic', cascade="all, delete-orphan")

    def set_tokens(self, access_token, access_token_secret=None, refresh_token=None, expires_at=None):
        self.encrypted_access_token = encrypt_token(access_token) if access_token else None
        self.encrypted_access_token_secret = encrypt_token(access_token_secret) if access_token_secret else None
        self.encrypted_refresh_token = encrypt_token(refresh_token) if refresh_token else None
        self.token_expires_at = expires_at

    @property
    def access_token(self):
        return decrypt_token(self.encrypted_access_token) if self.encrypted_access_token else None

    @property
    def access_token_secret(self):
        return decrypt_token(self.encrypted_access_token_secret) if self.encrypted_access_token_secret else None

    @property
    def refresh_token(self):
        return decrypt_token(self.encrypted_refresh_token) if self.encrypted_refresh_token else None
    
    def __repr__(self):
        return f'<ManagedAccount {self.platform_name}:{self.account_display_name}>'

class KnowledgeDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portal_user_id = db.Column(db.Integer, db.ForeignKey('portal_user.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255))
    file_type = db.Column(db.String(10))
    status = db.Column(db.String(50), default="uploaded")
    processed_at = db.Column(db.DateTime)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    portal_user = db.relationship('PortalUser', backref=db.backref('knowledge_documents', lazy=True))
    chunks = db.relationship('KnowledgeChunk', backref='document', lazy='dynamic', cascade="all, delete-orphan")

    def __repr__(self):
        return f'<KnowledgeDocument {self.original_filename}>'

class KnowledgeChunk(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('knowledge_document.id'), nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    faiss_index_id = db.Column(db.Integer, index=True, nullable=True) 
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))

    def __repr__(self):
        return f'<KnowledgeChunk {self.id} for Doc {self.document_id}>'

class ScheduledPost(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portal_user_id = db.Column(db.Integer, db.ForeignKey('portal_user.id'), nullable=False, index=True)
    managed_account_id = db.Column(db.Integer, db.ForeignKey('managed_account.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    media_urls = db.Column(db.String(1024)) 
    scheduled_time = db.Column(db.DateTime, nullable=False, index=True)
    status = db.Column(db.String(50), default="pending")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    posted_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    platform_post_id = db.Column(db.String(255), nullable=True)

    owner = db.relationship('PortalUser', backref=db.backref('owned_scheduled_posts', lazy='dynamic'))

    def __repr__(self):
        return f'<ScheduledPost {self.id} for acc {self.managed_account_id} at {self.scheduled_time}>'

class GeneratedComment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    managed_account_id_to_post_from = db.Column(db.Integer, db.ForeignKey('managed_account.id'), nullable=False)
    target_platform = db.Column(db.String(50)) 
    target_post_id_on_platform = db.Column(db.String(255), nullable=False)
    target_post_content = db.Column(db.Text)
    target_post_author = db.Column(db.String(255))
    target_post_url = db.Column(db.String(1024))
    suggested_comment_text = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(50), default="pending_review")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    reviewed_at = db.Column(db.DateTime, nullable=True)
    posted_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    platform_comment_id = db.Column(db.String(255), nullable=True)

    posting_account = db.relationship('ManagedAccount', foreign_keys=[managed_account_id_to_post_from])

    def __repr__(self):
        return f'<GeneratedComment {self.id} for {self.target_post_id_on_platform}>'

class ActionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portal_user_id = db.Column(db.Integer, db.ForeignKey('portal_user.id'))
    managed_account_id = db.Column(db.Integer, db.ForeignKey('managed_account.id'), nullable=True)
    action_type = db.Column(db.String(100), nullable=False) 
    status = db.Column(db.String(50), nullable=False)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    
    portal_user = db.relationship('PortalUser', backref=db.backref('action_logs', lazy=True))

    def __repr__(self):
        return f'<ActionLog {self.action_type} ({self.status}) at {self.timestamp}>'

class CommentAutomationSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    portal_user_id = db.Column(db.Integer, db.ForeignKey('portal_user.id'), nullable=False)
    keywords = db.Column(db.Text)
    monitored_x_accounts = db.Column(db.Text)
    default_posting_managed_account_id = db.Column(db.Integer, db.ForeignKey('managed_account.id'), nullable=True)
    is_active = db.Column(db.Boolean, default=True)
    last_run_at = db.Column(db.DateTime, nullable=True)

    portal_user = db.relationship('PortalUser', backref=db.backref('comment_automation_settings', lazy=True, uselist=False))
    default_posting_account = db.relationship('ManagedAccount', foreign_keys=[default_posting_managed_account_id])
