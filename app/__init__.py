import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import config_by_name, get_config_name

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
login_manager.login_view = 'auth_bp.login'
login_manager.login_message_category = 'info'


def create_app(config_name=None):
    from app.services.scheduler_service import SchedulerService
    from app.services.knowledge_base_manager import initialize_kb_components
    import atexit
    scheduler_service = SchedulerService()
    if config_name is None:
        config_name = get_config_name()
    
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_by_name[config_name])
    app.scheduler_service = scheduler_service  # Attach to app for global access
    
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError:
        pass

    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    csrf.init_app(app)
    
    # Initialize limiter with app
    limiter.init_app(app)

    # Initialize Knowledge Base components
    with app.app_context():
        initialize_kb_components(app)
    # Optionally register save_kb_components if available
    # from app.services.knowledge_base_manager import save_kb_components
    # atexit.register(save_kb_components)

    # Only initialize scheduler if enabled
    if app.config.get('SCHEDULER_API_ENABLED', True):
        scheduler_service.init_app(app)

    # Register blueprints
    from app.routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/auth')

    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    from app.routes.accounts import accounts_bp
    app.register_blueprint(accounts_bp, url_prefix='/accounts')

    from app.routes.content_generation import bp as content_generation_bp
    app.register_blueprint(content_generation_bp, url_prefix='/content-generation')

    from app.routes.knowledge_base import kb_bp
    app.register_blueprint(kb_bp, url_prefix='/knowledge-base')
    
    from app.routes.content_studio import content_studio_bp
    app.register_blueprint(content_studio_bp, url_prefix='/content-studio')

    from app.routes.engagement import engagement_bp
    app.register_blueprint(engagement_bp, url_prefix='/engagement')

    if not app.debug and not app.testing:
        if app.config['LOG_TO_STDOUT']:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            app.logger.addHandler(stream_handler)
        else:
            logs_dir = os.path.join(app.root_path, '..', 'logs')
            if not os.path.exists(logs_dir):
                os.mkdir(logs_dir)
            file_handler = RotatingFileHandler(os.path.join(logs_dir, 'amplify_impact.log'), maxBytes=10240, backupCount=10)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)

        app.logger.setLevel(logging.INFO)
        app.logger.info('Amplify Impact Pro startup')

    @app.context_processor
    def inject_current_time():
        from datetime import datetime, timezone
        return {'current_time_utc': datetime.now(timezone.utc)}
        
    @app.before_request
    def log_request_info():
        if request:
            if app.debug:
                app.logger.debug('Headers: %s', request.headers)
                app.logger.debug('Body: %s', request.get_data())
    
    return app
