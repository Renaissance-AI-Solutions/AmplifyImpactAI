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
    print("\n=== APP INITIALIZATION STARTED ===")
    print(f"--- Current working directory: {os.getcwd()}")
    
    # Import services
    try:
        from app.services.scheduler_service import SchedulerService
        from app.services.embedding_service import initialize_embedding_service
        from app.services.knowledge_base_manager import initialize_kb_components, save_kb_components
        print("--- Successfully imported service modules")
    except ImportError as e:
        print(f"--- CRITICAL: Failed to import service modules: {e}")
        raise
    
    import atexit
    
    # Initialize scheduler service
    print("--- Initializing SchedulerService...")
    scheduler_service = SchedulerService()
    
    # Determine config
    if config_name is None:
        config_name = get_config_name()
    print(f"--- Using config: {config_name}")
    
    # Create Flask app
    print("--- Creating Flask application...")
    app = Flask(__name__, instance_relative_config=True)
    print("--- Loading configuration...")
    app.config.from_object(config_by_name[config_name])
    app.scheduler_service = scheduler_service  # Attach to app for global access
    print("--- Flask app created and configured")
    
    try:
        os.makedirs(app.instance_path, exist_ok=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except OSError:
        pass

    # Initialize extensions with debug info
    print("\n--- Initializing Flask extensions...")
    try:
        print("--- Initializing SQLAlchemy...")
        db.init_app(app)
        print("--- Initializing Migrate...")
        migrate.init_app(app, db)
        print("--- Initializing LoginManager...")
        login_manager.init_app(app)
        print("--- Initializing CSRF protection...")
        csrf.init_app(app)
        print("--- All Flask extensions initialized successfully")
    except Exception as e:
        print(f"--- ERROR initializing Flask extensions: {e}")
        raise
    
    # Initialize limiter with app
    limiter.init_app(app)

    # Initialize EmbeddingService first, then Knowledge Base components
    with app.app_context():
        # Initialize services with debug info
        print("\n--- Initializing application services...")
        
        # Import the embedding service module
        from app.services import embedding_service as es_module
        
        try:
            print("\n--- [1] Before initialize_embedding_service() ---")
            print(f"--- Global embedding_service_instance ID: {id(es_module.embedding_service_instance) if es_module.embedding_service_instance is not None else 'None'}")
            
            print("--- Initializing Embedding Service...")
            initialize_embedding_service(app)
            
            print("\n--- [2] After initialize_embedding_service() ---")
            print(f"--- Global embedding_service_instance ID: {id(es_module.embedding_service_instance) if es_module.embedding_service_instance is not None else 'None'}")
            if es_module.embedding_service_instance is not None:
                print(f"--- Client loaded: {hasattr(es_module.embedding_service_instance, 'client') and es_module.embedding_service_instance.client is not None}")
                print(f"--- Model: {getattr(es_module.embedding_service_instance, 'model_name', 'unknown')}")
            
            print("\n--- [3] Before initialize_kb_components() ---")
            print("--- Initializing Knowledge Base components...")
            
            with app.app_context():
                initialize_kb_components(app)
                
            print("\n--- [4] After initialize_kb_components() ---")
            print("--- Knowledge Base initialization complete")
                
            print("--- Registering cleanup handlers...")
            atexit.register(save_kb_components)
            print("--- Cleanup handlers registered")
            
            print("\n=== ALL SERVICES INITIALIZED SUCCESSFULLY ===\n")
        except Exception as e:
            print(f"\n!!! CRITICAL ERROR DURING SERVICE INITIALIZATION: {e}")
            print("!!! Application may not function correctly")
            print("!!! Full traceback:")
            import traceback
            traceback.print_exc()
            raise

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
