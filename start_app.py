from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from flask_bootstrap import Bootstrap
from config import config_by_name

app = Flask(__name__)
app.config.from_object(config_by_name['development'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)
login_manager = LoginManager(app)
login_manager.login_view = 'auth_bp.login'
login_manager.login_message_category = 'info'
bootstrap = Bootstrap(app)

# Import models
from app.models import PortalUser, ManagedAccount, KnowledgeDocument, ScheduledPost, ActionLog, CommentAutomationSetting, GeneratedComment, KnowledgeChunk

# Import blueprints
from app.routes.main import main_bp
from app.routes.auth import auth_bp
from app.routes.engagement import engagement_bp
from app.routes.content_studio import content_studio_bp

# Register blueprints
app.register_blueprint(main_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(engagement_bp)
app.register_blueprint(content_studio_bp)

@app.shell_context_processor
def make_shell_context():
    return {
        'db': db,
        'PortalUser': PortalUser,
        'ManagedAccount': ManagedAccount,
        'KnowledgeDocument': KnowledgeDocument,
        'KnowledgeChunk': KnowledgeChunk,
        'ScheduledPost': ScheduledPost,
        'GeneratedComment': GeneratedComment,
        'ActionLog': ActionLog,
        'CommentAutomationSetting': CommentAutomationSetting
    }

if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True, host='0.0.0.0', port=5000)
