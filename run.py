import os
from app import create_app, db
from app.models import PortalUser, ManagedAccount, KnowledgeDocument, ScheduledPost, ActionLog, CommentAutomationSetting, GeneratedComment, KnowledgeChunk
from dotenv import load_dotenv

load_dotenv()

app = create_app(os.getenv('FLASK_CONFIG') or 'default')

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
