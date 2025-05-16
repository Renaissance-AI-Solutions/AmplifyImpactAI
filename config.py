import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'instance', 'uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')

    # LLM Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

    # X/Twitter API Credentials
    X_CLIENT_ID = os.environ.get('X_CLIENT_ID')
    X_CLIENT_SECRET = os.environ.get('X_CLIENT_SECRET')
    X_CONSUMER_KEY = os.environ.get('X_CONSUMER_KEY')
    X_CONSUMER_SECRET = os.environ.get('X_CONSUMER_SECRET')
    X_CALLBACK_URL = os.environ.get('X_CALLBACK_URL') or 'http://localhost:5000/accounts/x/callback'

    # Fernet encryption key for tokens
    FERNET_KEY = os.environ.get('FERNET_KEY')

    # Scheduler settings
    SCHEDULER_API_ENABLED = True

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'instance', 'dev_app.db')

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL_PROD') or \
        'postgresql://user:password@host:port/dbname'

config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config_name():
    return os.getenv('FLASK_CONFIG') or 'default'
