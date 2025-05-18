import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    SECRET_KEY = os.environ['SECRET_KEY']  # Must be set in environment
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'instance', 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    UPLOAD_FOLDER = os.path.join(basedir, 'instance', 'uploads')
    ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT')  # Optional; set to 'True' to log to stdout

    # LLM Configuration
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY') 

    # Embedding Service Configuration
    KB_EMBEDDING_MODEL_NAME = os.environ.get("KB_EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")
    KB_EMBEDDING_NORMALIZE = os.environ.get("KB_EMBEDDING_NORMALIZE", "True").lower() == "true"
    KB_EMBEDDING_PASSAGE_PREFIX = os.environ.get("KB_EMBEDDING_PASSAGE_PREFIX", "") 
    KB_EMBEDDING_QUERY_PREFIX = os.environ.get("KB_EMBEDDING_QUERY_PREFIX", "Represent this sentence for searching relevant passages: ") 
    KB_EMBEDDING_ENABLE_CACHE = os.environ.get("KB_EMBEDDING_ENABLE_CACHE", "False").lower() == "true"

    # Knowledge Base & FAISS Configuration
    KB_CHUNK_SIZE_TOKENS = int(os.environ.get("KB_CHUNK_SIZE_TOKENS", "256"))
    KB_CHUNK_OVERLAP_TOKENS = int(os.environ.get("KB_CHUNK_OVERLAP_TOKENS", "32"))
    FAISS_INDEX_FILENAME_TPL = os.environ.get("FAISS_INDEX_FILENAME_TPL", "kb_faiss_{model}.index")
    FAISS_MAP_FILENAME_TPL = os.environ.get("FAISS_MAP_FILENAME_TPL", "kb_faiss_map_{model}.pkl")

    # LLM Generation Provider API Keys (add as needed)
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

    # Default LLM settings for generation
    DEFAULT_GENERATION_PROVIDER = 'gemini' if GEMINI_API_KEY else os.getenv('DEFAULT_GENERATION_PROVIDER', 'openai') 
    GENERATION_FALLBACK_ORDER = ['claude', 'deepseek', 'openai'] 
    if DEFAULT_GENERATION_PROVIDER in GENERATION_FALLBACK_ORDER:
        GENERATION_FALLBACK_ORDER.remove(DEFAULT_GENERATION_PROVIDER)
    
    OPENAI_CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")
    GEMINI_CHAT_MODEL = os.environ.get("GEMINI_CHAT_MODEL", "gemini-1.5-flash") 
    CLAUDE_CHAT_MODEL = os.environ.get("CLAUDE_CHAT_MODEL", "claude-3-haiku-20240307")
    DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    GENERATION_MAX_TOKENS = int(os.environ.get("GENERATION_MAX_TOKENS", 2000)) 
    DEFAULT_LLM_STRATEGY = os.environ.get("DEFAULT_LLM_STRATEGY", "primary") 

    # X/Twitter API Credentials - must be set in environment
    X_CLIENT_ID = os.environ.get('X_CLIENT_ID')
    X_CLIENT_SECRET = os.environ.get('X_CLIENT_SECRET')
    X_CONSUMER_KEY = os.environ.get('X_CONSUMER_KEY')
    X_CONSUMER_SECRET = os.environ.get('X_CONSUMER_SECRET')
    X_CALLBACK_URL = os.environ.get('X_CALLBACK_URL')  

    # Fernet encryption key for tokens - must be set in environment
    FERNET_KEY = os.environ.get('FERNET_KEY')

    # Scheduler settings
    SCHEDULER_API_ENABLED = True

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'instance', 'dev_app.db')

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL_PROD') or \
        'postgresql://{user}:{password}@{host}:{port}/{dbname}'.format(
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', ''),
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5432'),
            dbname=os.environ.get('DB_NAME', 'amplify_impact')
        )

config_by_name = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config_name():
    return os.getenv('FLASK_CONFIG') or 'default'
