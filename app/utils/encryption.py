from cryptography.fernet import Fernet
from flask import current_app
import logging

logger = logging.getLogger(__name__)

_fernet_instance = None

def get_fernet():
    global _fernet_instance
    if _fernet_instance is None:
        key = current_app.config.get('FERNET_KEY')
        if not key:
            logger.error("FERNET_KEY not set in application configuration. Cannot perform encryption/decryption.")
            raise ValueError("FERNET_KEY not set in application configuration.")
        try:
            _fernet_instance = Fernet(key.encode())
        except Exception as e:
            logger.error(f"Failed to initialize Fernet with key: {e}")
            raise ValueError(f"Invalid FERNET_KEY: {e}")
    return _fernet_instance

def encrypt_token(token_data: str) -> str | None:
    """Encrypts token data (e.g., OAuth access token)."""
    if not token_data:
        return None
    try:
        fernet = get_fernet()
        return fernet.encrypt(token_data.encode()).decode()
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        return None


def decrypt_token(encrypted_token_data: str) -> str | None:
    """Decrypts token data."""
    if not encrypted_token_data:
        return None
    try:
        fernet = get_fernet()
        return fernet.decrypt(encrypted_token_data.encode()).decode()
    except Exception as e:
        logger.error(f"Decryption failed. This might be due to an incorrect key or corrupted data: {e}")
        return None
