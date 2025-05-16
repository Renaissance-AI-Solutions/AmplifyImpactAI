#!/usr/bin/env python3
"""
Helper script to generate a secure .env file.
Run this script to generate a new .env file with secure random values.
"""
import os
import secrets
import string
from pathlib import Path

def generate_fernet_key():
    """Generate a URL-safe base64-encoded 32-byte key for Fernet encryption."""
    return secrets.token_urlsafe(32)

def generate_secret_key():
    """Generate a secure secret key for Flask."""
    return secrets.token_hex(32)

def main():
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    env_file = project_root / '.env'
    
    # Check if .env already exists
    if env_file.exists():
        print("Warning: .env file already exists. Do you want to overwrite it? (y/N): ")
        if input().lower() != 'y':
            print("Aborted.")
            return
    
    # Generate secure values
    secret_key = generate_secret_key()
    fernet_key = generate_fernet_key()
    
    # Read the example file
    example_file = project_root / '.env.example'
    if not example_file.exists():
        print("Error: .env.example file not found.")
        return
    
    # Replace placeholders with generated values
    env_content = example_file.read_text()
    env_content = env_content.replace('generate_a_strong_secret_key_here', secret_key)
    env_content = env_content.replace('generate_a_strong_fernet_key_here', fernet_key)
    
    # Write the new .env file
    with env_file.open('w') as f:
        f.write(env_content)
    
    print(f"Created new .env file at {env_file}")
    print("IMPORTANT: Keep this file secure and do not commit it to version control!")

if __name__ == '__main__':
    main()
