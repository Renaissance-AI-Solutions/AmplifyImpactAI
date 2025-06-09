# AmplifyImpactAI

## Security Notice

This application handles sensitive information including API keys and user credentials. Please follow these security best practices:

1. **Never commit sensitive information** to version control
2. Use the `.env` file for local development and ensure it's in your `.gitignore`
3. Generate strong, unique secrets for production
4. Rotate credentials immediately if they are compromised
5. Use environment variables or a secure secret manager in production

## Setup Instructions

### Prerequisites

- Python 3.8+
- pip
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AmplifyImpactAI.git
   cd AmplifyImpactAI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Generate secure values and update .env
   python scripts/setup_env.py
   ```
   
   Then edit the `.env` file to add your actual API keys and configuration.

5. Initialize the database:
   ```bash
   flask db upgrade
   ```

6. Run the application:
   ```bash
   py run.py
   ```

## Environment Variables

Required environment variables are listed in `.env.example`. The application will not start without these variables set.

## Security Best Practices

1. **Never commit sensitive data** to version control
2. Use strong, randomly generated secrets
3. Use HTTPS in production
4. Keep dependencies up to date
5. Regularly rotate credentials
6. Use a proper secret manager in production (AWS Secrets Manager, HashiCorp Vault, etc.)

## Reporting Security Issues

If you discover a security vulnerability, please report it to your security team immediately. Do not create a public GitHub issue.
