from app import create_app, db
from flask_migrate import Migrate

app = create_app('development')  # Explicitly specify config
with app.app_context():
    migrate = Migrate(app, db)
    print("Migrations initialized successfully!")
