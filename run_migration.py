from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, upgrade
from config import config_by_name

app = Flask(__name__)
app.config.from_object(config_by_name['development'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Create the migration script
with app.app_context():
    upgrade()

print("Migration created successfully!")
