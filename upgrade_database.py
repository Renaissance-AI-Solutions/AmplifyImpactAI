from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate, upgrade
from config import config_by_name

app = Flask(__name__)
app.config.from_object(config_by_name['development'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize the migrations
migrate.init_app(app, db)

# Run the upgrade
with app.app_context():
    upgrade()

print("Database upgraded successfully!")
