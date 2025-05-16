from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from config import config_by_name

app = Flask(__name__)
app.config.from_object(config_by_name['development'])

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Initialize the migrations
migrate.init_app(app, db)

# Run the migrations
with app.app_context():
    # Create the migration
    from flask_migrate import upgrade
    upgrade()

print("Migrations created successfully!")
