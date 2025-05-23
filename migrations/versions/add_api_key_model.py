"""add api key model

Revision ID: add_api_key_model
Revises: add_keyphrase_fields
Create Date: 2025-05-23 07:17:00

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'add_api_key_model'
down_revision = 'add_keyphrase_fields'
branch_labels = None
depends_on = None


def upgrade():
    # Create the api_key table
    op.create_table(
        'api_key',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portal_user_id', sa.Integer(), nullable=False),
        sa.Column('openai_api_key', sa.String(length=256), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow),
        sa.ForeignKeyConstraint(['portal_user_id'], ['portal_user.id'], name='fk_api_key_portal_user'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('portal_user_id', name='uq_api_key_portal_user_id')
    )
    
    # Create an index on portal_user_id for faster lookups
    op.create_index('ix_api_key_portal_user_id', 'api_key', ['portal_user_id'], unique=True)


def downgrade():
    # Drop the api_key table
    op.drop_index('ix_api_key_portal_user_id', table_name='api_key')
    op.drop_table('api_key')
