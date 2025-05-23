"""add keyphrase fields

Revision ID: add_keyphrase_fields
Revises: bab2afd5c5ff
Create Date: 2025-05-22 21:17:00

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_keyphrase_fields'
down_revision = 'bab2afd5c5ff'
branch_labels = None
depends_on = None


def upgrade():
    # It appears these columns were already added to the database outside of Alembic
    # This empty upgrade will just mark this migration as complete in Alembic's version table
    # without trying to add the columns again
    pass

def downgrade():
    with op.batch_alter_table('knowledge_chunk') as batch_op:
        batch_op.drop_constraint('fk_knowledge_chunk_portal_user', type_='foreignkey')
        batch_op.drop_column('embedding_vector')
        batch_op.drop_column('embedding_model_name')
        batch_op.drop_column('portal_user_id')
    with op.batch_alter_table('knowledge_document') as batch_op:
        batch_op.drop_column('chunk_count')
        batch_op.drop_column('embedding_model_name')
        batch_op.drop_column('summary')
        batch_op.drop_column('keyphrases')

