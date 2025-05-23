"""add recurring post schedule

Revision ID: add_recurring_post_schedule
Revises: add_api_key_model
Create Date: 2025-05-23 07:30:00

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'add_recurring_post_schedule'
down_revision = 'add_api_key_model'
branch_labels = None
depends_on = None


def upgrade():
    # Create the recurring_post_schedule table
    op.create_table(
        'recurring_post_schedule',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('portal_user_id', sa.Integer(), nullable=False),
        sa.Column('managed_account_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('content_template', sa.Text(), nullable=False),
        sa.Column('media_urls', sa.String(length=1024), nullable=True),
        sa.Column('frequency', sa.String(length=50), nullable=False),
        sa.Column('time_of_day', sa.Time(), nullable=False),
        sa.Column('day_of_week', sa.Integer(), nullable=True),
        sa.Column('day_of_month', sa.Integer(), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(), nullable=True, default=datetime.utcnow),
        sa.Column('updated_at', sa.DateTime(), nullable=True, default=datetime.utcnow, onupdate=datetime.utcnow),
        sa.Column('last_run_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['portal_user_id'], ['portal_user.id'], name='fk_recurring_post_schedule_portal_user'),
        sa.ForeignKeyConstraint(['managed_account_id'], ['managed_account.id'], name='fk_recurring_post_schedule_managed_account'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes for faster lookups
    op.create_index('ix_recurring_post_schedule_portal_user_id', 'recurring_post_schedule', ['portal_user_id'], unique=False)
    op.create_index('ix_recurring_post_schedule_managed_account_id', 'recurring_post_schedule', ['managed_account_id'], unique=False)
    
    # Add recurring post fields to scheduled_post table
    with op.batch_alter_table('scheduled_post') as batch_op:
        batch_op.add_column(sa.Column('is_from_recurring_schedule', sa.Boolean(), nullable=True, default=False))
        batch_op.add_column(sa.Column('recurring_schedule_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_scheduled_post_recurring_schedule', 'recurring_post_schedule', ['recurring_schedule_id'], ['id'])


def downgrade():
    # Remove recurring post fields from scheduled_post table
    with op.batch_alter_table('scheduled_post') as batch_op:
        batch_op.drop_constraint('fk_scheduled_post_recurring_schedule', type_='foreignkey')
        batch_op.drop_column('recurring_schedule_id')
        batch_op.drop_column('is_from_recurring_schedule')
    
    # Drop indexes
    op.drop_index('ix_recurring_post_schedule_managed_account_id', table_name='recurring_post_schedule')
    op.drop_index('ix_recurring_post_schedule_portal_user_id', table_name='recurring_post_schedule')
    
    # Drop the recurring_post_schedule table
    op.drop_table('recurring_post_schedule')
