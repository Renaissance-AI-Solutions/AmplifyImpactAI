from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField, DateTimeLocalField, IntegerField, TimeField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, ValidationError, NumberRange
from app import db 
from app.models import PortalUser, ManagedAccount, KnowledgeDocument

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=3, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email(), Length(max=120)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField(
        'Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = db.session.scalar(db.select(PortalUser).filter_by(username=username.data))
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = db.session.scalar(db.select(PortalUser).filter_by(email=email.data))
        if user is not None:
            raise ValidationError('Please use a different email address.')

class KnowledgeDocumentUploadForm(FlaskForm):
    document = FileField('Upload Document (PDF, TXT, DOCX)', validators=[
        FileRequired(),
        FileAllowed(['pdf', 'txt', 'docx'], 'Only PDF, TXT, and DOCX files are allowed!')
    ])
    submit = SubmitField('Upload')

class PostComposerForm(FlaskForm):
    target_account_id = SelectField('Post to Account', coerce=int, validators=[DataRequired()])
    content = TextAreaField('Post Content', validators=[DataRequired(), Length(max=280)])
    topic = StringField('Topic/Idea (for AI)', validators=[Optional(), Length(max=200)])
    tone = SelectField('Tone (for AI)', choices=[
        ('informative', 'Informative'), ('friendly', 'Friendly'), ('formal', 'Formal'), 
        ('urgent', 'Urgent'), ('inspirational', 'Inspirational'), ('humorous', 'Humorous')
    ], validators=[Optional()])
    style = SelectField('Style (for AI)', choices=[
        ('concise', 'Concise'), ('detailed', 'Detailed'), ('question', 'Question'), 
        ('story', 'Story-telling')
    ], validators=[Optional()])
    
    post_now = SubmitField('Post Now')
    schedule_post_action = SubmitField('Schedule Post')
    save_draft_action = SubmitField('Save Draft')
    generate_with_ai = SubmitField('Generate with AI')
    generated_content_display = TextAreaField('AI Suggestion', render_kw={'readonly': True, 'rows': 5})

class SchedulePostForm(FlaskForm):
    target_account_id_schedule = SelectField('Post to Account', coerce=int, validators=[DataRequired()])
    content_schedule = TextAreaField('Post Content', validators=[DataRequired(), Length(max=280)])
    scheduled_time = DateTimeLocalField('Schedule Time (UTC)', format='%Y-%m-%dT%H:%M', validators=[DataRequired()])
    draft_id = StringField('Draft ID', render_kw={'style': 'display:none;'})
    submit_schedule = SubmitField('Confirm Schedule')

class CommentSettingsForm(FlaskForm):
    keywords = TextAreaField('Keywords/Hashtags (comma-separated, for X post discovery)', validators=[Optional(), Length(max=1000)])
    monitored_x_accounts = TextAreaField('X Accounts to Monitor (comma-separated screen names, e.g., account1,account2)', validators=[Optional(), Length(max=1000)])
    default_posting_account_id = SelectField('Default X Account for Commenting', coerce=int, validators=[Optional()])
    is_active = BooleanField('Enable Automated Comment Discovery', default=True)
    submit = SubmitField('Save Comment Settings')

class ApiKeyForm(FlaskForm):
    openai_api_key = StringField('OpenAI API Key', validators=[Optional(), Length(min=20, max=100)])
    submit = SubmitField('Save API Key(s)')

class EditCommentForm(FlaskForm):
    comment_text = TextAreaField('Edit Comment', validators=[DataRequired(), Length(max=280)])
    submit_edit = SubmitField('Save and Approve')
    approve_direct = SubmitField('Approve Original')

def get_managed_account_choices(portal_user_id, platform=None, add_blank=False):
    # Build the query based on provided platform filter
    query = db.select(ManagedAccount).filter_by(portal_user_id=portal_user_id, is_active=True)
    
    # If a specific platform is requested, filter by it
    if platform:
        query = query.filter_by(platform_name=platform)
        
    # Get accounts and order by platform name and display name
    accounts = db.session.scalars(
        query.order_by(ManagedAccount.platform_name, ManagedAccount.account_display_name)
    ).all()
    
    # Create choices with platform indicator in the display text
    choices = [(acc.id, f"{acc.account_display_name or acc.account_id_on_platform} ({acc.platform_name})") for acc in accounts]
    
    if add_blank:
        choices.insert(0, ('', '--- Select Account ---'))
        
    return choices


def get_document_choices(portal_user_id, add_blank=True):
    """Get a list of knowledge documents for the user."""
    documents = db.session.scalars(
        db.select(KnowledgeDocument)
        .filter_by(portal_user_id=portal_user_id)
        .order_by(KnowledgeDocument.uploaded_at.desc())
    ).all()
    
    choices = [(str(doc.id), f"{doc.filename} (uploaded {doc.uploaded_at.strftime('%Y-%m-%d')})") 
              for doc in documents]
    
    if add_blank:
        choices.insert(0, ('', '--- Select a Document ---'))
    
    return choices


class RecurringPostScheduleForm(FlaskForm):
    """Form for creating and editing recurring post schedules."""
    name = StringField('Schedule Name', validators=[DataRequired(), Length(max=255)])
    target_account_id = SelectField('Post to Account', coerce=int, validators=[DataRequired()])
    content_template = TextAreaField('Post Content Template', validators=[DataRequired(), Length(max=280)])
    
    frequency = SelectField('Frequency', choices=[
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly')
    ], validators=[DataRequired()])
    
    time_of_day = TimeField('Time of Day (UTC)', format='%H:%M', validators=[DataRequired()])
    day_of_week = SelectField('Day of Week', choices=[
        (0, 'Monday'),
        (1, 'Tuesday'),
        (2, 'Wednesday'),
        (3, 'Thursday'),
        (4, 'Friday'),
        (5, 'Saturday'),
        (6, 'Sunday')
    ], coerce=int, validators=[Optional()])
    day_of_month = SelectField('Day of Month', 
                             choices=[(i, str(i)) for i in range(1, 32)],
                             coerce=int, validators=[Optional()])
    
    is_active = BooleanField('Active', default=True)
    submit = SubmitField('Save Schedule')


class ContentGenerationForm(FlaskForm):
    """Form for generating content from knowledge documents."""
    document_id = SelectField('Knowledge Document', 
                           coerce=lambda x: int(x) if x else None,
                           validators=[DataRequired(message='Please select a document')])
    platform = SelectField('Platform', choices=[
        ('twitter', 'Twitter/X'),
        ('linkedin', 'LinkedIn'),
        ('facebook', 'Facebook'),
        ('instagram', 'Instagram')
    ], validators=[DataRequired()])
    tone = SelectField('Tone', choices=[
        ('informative', 'Informative'),
        ('friendly', 'Friendly'),
        ('formal', 'Formal'),
        ('urgent', 'Urgent'),
        ('inspirational', 'Inspirational'),
        ('humorous', 'Humorous')
    ], validators=[DataRequired()])
    style = SelectField('Style', choices=[
        ('concise', 'Concise'),
        ('detailed', 'Detailed'),
        ('question', 'Question'),
        ('story', 'Story-telling')
    ], validators=[DataRequired()])
    topic = StringField('Specific Topic/Focus (optional)', validators=[Optional(), Length(max=200)])
    max_length = IntegerField('Maximum Length (characters)', validators=[
        DataRequired(),
        NumberRange(min=10, max=3000, message='Length must be between 10 and 3000 characters')
    ], default=280)
    include_hashtags = BooleanField('Include Hashtags', default=True)
    include_emoji = BooleanField('Include Emoji', default=True)
    generate_button = SubmitField('Generate Content')
    copy_button = SubmitField('Copy to Clipboard')
    save_button = SubmitField('Save as Draft')
