from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField, DateTimeLocalField
from wtforms.validators import DataRequired, Email, EqualTo, Length, Optional, ValidationError
from app import db 
from app.models import PortalUser, ManagedAccount

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

def get_managed_account_choices(portal_user_id, platform='X', add_blank=False):
    accounts = db.session.scalars(
        db.select(ManagedAccount).filter_by(portal_user_id=portal_user_id, platform_name=platform, is_active=True).order_by(ManagedAccount.account_display_name)
    ).all()
    choices = [(acc.id, acc.account_display_name or acc.account_id_on_platform) for acc in accounts]
    if add_blank:
        choices.insert(0, ('', '--- Select Account ---'))
    return choices
