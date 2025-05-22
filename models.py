# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

# Initialize SQLAlchemy extension instance
# This 'db' object will be imported by your main app and initialized with db.init_app(app)
db = SQLAlchemy()

def generate_uuid(): # Renamed to avoid potential conflict, though 'uuid' is fine here.
    return str(uuid.uuid4())

class User(db.Model):
    # ... your User model definition using 'db' ...
    __tablename__ = 'users'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    username = db.Column(db.String(80), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    quiz_attempts = db.relationship('QuizAttempt', backref='user', lazy=True, cascade="all, delete-orphan")
    topic_performances = db.relationship('TopicPerformance', backref='user', lazy=True, cascade="all, delete-orphan")
    def __repr__(self): return f'<User {self.username}>'

class TopicPerformance(db.Model):
    __tablename__ = 'topic_performances'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    topic_name = db.Column(db.String(255), nullable=False)
    attempts = db.Column(db.Integer, default=0)
    total_score_points = db.Column(db.Integer, default=0)
    total_questions_attempted = db.Column(db.Integer, default=0)
    current_difficulty = db.Column(db.String(50), default='medium')
    __table_args__ = (db.UniqueConstraint('user_id', 'topic_name', name='_user_topic_uc'),)
    # Note: In your __repr__ you had self.user.username, which might cause issues if user is not eagerly loaded
    # It's safer to use self.user_id or ensure you query with user loaded if needed in repr.
    # For simplicity, changed to user_id for now.
    def __repr__(self): return f'<TopicPerformance UserID:{self.user_id} Topic:{self.topic_name}>'


class SubtopicMastery(db.Model):
    __tablename__ = 'subtopic_mastery'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    topic_performance_id = db.Column(db.String(36), db.ForeignKey('topic_performances.id'), nullable=False)
    subtopic_name = db.Column(db.String(255), nullable=False)
    correct_answers = db.Column(db.Integer, default=0)
    times_seen = db.Column(db.Integer, default=0)
    topic_performance = db.relationship('TopicPerformance', backref=db.backref('subtopic_masteries', lazy='dynamic', cascade="all, delete-orphan")) # lazy='dynamic' is good
    __table_args__ = (db.UniqueConstraint('topic_performance_id', 'subtopic_name', name='_topic_subtopic_uc'),)


class QuizAttempt(db.Model):
    __tablename__ = 'quiz_attempts'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    topic_name = db.Column(db.String(255), nullable=False)
    difficulty = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    score = db.Column(db.Integer, nullable=False)
    total_questions = db.Column(db.Integer, nullable=False)
    answered_questions = db.relationship('AnsweredQuestion', backref='quiz_attempt', lazy=True, cascade="all, delete-orphan")
    # Similar to TopicPerformance, using self.user_id in repr to avoid loading issues
    def __repr__(self): return f'<QuizAttempt {self.id} UserID:{self.user_id} on {self.topic_name}>'

class AnsweredQuestion(db.Model):
    __tablename__ = 'answered_questions'
    id = db.Column(db.String(36), primary_key=True, default=generate_uuid)
    quiz_attempt_id = db.Column(db.String(36), db.ForeignKey('quiz_attempts.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    options = db.Column(db.JSON, nullable=False)
    selected_answer = db.Column(db.Text, nullable=True) # Allow null if no answer selected
    correct_answer = db.Column(db.Text, nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False)
    subtopic = db.Column(db.String(255), nullable=True) # Allow null
    def __repr__(self): return f'<AnsweredQuestion for Quiz {self.quiz_attempt_id} - Correct: {self.is_correct}>'