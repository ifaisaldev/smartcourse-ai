import json
from app import db
from datetime import datetime

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    search_text = db.Column(db.String(500), nullable=False)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)
    model_used = db.Column(db.String(50), nullable=False) # 'tfidf', 'neural', or 'comparison'
    results_json = db.Column(db.Text, nullable=True)  # stores the results returned for this search

    saved_courses = db.relationship('SavedCourse', backref='search', lazy=True)

    def to_dict(self, include_results=False):
        data = {
            'id': self.id,
            'query': self.search_text,
            'timestamp': self.timestamp.isoformat(),
            'model_used': self.model_used
        }
        if include_results and self.results_json:
            data['results'] = json.loads(self.results_json)
        return data

class SavedCourse(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    course_title = db.Column(db.String(200), nullable=False)
    course_url = db.Column(db.String(500))
    score = db.Column(db.Float)
    search_id = db.Column(db.Integer, db.ForeignKey('search_history.id'), nullable=True)
    timestamp = db.Column(db.DateTime, index=True, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'course_title': self.course_title,
            'course_url': self.course_url,
            'score': self.score,
            'search_id': self.search_id,
            'timestamp': self.timestamp.isoformat()
        }
