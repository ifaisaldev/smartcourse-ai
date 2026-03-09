import json
from flask import Blueprint, render_template, request, jsonify
from app import db
from app.models import SearchHistory, SavedCourse
from app.recommender import CourseRecommender

bp = Blueprint('main', __name__)

# Initialize recommender (lazy loading or global)
# Ideally, we initialize it once.
recommender = CourseRecommender()

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/recommend_page')
def recommend_page():
    return render_template('recommend.html')

@bp.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@bp.route('/about')
def about():
    return render_template('about.html')

@bp.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    query = data.get('query')
    model_type = data.get('model', 'tfidf') # 'tfidf' or 'neural'

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if model_type == 'neural':
        results = recommender.search_neural(query)
    else:
        results = recommender.search_tfidf(query)

    # Log history with results
    history = SearchHistory(
        search_text=query,
        model_used=model_type,
        results_json=json.dumps(results)
    )
    db.session.add(history)
    db.session.commit()

    return jsonify({'results': results, 'search_id': history.id})

@bp.route('/api/history', methods=['GET'])
def get_history():
    include_results = request.args.get('include_results', 'false').lower() == 'true'
    history = SearchHistory.query.order_by(SearchHistory.timestamp.desc()).limit(50).all()
    return jsonify([h.to_dict(include_results=include_results) for h in history])

@bp.route('/api/save', methods=['POST'])
def save_course():
    data = request.get_json()

    course_title = data.get('course_title')
    course_url = data.get('url')
    score = data.get('score')
    search_id = data.get('search_id')

    if not course_title:
        return jsonify({'error': 'Course title is required'}), 400

    # Check for duplicate: same title already saved under same search session
    existing = SavedCourse.query.filter_by(
        course_title=course_title,
        search_id=search_id
    ).first()
    if existing:
        return jsonify({'error': 'Course already saved', 'duplicate': True}), 409

    course = SavedCourse(
        course_title=course_title,
        course_url=course_url,
        score=score,
        search_id=search_id
    )
    db.session.add(course)
    db.session.commit()
    return jsonify({'success': True})

@bp.route('/api/saved', methods=['GET'])
def get_saved():
    saved = SavedCourse.query.order_by(SavedCourse.timestamp.desc()).all()

    # Group saved courses by search session
    sessions = {}
    ungrouped = []
    for s in saved:
        item = s.to_dict()
        if s.search_id:
            if s.search_id not in sessions:
                search = SearchHistory.query.get(s.search_id)
                sessions[s.search_id] = {
                    'search_id': s.search_id,
                    'query': search.search_text if search else 'Unknown',
                    'model_used': search.model_used if search else 'Unknown',
                    'timestamp': search.timestamp.isoformat() if search else '',
                    'courses': []
                }
            sessions[s.search_id]['courses'].append(item)
        else:
            ungrouped.append(item)

    return jsonify({
        'sessions': list(sessions.values()),
        'ungrouped': ungrouped
    })

@bp.route('/api/compare', methods=['POST'])
def compare_models():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    tfidf_results = recommender.search_tfidf(query)
    neural_results = recommender.search_neural(query)

    # Log comparison search with both result sets
    history = SearchHistory(
        search_text=query,
        model_used='comparison',
        results_json=json.dumps({'tfidf': tfidf_results, 'neural': neural_results})
    )
    db.session.add(history)
    db.session.commit()

    return jsonify({
        'tfidf': tfidf_results,
        'neural': neural_results,
        'search_id': history.id
    })

@bp.route('/api/courses', methods=['GET'])
def get_courses_meta():
    # Return metadata about the dataset
    if recommender.df is not None:
        return jsonify({
            'total_courses': len(recommender.df),
            'subjects': recommender.df['subject'].nunique() if 'subject' in recommender.df.columns else 0,
            'levels': recommender.df['level'].unique().tolist() if 'level' in recommender.df.columns else []
        })
    return jsonify({'error': 'Dataset not loaded'}), 500
