# SmartCourse - AI Powered Course Recommendation System

SmartCourse is an intelligent recommendation engine designed to suggest educational courses based on user queries. It utilizes two machine learning approaches:
1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: For keyword-based matching.
2. **Neural Sentence Transformers**: For semantic understanding and context-aware recommendations.

## Features
- **Natural Language Search**: Describe what you want to learn using a multi-line text area for detailed preferences.
- **Dual Model Support**: Switch between Keyword (TF-IDF) and Semantic (Neural) search, or compare both side-by-side.
- **Top 10 Recommendations**: Each search returns the top 10 most relevant courses with relevance scores (0-100%) and visual progress bars.
- **Expandable Descriptions**: Course descriptions can be expanded/collapsed to view full details.
- **Save Courses**: Save preferred recommendations linked to their search session for easy organization.
- **Duplicate Prevention**: The system prevents saving the same course twice within a session.
- **User Dashboard**:
  - **Search History**: Complete history with timestamps, model used, and the actual results returned per search.
  - **Saved Courses**: Organized by search session, showing the original query and model used.
  - **Model Comparison**: Side-by-side comparison results stored and viewable in history.
- **About Page**: Technical implementation details and live dataset statistics.
- **Responsive UI**: Built with Bootstrap 5.

## Tech Stack
- **Backend**: Python 3.10+, Flask, SQLAlchemy, SQLite
- **ML/NLP**: spaCy, Sentence-Transformers (all-MiniLM-L6-v2), Scikit-learn, Joblib
- **Frontend**: HTML5, Bootstrap 5, JavaScript (Fetch API), Font Awesome
- **Data**: Pandas, NumPy

## API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/recommend` | Get recommendations (accepts `query` and `model`). Returns results and `search_id`. |
| POST | `/api/compare` | Compare TF-IDF and Neural results side-by-side for the same query. |
| GET | `/api/history` | Retrieve search history. Use `?include_results=true` to include stored results. |
| POST | `/api/save` | Save a course linked to a search session. Validates for duplicates. |
| GET | `/api/saved` | Get saved courses grouped by search session. |
| GET | `/api/courses` | Get dataset metadata (total courses, subjects, levels). |

## Setup & Installation

### Prerequisites
- Python 3.10+
- Virtual Environment (recommended)

### Installation
1. Clone the repository.
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

### Data Preparation & Training
Before running the app, you must prepare the data and train the models.

1. **Prepare Data**:
   ```bash
   python scripts/prepare_data.py
   ```
   This processes `data/raw/courses.csv` and creates `data/processed/courses.csv`.

2. **Train Models**:
   ```bash
   python scripts/train_models.py
   ```
   This generates the TF-IDF matrix and Neural Embeddings in the `models/` directory.

### Running the Application
Start the Flask server:
```bash
python run.py
```
Access the application at `http://127.0.0.1:5000`.

## Project Structure
- `app/`: Application source code (Routes, Models, Recommender).
  - `routes.py`: Flask API endpoints and page routes.
  - `models.py`: SQLAlchemy database models (SearchHistory, SavedCourse).
  - `recommender.py`: TF-IDF and Neural recommendation engine.
  - `utils.py`: Text preprocessing with spaCy (cleaning, lemmatization).
- `data/`: Dataset storage (Raw and Processed).
- `models/`: Saved ML models (.pkl files).
- `scripts/`: Utility scripts for data preparation and model training.
- `static/`: CSS and JavaScript files.
- `templates/`: HTML templates (Home, Recommend, Dashboard, About).

## Database Schema
- **SearchHistory**: Stores search queries, model used, timestamps, and the full results JSON.
- **SavedCourse**: Stores saved courses with a foreign key to SearchHistory for session-based grouping.
