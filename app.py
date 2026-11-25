"""
app.py — Flask application for SmartStudy Generator Agent
"""

import os
import json
import uuid
from pathlib import Path
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for)
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import Config
from agent import SmartStudyAgent
from tools import (
    extract_text_from_pdf,
    extract_text_from_image,
    extract_text_from_audio,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)
app.config["UPLOAD_FOLDER"] = Config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = Config.MAX_CONTENT_LENGTH
CORS(app)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.DATA_FOLDER, exist_ok=True)

# Per-session agents (simple in-memory store; use Redis in production)
from typing import Dict as _Dict
_agents: _Dict[str, SmartStudyAgent] = {}


def get_agent() -> SmartStudyAgent:
    sid = session.get("sid")
    if not sid or sid not in _agents:
        sid = str(uuid.uuid4())
        session["sid"] = sid
        try:
            _agents[sid] = SmartStudyAgent()
        except ValueError as e:
            # Return a stub that always returns the config error
            class _ErrAgent:
                def __getattr__(self, _):
                    def _err(*a, **kw):
                        return {"response": str(e), "type": "error", "tool_calls": []}
                    return _err
                def reset_conversation(self): pass
            _agents[sid] = _ErrAgent()
    return _agents[sid]


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def _load_progress(sid: str) -> dict:
    path = Path(Config.DATA_FOLDER) / f"{sid}_progress.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "quizzes_taken": 0,
        "flashcards_reviewed": 0,
        "summaries_created": 0,
        "topics_studied": [],
        "quiz_scores": {},
        "study_streak": 0,
        "total_study_time": 0,
        "weak_areas": [],
        "strong_areas": [],
    }


def _save_progress(sid: str, data: dict):
    path = Path(Config.DATA_FOLDER) / f"{sid}_progress.json"
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Routes — Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    session.setdefault("sid", str(uuid.uuid4()))
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    sid = session.get("sid", str(uuid.uuid4()))
    session["sid"] = sid
    progress = _load_progress(sid)
    return render_template("dashboard.html", progress=progress)


@app.route("/study")
def study():
    return render_template("study.html")


@app.route("/quiz")
def quiz_page():
    return render_template("quiz.html")


# ---------------------------------------------------------------------------
# API — Chat
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    context = data.get("context", {})

    if not message:
        return jsonify({"error": "Message is required"}), 400

    agent = get_agent()
    result = agent.chat(message, context)
    return jsonify(result)


@app.route("/api/chat/reset", methods=["POST"])
def api_chat_reset():
    get_agent().reset_conversation()
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# API — Document processing
# ---------------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4()}_{filename}"
    filepath = os.path.join(Config.UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    ext = filename.rsplit(".", 1)[1].lower()
    if ext == "pdf":
        text = extract_text_from_pdf(filepath)
    elif ext in {"png", "jpg", "jpeg"}:
        text = extract_text_from_image(filepath)
    elif ext in {"wav", "mp3", "m4a"}:
        text = extract_text_from_audio(filepath)
    else:  # txt
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")

    if not text or text.startswith("Error"):
        return jsonify({"error": text or "Could not extract text"}), 422

    return jsonify({"text": text, "filename": filename, "characters": len(text)})


@app.route("/api/process", methods=["POST"])
def api_process():
    data   = request.get_json(force=True)
    text   = data.get("text", "").strip()
    action = data.get("action", "summarize")
    options = data.get("options", {})

    if not text:
        return jsonify({"error": "Text is required"}), 400

    agent  = get_agent()
    result = agent.process_document(text, action, options)

    # Update progress
    sid = session.get("sid", "anon")
    progress = _load_progress(sid)
    if action == "summarize":
        progress["summaries_created"] += 1
    elif action == "flashcards":
        progress["flashcards_reviewed"] += result.get("data", {}).get("count", 0)
    elif action == "quiz":
        progress["quizzes_taken"] += 1
    subject = options.get("subject", "")
    if subject and subject not in progress["topics_studied"]:
        progress["topics_studied"].append(subject)
    _save_progress(sid, progress)

    return jsonify(result)


# ---------------------------------------------------------------------------
# API — Study Plan
# ---------------------------------------------------------------------------

@app.route("/api/study-plan", methods=["POST"])
def api_study_plan():
    data        = request.get_json(force=True)
    topics      = data.get("topics", [])
    days        = int(data.get("days", 7))
    hours       = float(data.get("hours_per_day", 2.0))
    weak_areas  = data.get("weak_areas", [])

    if not topics:
        return jsonify({"error": "Topics list is required"}), 400

    agent  = get_agent()
    result = agent.generate_study_plan(topics, days, hours, weak_areas)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API — Exam predictions
# ---------------------------------------------------------------------------

@app.route("/api/predict-topics", methods=["POST"])
def api_predict_topics():
    data      = request.get_json(force=True)
    text      = data.get("text", "").strip()
    subject   = data.get("subject", "")
    exam_type = data.get("exam_type", "final")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    agent  = get_agent()
    result = agent.get_exam_predictions(text, subject, exam_type)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API — Quiz submission & performance analysis
# ---------------------------------------------------------------------------

@app.route("/api/submit-quiz", methods=["POST"])
def api_submit_quiz():
    data    = request.get_json(force=True)
    answers = data.get("answers", {})   # {question_id: selected_answer}
    correct = data.get("correct", {})   # {question_id: correct_answer}
    subject = data.get("subject", "")

    if not answers or not correct:
        return jsonify({"error": "Answers and correct answers required"}), 400

    # Calculate score per topic
    topic_scores: dict[str, list[int]] = {}
    total_correct = 0
    for qid, ans in answers.items():
        is_correct = int(ans == correct.get(qid))
        total_correct += is_correct
        topic = data.get("topics", {}).get(qid, subject or "General")
        topic_scores.setdefault(topic, []).append(is_correct * 100)

    quiz_results = {t: int(sum(v) / len(v)) for t, v in topic_scores.items()}
    overall = int(total_correct / len(answers) * 100) if answers else 0

    # Persist
    sid = session.get("sid", "anon")
    progress = _load_progress(sid)
    progress["quizzes_taken"] += 1
    for topic, score in quiz_results.items():
        progress["quiz_scores"][topic] = score
    _save_progress(sid, progress)

    agent  = get_agent()
    result = agent.analyze_performance(quiz_results, subject)
    result["overall_score"] = overall
    result["total_correct"] = total_correct
    result["total_questions"] = len(answers)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API — Progress / Dashboard data
# ---------------------------------------------------------------------------

@app.route("/api/progress", methods=["GET"])
def api_progress():
    sid = session.get("sid", "anon")
    return jsonify(_load_progress(sid))


@app.route("/api/progress/update", methods=["POST"])
def api_progress_update():
    sid = session.get("sid", "anon")
    data = request.get_json(force=True)
    progress = _load_progress(sid)
    progress.update(data)
    _save_progress(sid, progress)
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# API — Concept explanation
# ---------------------------------------------------------------------------

@app.route("/api/explain", methods=["POST"])
def api_explain():
    data    = request.get_json(force=True)
    concept = data.get("concept", "").strip()
    level   = data.get("level", "intermediate")
    subject = data.get("subject", "")

    if not concept:
        return jsonify({"error": "Concept is required"}), 400

    from tools import explain_concept
    result = explain_concept(concept, level, subject)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("🎓 SmartStudy AI Agent starting on http://127.0.0.1:5000")
    app.run(debug=Config.DEBUG, port=5000, use_reloader=False)
