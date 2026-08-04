from flask import Flask, render_template, request, send_file
from utils.parser import extract_text_from_pdf
from utils.processor import compute_hybrid_score, compute_similarity
from utils.pdf_generator import generate_pdf_summary
from utils.skill_extractor import (
    load_skills, compare_skills, extract_skills,
    generate_suggestions, infer_job_role
)
from utils.history_tracker import save_history, load_user_history
import os
import pandas as pd
import json
from werkzeug.utils import secure_filename
from extensions import db, login_manager
from flask_login import current_user, login_required
from auth import auth
import zipfile
import io

from auth.models import User, MatchHistory

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-only-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///users.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"

uploads = "uploads"
summaries = "summaries"
matched_results = []


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


app.register_blueprint(auth)

os.makedirs(uploads, exist_ok=True)
os.makedirs(summaries, exist_ok=True)

with app.app_context():
    db.create_all()
    from utils.history_tracker import _migrate_csv_history
    _migrate_csv_history()


def load_role_map(filepath="utils/role_skills.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jd_categories(filepath="utils/jd_categories.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def tag_jd(jd_text, category_map=None):
    category_map = category_map or load_jd_categories()
    jd_text_lower = jd_text.lower()
    scores = {cat: sum(kw in jd_text_lower for kw in kws) for cat, kws in category_map.items()}
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "Other"


def save_upload(file_storage, folder=uploads):
    filename = secure_filename(file_storage.filename)
    path = os.path.join(folder, filename)
    file_storage.save(path)
    return filename, path


def score_pair(resume_filename, jd_filename, resume_text, jd_text, skills_dict, role_skill_map):
    """Score one resume/JD pair and return a display row."""
    matched, missing, extra = compare_skills(jd_text, resume_text, skills_dict)
    resume_skills = extract_skills(resume_text, skills_dict)
    jd_skills = extract_skills(jd_text, skills_dict)
    score, _, _, _ = compute_hybrid_score(jd_text, resume_text, jd_skills, resume_skills)
    suggestions = generate_suggestions(missing)
    role = infer_job_role(resume_skills, role_skill_map)

    pdf_path = generate_pdf_summary(
        resume_filename, jd_filename, score, matched, missing, extra, suggestions, role
    )
    pdf_name = os.path.basename(pdf_path)

    return float(score), matched, missing, extra, suggestions, role, pdf_name


def process_recruiter_mode(jd_file, resume_files, skills_dict, role_skill_map):
    jd_filename, jd_path = save_upload(jd_file)
    jd_text = extract_text_from_pdf(jd_path)
    jd_tag = tag_jd(jd_text)
    jd_skills = extract_skills(jd_text, skills_dict)

    resumes_data = []
    resume_entries = []

    for resume_file in resume_files:
        resume_filename, resume_path = save_upload(resume_file)
        resume_text = extract_text_from_pdf(resume_path)
        resume_entries.append((resume_filename, resume_text))

    resume_skills_map = {
        name: extract_skills(text, skills_dict) for name, text in resume_entries
    }
    ranked = compute_similarity(jd_text, resume_entries, jd_skills, resume_skills_map)
    rank_lookup = {name: score for name, score in ranked}

    for resume_filename, resume_text in resume_entries:
        score = rank_lookup[resume_filename]
        matched, missing, extra = compare_skills(jd_text, resume_text, skills_dict)
        resume_skills = resume_skills_map[resume_filename]
        suggestions = generate_suggestions(missing)
        role = infer_job_role(resume_skills, role_skill_map)

        pdf_path = generate_pdf_summary(
            resume_filename, jd_filename, score, matched, missing, extra, suggestions, role
        )
        pdf_name = os.path.basename(pdf_path)

        row = (resume_filename, score, matched, missing, extra, suggestions, role, pdf_name, jd_tag)
        resumes_data.append(row)

        save_history(
            current_user.id, "recruiter", resume_filename, jd_filename,
            score, matched, missing, extra, role, pdf_name,
        )

    resumes_data.sort(key=lambda row: row[1], reverse=True)
    return resumes_data


def process_candidate_mode(resume_file, jd_files, skills_dict, role_skill_map):
    resume_filename, resume_path = save_upload(resume_file)
    resume_text = extract_text_from_pdf(resume_path)
    jd_tag = tag_jd(resume_text)

    resumes_data = []

    for jd_file in jd_files:
        jd_filename, jd_path = save_upload(jd_file)
        jd_text = extract_text_from_pdf(jd_path)

        score, matched, missing, extra, suggestions, role, pdf_name = score_pair(
            resume_filename, jd_filename, resume_text, jd_text, skills_dict, role_skill_map
        )

        row = (jd_filename, score, matched, missing, extra, suggestions, role, pdf_name, jd_tag)
        resumes_data.append(row)

        save_history(
            current_user.id, "candidate", resume_filename, jd_filename,
            score, matched, missing, extra, role, pdf_name,
        )

    resumes_data.sort(key=lambda row: row[1], reverse=True)
    return resumes_data


@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    global matched_results
    matched_results = []

    if request.method == 'POST':
        mode = request.form.get("mode")
        skills_dict = load_skills()
        role_skill_map = load_role_map()

        if mode == "recruiter":
            jd_file = request.files.get("jd")
            resume_files = request.files.getlist("resumes")

            if not jd_file or not resume_files:
                return render_template("index.html", error="Please upload JD and resumes.")

            matched_results = process_recruiter_mode(jd_file, resume_files, skills_dict, role_skill_map)
            return render_template("results.html", results=matched_results, mode=mode)

        if mode == "candidate":
            jd_files = request.files.getlist("jds")
            resume_file = request.files.get("resume")

            if not jd_files or not resume_file:
                return render_template("index.html", error="Please upload a resume and multiple JDs.")

            matched_results = process_candidate_mode(resume_file, jd_files, skills_dict, role_skill_map)
            return render_template("results.html", results=matched_results, mode=mode)

    return render_template("index.html")


@app.route('/download')
@login_required
def download():
    global matched_results
    df = pd.DataFrame(matched_results, columns=[
        "Resume", "Score", "Matched Skills", "Missing Skills",
        "Extra Skills", "Suggestions", "Inferred Role", "PDF Name", "JD Tag"
    ])
    df["Score (%)"] = (df["Score"] * 100).round(2)
    df["Matched Skills"] = df["Matched Skills"].apply(lambda x: ", ".join(x))
    df["Missing Skills"] = df["Missing Skills"].apply(lambda x: ", ".join(x))
    df["Extra Skills"] = df["Extra Skills"].apply(lambda x: ", ".join(x))
    df["Suggestions"] = df["Suggestions"].apply(lambda x: "; ".join(x))
    df.drop(columns=["Score", "PDF Name"], inplace=True)
    df.to_csv("results.csv", index=False)
    return send_file("results.csv", as_attachment=True, download_name="resume_match_results.csv")


@app.route('/summary/<filename>')
@login_required
def serve_summary(filename):
    pdf_path = os.path.join(summaries, filename)
    if os.path.exists(pdf_path):
        return send_file(pdf_path, as_attachment=True)
    return "Summary PDF not found.", 404


@app.route('/history')
@login_required
def history():
    user_history = load_user_history(current_user.id)
    return render_template('history.html', history=user_history)


@app.route('/download_all_pdfs')
@login_required
def download_all_pdfs():
    pdf_names = [row[7] for row in matched_results if row[7].endswith('.pdf')]
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for pdf_name in pdf_names:
            pdf_path = os.path.join(summaries, pdf_name)
            if os.path.exists(pdf_path):
                zipf.write(pdf_path, arcname=pdf_name)
    zip_buffer.seek(0)
    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='all_summaries.zip'
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
