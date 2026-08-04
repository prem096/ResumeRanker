from flask import Flask, render_template, request, send_file
from utils.parser import extract_text_from_pdf
from utils.processor import compute_similarity
from utils.pdf_generator import generate_pdf_summary
from utils.skill_extractor import (
    load_skills, compare_skills, extract_skills,
    generate_suggestions, infer_job_role
)
from utils.history_tracker import save_history, load_user_history
import os
import pandas as pd
import json
import uuid
from werkzeug.utils import secure_filename
from extensions import db, login_manager
from flask_login import current_user, login_required
from auth import auth
import zipfile
import io

from auth.models import User, MatchHistory 

app = Flask(__name__)
app.secret_key = "410f653a05a8c3bb10e128a79f63c4650e41660d8e89c7515423980327690de3"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = "auth.login"

from auth.models import User  # ✅ Place here to avoid circular import

uploads = "uploads"
summaries = "summaries"

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Register blueprints
app.register_blueprint(auth)

# Ensure folders exist
os.makedirs(uploads, exist_ok=True)
os.makedirs(summaries, exist_ok=True)

matched_results = []

def load_role_map(filepath="utils/role_skills.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jd_categories(filepath="utils/jd_categories.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def tag_jd(jd_text, category_map=load_jd_categories()):
    jd_text_lower = jd_text.lower()
    scores = {cat: sum(kw in jd_text_lower for kw in kws) for cat, kws in category_map.items()}
    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] > 0 else "Other"

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

            jd_filename = secure_filename(jd_file.filename)
            jd_path = os.path.join(uploads, jd_filename)
            jd_file.save(jd_path)
            jd_text = extract_text_from_pdf(jd_path)

            jd_tag = tag_jd(jd_text)

            resumes_data = []
            for resume_file in resume_files:
                resume_filename = secure_filename(resume_file.filename)
                resume_path = os.path.join(uploads, resume_filename)
                resume_file.save(resume_path)
                resume_text = extract_text_from_pdf(resume_path)

                score = compute_similarity(jd_text, [(resume_filename, resume_text)])[0][1]
                matched, missing, extra = compare_skills(jd_text, resume_text, skills_dict)
                suggestions = generate_suggestions(missing)
                resume_skills = extract_skills(resume_text, skills_dict)
                role = infer_job_role(resume_skills, role_skill_map)

                pdf_path = generate_pdf_summary(resume_filename, jd_filename, score, matched, missing, extra, suggestions, role)
                pdf_name = os.path.basename(pdf_path)

                resumes_data.append((resume_filename, float(score), matched, missing, extra, suggestions, role, pdf_name, jd_tag))

                # ✅ Save history
                save_history(
                    current_user.id,
                    mode,
                    resume_filename,
                    jd_filename,
                    score,
                    matched,
                    missing,
                    extra,
                    role,
                    pdf_name
                )

            matched_results = resumes_data
            return render_template("results.html", results=matched_results, mode=mode)

        elif mode == "candidate":
            jd_files = request.files.getlist("jds")
            resume_file = request.files.get("resume")

            if not jd_files or not resume_file:
                return render_template("index.html", error="Please upload a resume and multiple JDs.")

            resume_filename = secure_filename(resume_file.filename)
            resume_path = os.path.join(uploads, resume_filename)
            resume_file.save(resume_path)
            resume_text = extract_text_from_pdf(resume_path)

            jd_tag = tag_jd(resume_text)

            resumes_data = []
            for jd_file in jd_files:
                jd_filename = secure_filename(jd_file.filename)
                jd_path = os.path.join(uploads, jd_filename)
                jd_file.save(jd_path)
                jd_text = extract_text_from_pdf(jd_path)

                score = compute_similarity(jd_text, [(resume_filename, resume_text)])[0][1]
                matched, missing, extra = compare_skills(jd_text, resume_text, skills_dict)
                suggestions = generate_suggestions(missing)
                resume_skills = extract_skills(resume_text, skills_dict)
                role = infer_job_role(resume_skills, role_skill_map)

                pdf_path = generate_pdf_summary(resume_filename, jd_filename, score, matched, missing, extra, suggestions, role)
                pdf_name = os.path.basename(pdf_path)

                resumes_data.append((jd_filename, float(score), matched, missing, extra, suggestions, role, pdf_name, jd_tag))

                save_history(
                    current_user.id,
                    mode,
                    resume_filename,
                    jd_filename,
                    score,
                    matched,
                    missing,
                    extra,
                    role,
                    pdf_name
                )

            matched_results = resumes_data
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
